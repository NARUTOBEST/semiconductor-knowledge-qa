# -*- coding: utf-8 -*-
"""记忆链【兜底维护】纯逻辑:由 mem_consolidate 节点开头调用(无独立节点)。

每轮记忆图第一步先跑本维护(不挑门控,匿名/关闭记忆也会经过),
只做"故障恢复后把欠账补回来"的维护,不参与作答、不阻塞首字节:

  1) 短期流水 WAL 回填 —— Redis 故障期落本地(event_spool)的对话流水,在 Redis
     恢复后(redis_ready_fast 快探通过)有界、幂等推回 Redis;快探不通过则 0ms 跳过。
  2) 长期 NULL 向量 backfill —— 检索模型故障期以 embedding IS NULL 落库、因而永远
     召不回的语义事实,在模型恢复后重新向量化补回(扫不到 NULL 时不调用模型,零成本)。
  3) 长期升迁 spool 重放 —— PG 故障期暂存本地(long/spool)的升迁批,在 PG 恢复后
     有界重放(ping 探测,不可用则 0ms 跳过)。
  4) 记忆欠账补做(work_spool)—— Redis 故障期的事实表欠账 + 升迁门 LLM 故障期的
     判定欠账(裸写降级轮),在 Redis 恢复后逐条补跑完整沉淀(去重/升迁/落库);
     每条可能调一次升迁门 LLM,条数/时间盒刻意收紧。

各步都严格有界(条数 + 时间盒)、best-effort:单轮只推进一小批,剩余留给后续请求;
任何异常只记 info、绝不抛到图里影响 emit_done。
"""
from __future__ import annotations

import logging
import os
import sys

_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "config"))
import config as C  # noqa: E402

logger = logging.getLogger("agent")

_DISABLED = ("0", "false", "False", "")


def _enabled(key: str, default: str = "1") -> bool:
    return os.getenv(key, default) not in _DISABLED


def _env_int(key: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(key, str(default))))
    except Exception:  # noqa: BLE001
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return max(0.0, float(os.getenv(key, str(default))))
    except Exception:  # noqa: BLE001
        return default


def _drain_wal(wal_limit, wal_budget) -> dict:
    """短期流水 WAL 回填;Redis 快探不通过时直接跳过(冷却期 0ms)。"""
    if not _enabled("MEM_SPOOL_DRAIN_ENABLED"):
        return {"skipped": "disabled"}
    try:
        from memories.storage.connections import redis_ready_fast
        from memories.storage.short.short_term import short_term
        if not redis_ready_fast():
            return {"skipped": "redis_down"}
        return short_term.drain_spooled(max_items=wal_limit, time_budget_s=wal_budget)
    except Exception as e:  # noqa: BLE001
        logger.info("resilience wal-drain skipped: %s: %s",
                    type(e).__name__, str(e)[:120])
        return {"error": f"{type(e).__name__}:{str(e)[:80]}"}


def _backfill_embeddings(bf_limit, bf_budget) -> dict:
    """长期 NULL 向量回填;长期记忆关闭/缺依赖/模型不可用时安全跳过。"""
    if not _enabled("MEM_BACKFILL_ENABLED"):
        return {"skipped": "disabled"}
    if not getattr(C, "LONG_MEM_ENABLED", True):
        return {"skipped": "long_mem_disabled"}
    try:
        from memories.storage.long.long_term import long_term
        from memories.storage.long.embed import embed_texts
        return long_term.backfill_missing_embeddings(
            embed_texts,
            limit=bf_limit or _env_int("MEM_BACKFILL_LIMIT", 8),
            time_budget_s=bf_budget if bf_budget is not None
            else _env_float("MEM_BACKFILL_BUDGET", 1.5))
    except Exception as e:  # noqa: BLE001  缺 psycopg/驱动等:旁路不可用,不致命
        logger.info("resilience embedding-backfill skipped: %s: %s",
                    type(e).__name__, str(e)[:120])
        return {"error": f"{type(e).__name__}:{str(e)[:80]}"}


def _replay_pg_spool() -> dict:
    """长期升迁 spool 重放;PG 快探不通过/长期记忆关闭时直接跳过(0ms)。"""
    try:
        from ..long import extract
        return extract.replay_spooled()
    except Exception as e:  # noqa: BLE001  缺 psycopg/驱动等:旁路不可用,不致命
        logger.info("resilience pg-spool replay skipped: %s: %s",
                    type(e).__name__, str(e)[:120])
        return {"error": f"{type(e).__name__}:{str(e)[:80]}"}


def _replay_work_spool() -> dict:
    """记忆欠账补做(事实表欠账 + 升迁门判定欠账);Redis 快探不通过时跳过。

    必须等 Redis 恢复:重放走 replay_turn(落事实依赖 Redis),Redis 仍挂时
    贸然重放会被 run 侧记成"已完成"而丢账。每条可能调一次升迁门 LLM(秒级),
    故条数/时间盒默认收紧(limit=3 / budget=8s)。
    """
    if not _enabled("MEM_WORK_SPOOL_DRAIN_ENABLED"):
        return {"skipped": "disabled"}
    try:
        from ...storage import work_spool
        from . import consolidate as CSL
        from memories.storage.connections import redis_ready_fast
        if not redis_ready_fast():
            return {"skipped": "redis_down"}
        return work_spool.drain(CSL.replay_turn)
    except Exception as e:  # noqa: BLE001
        logger.info("resilience work-spool replay skipped: %s: %s",
                    type(e).__name__, str(e)[:120])
        return {"error": f"{type(e).__name__}:{str(e)[:80]}"}


def run_resilience_maintenance(*, wal_limit: int = None,
                               wal_budget: float = None,
                               bf_limit: int = None,
                               bf_budget: float = None) -> dict:
    """兜底维护入口(后台记忆管道调用)。返回各步计数,异常不外抛。"""
    stats = {"wal": None, "backfill": None, "pg_spool": None, "work_spool": None}
    if not _enabled("MEM_RESILIENCE_ENABLED"):
        return {"wal": {"skipped": "disabled"},
                "backfill": {"skipped": "disabled"},
                "pg_spool": {"skipped": "disabled"},
                "work_spool": {"skipped": "disabled"}}
    stats["wal"] = _drain_wal(wal_limit, wal_budget)
    stats["backfill"] = _backfill_embeddings(bf_limit, bf_budget)
    stats["pg_spool"] = _replay_pg_spool()
    stats["work_spool"] = _replay_work_spool()
    return stats


__all__ = ["run_resilience_maintenance"]
