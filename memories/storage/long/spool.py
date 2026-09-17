# -*- coding: utf-8 -*-
"""长期记忆升迁数据的本地 spool(仿 short/event_spool 的 WAL 范式)。

解决的问题:升迁门判定达标后的偏好写入依赖 PG(pgvector)。PG 短暂不可达时,
直接丢弃会永久丢失该批长期偏好。故在 PG 不可用时把整批 items 追加到本地
spool 文件,待 PG 恢复后由记忆兜底维护(memory_loop/resilience)有界重放。

设计(与 short/event_spool 同范式):
  - 单文件 append-only JSONL,O_APPEND 追加 + 可选 fsync;
  - 重放有界(条数 + 时间盒),重放失败的记录保留到下轮(重放依赖
    upsert_memory 的结构化/语义去重,天然幂等,重复执行无害);
  - 清理原子:重放后剩余记录用【临时文件 + fsync + os.replace】重写;
  - 容量保护:超 MEM_SPOOL_MAX_MB 丢弃最老一半(复用 event_spool 的体积上限)。

与"降级裸写短期事实表(degraded=True)"的关系:裸写只保证对话内容不丢、
永不回流长期表;spool 里的数据在 PG 恢复后真正完成升迁,是更完整的兜底。
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
import threading
import time
from typing import Callable, Optional

logger = logging.getLogger("agent")

_LOCK = threading.Lock()
_FILENAME = "long_promo.spool.jsonl"


def _enabled() -> bool:
    return os.getenv("MEM_SPOOL_ENABLED", "1") not in ("0", "false", "False", "")


def _spool_path() -> str:
    """与短期 WAL 同一数据根(memories_data/spool),便于统一挂卷。"""
    from ...storage.short.event_spool import _spool_dir
    return os.path.join(_spool_dir(), _FILENAME)


def _fsync_enabled() -> bool:
    return os.getenv("MEM_SPOOL_FSYNC", "1") not in ("0", "false", "False")


def _max_bytes() -> int:
    try:
        return max(1, int(float(os.getenv("MEM_SPOOL_MAX_MB", "200")))) * 1024 * 1024
    except Exception:  # noqa: BLE001
        return 200 * 1024 * 1024


def _dump_line(record: dict) -> str:
    return json.dumps(record, ensure_ascii=False, default=str,
                      separators=(",", ":"))


def _read_lines(path: str) -> tuple[list[dict], int]:
    """读全部行 -> (合法记录列表, 坏行数)。文件不存在返回 ([], 0)。"""
    records: list[dict] = []
    bad = 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if isinstance(rec, dict) and rec.get("username") \
                            and isinstance(rec.get("items"), list):
                        records.append(rec)
                    else:
                        bad += 1
                except Exception:  # noqa: BLE001
                    bad += 1
    except FileNotFoundError:
        return [], 0
    return records, bad


def _atomic_rewrite(path: str, records: list[dict]) -> None:
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    if not records:
        with contextlib.suppress(OSError):
            os.unlink(path)
        return
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp-lspool-", suffix=".jsonl")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(_dump_line(rec) + "\n")
            f.flush()
            if _fsync_enabled():
                os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def _enforce_cap(path: str) -> None:
    try:
        if not os.path.exists(path) or os.path.getsize(path) <= _max_bytes():
            return
        records, _bad = _read_lines(path)
        keep = records[len(records) // 2:]
        logger.error("long-memory spool 超过体积上限,丢弃最老 %d 批(保留 %d 批)",
                     len(records) - len(keep), len(keep))
        _atomic_rewrite(path, keep)
    except Exception as e:  # noqa: BLE001
        logger.info("long spool cap enforce skipped: %s: %s",
                    type(e).__name__, str(e)[:120])


def append_record(username: str, thread_id, items: list[dict]) -> bool:
    """把一批待升迁 items 追加到本地 spool。成功 True;关闭/失败 False。"""
    if not _enabled() or not username or not items:
        return False
    path = _spool_path()
    line = _dump_line({"username": username, "thread_id": thread_id,
                       "items": items, "ts": time.time()}) + "\n"
    with _LOCK:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            _enforce_cap(path)
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
                if _fsync_enabled():
                    os.fsync(f.fileno())
            return True
        except Exception as e:  # noqa: BLE001  本地盘也不可写:只告警
            logger.error("long-memory spool append failed: %s: %s",
                         type(e).__name__, str(e)[:120])
            return False


def pending_count() -> int:
    path = _spool_path()
    with _LOCK:
        records, _ = _read_lines(path)
        return len(records)


def drain(replay_one: Callable[[dict], bool], *,
          max_items: Optional[int] = None,
          time_budget_s: Optional[float] = None) -> dict:
    """把 spool 中的记录逐批用 replay_one 重放进 PG。

    :param replay_one: 入参一条 record,返回 True=已写入 PG,False/抛异常=保留下轮再试。
    :param max_items: 单次最多处理批数(默认 MEM_SPOOL_DRAIN_LIMIT)。
    :param time_budget_s: 时间盒(默认 MEM_SPOOL_DRAIN_BUDGET)。
    :returns: {"processed", "sent", "remain", "bad"}。
    """
    if max_items is None:
        try:
            max_items = max(1, int(os.getenv("MEM_SPOOL_DRAIN_LIMIT", "20")))
        except Exception:  # noqa: BLE001
            max_items = 20
    if time_budget_s is None:
        try:
            time_budget_s = max(0.0, float(os.getenv("MEM_SPOOL_DRAIN_BUDGET", "1.5")))
        except Exception:  # noqa: BLE001
            time_budget_s = 1.5

    path = _spool_path()
    result = {"processed": 0, "sent": 0, "remain": 0, "bad": 0}
    with _LOCK:
        records, bad = _read_lines(path)
        result["bad"] = bad
        if bad:
            logger.info("long-memory spool 跳过 %d 条损坏行", bad)
        if not records:
            with contextlib.suppress(OSError):
                if os.path.exists(path):
                    os.unlink(path)
            return result

        start = time.monotonic()
        kept: list[dict] = []
        processed = 0
        for rec in records:
            if processed >= max_items or \
                    (time_budget_s and time.monotonic() - start > time_budget_s):
                kept.append(rec)
                continue
            processed += 1
            try:
                ok = bool(replay_one(rec))
            except Exception as e:  # noqa: BLE001
                ok = False
                logger.info("long spool replay failed: %s: %s",
                            type(e).__name__, str(e)[:120])
            if ok:
                result["sent"] += 1
            else:
                kept.append(rec)
        result["processed"] = processed
        result["remain"] = len(kept)
        try:
            _atomic_rewrite(path, kept)
        except Exception as e:  # noqa: BLE001  重写失败不致命:重放幂等,下次再来
            logger.info("long spool rewrite skipped: %s: %s",
                        type(e).__name__, str(e)[:120])
        if result["sent"]:
            logger.info("long-memory spool drained: sent=%d remain=%d",
                        result["sent"], result["remain"])
    return result
