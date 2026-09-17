# -*- coding: utf-8 -*-
"""记忆【欠账补做 spool】(turn 级):事实表/升迁门故障期的本地暂存与有界重放。

解决的问题(两类欠账,记录格式相同,重放动作也相同):
  1. Redis 不可达 —— 短期事实表(memf:*)写不了,本轮 Q+A 的"去重落库 + 升迁判定"
     整体欠账(事件流水另有 event_spool WAL,两者互补);
  2. 升迁门 LLM 失败 —— 重试/熔断走完后裸写降级(degraded=1 只保证内容短期可见,
     永不回流长期表),本轮的"长期偏好判定"欠账。

记录 = {username, thread_id, q, a, ts};重放 = 补跑完整沉淀
(memory_loop/consolidate.replay_turn:指纹去重 -> 升迁门 LLM -> 落事实 ->
标记 promoted)。对已裸写的 degraded 事实,重放经指纹去重 touch 后原地升级
promoted=1,不产生重复条目。

设计(与 short/event_spool、long/spool 同范式):
  - 单文件 append-only JSONL,O_APPEND 追加 + 可选 fsync(MEM_SPOOL_FSYNC);
  - 重放有界(条数 + 时间盒),失败/熔断期的记录保留到下轮;重放幂等
    (指纹去重 + PG 语义去重,重复执行无害);
  - 清理原子:临时文件 + fsync + os.replace;超 MEM_SPOOL_MAX_MB 丢最老一半。
  - 重放每条可能调一次升迁门 LLM(秒级),故默认条数限制比事件 WAL 紧得多。

依赖方向:本模块属 storage 层,不 import orchestration;重放回调由调用方
(memory_loop/resilience)注入 replay_turn,便于单测。
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
_FILENAME = "turn_work.spool.jsonl"


def _enabled() -> bool:
    return os.getenv("MEM_SPOOL_ENABLED", "1") not in ("0", "false", "False", "")


def _spool_path() -> str:
    """与短期 WAL 同一数据根(memories_data/spool),便于统一挂卷。"""
    from .short.event_spool import _spool_dir
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
                            and (rec.get("q") or rec.get("a")):
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
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp-wspool-", suffix=".jsonl")
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
        logger.error("turn work spool 超过体积上限,丢弃最老 %d 条(保留 %d 条)",
                     len(records) - len(keep), len(keep))
        _atomic_rewrite(path, keep)
    except Exception as e:  # noqa: BLE001
        logger.info("turn work spool cap enforce skipped: %s: %s",
                    type(e).__name__, str(e)[:120])


def append_record(username: str, thread_id, q: str, a: str) -> bool:
    """把一轮 Q+A 的记忆欠账追加到本地 spool。成功 True;关闭/失败 False。"""
    if not _enabled() or not username or not (q or a):
        return False
    path = _spool_path()
    line = _dump_line({"username": username, "thread_id": thread_id,
                       "q": q, "a": a, "ts": time.time()}) + "\n"
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
            logger.error("turn work spool append failed: %s: %s",
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
    """把欠账记录逐条用 replay_one 重放。

    :param replay_one: 入参一条 record,返回 True=已完成,False/抛异常=保留到下轮。
    :param max_items: 单次最多处理条数(默认 MEM_WORK_SPOOL_DRAIN_LIMIT;
                      每条可能调一次升迁门 LLM,刻意比事件 WAL 的 100 小)。
    :param time_budget_s: 时间盒(默认 MEM_WORK_SPOOL_DRAIN_BUDGET)。
    :returns: {"processed", "sent", "remain", "bad"}。
    """
    if max_items is None:
        try:
            max_items = max(1, int(os.getenv("MEM_WORK_SPOOL_DRAIN_LIMIT", "3")))
        except Exception:  # noqa: BLE001
            max_items = 3
    if time_budget_s is None:
        try:
            time_budget_s = max(0.0,
                                float(os.getenv("MEM_WORK_SPOOL_DRAIN_BUDGET", "8.0")))
        except Exception:  # noqa: BLE001
            time_budget_s = 8.0

    path = _spool_path()
    result = {"processed": 0, "sent": 0, "remain": 0, "bad": 0}
    with _LOCK:
        records, bad = _read_lines(path)
        result["bad"] = bad
        if bad:
            logger.info("turn work spool 跳过 %d 条损坏行", bad)
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
                logger.info("turn work spool replay failed: %s: %s",
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
            logger.info("turn work spool rewrite skipped: %s: %s",
                        type(e).__name__, str(e)[:120])
        if result["sent"]:
            logger.info("turn work spool drained: sent=%d remain=%d",
                        result["sent"], result["remain"])
    return result
