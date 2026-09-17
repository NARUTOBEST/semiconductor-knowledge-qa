# -*- coding: utf-8 -*-
"""短期对话流水的本地 WAL(Write-Ahead Log)暂存与有界回填。

解决的问题:短期流水(mem:* 事件)以 Redis 为唯一后端,Redis 短暂不可达(宕机切换/
重启/网络分区/半挂)时 ``append_event`` 直写会失败。为做到"对话流水无论如何都不丢",
直写失败的事件先【追加到本地 WAL 文件】,待 Redis 恢复后由记忆链兜底节点
(memory_loop/resilience)调用 ``drain`` 幂等推回 Redis。

设计:
  - 单文件 append-only JSONL(每行一条紧凑 JSON,不含换行),O_APPEND 追加;
    可选 fsync(MEM_SPOOL_FSYNC,默认开),进程被 kill 也不丢已落盘行。
  - 回填有界:每次 drain 受 max_items(条数)与 time_budget_s(时间盒)约束,
    处理不完的条目原样保留,由后续请求/下一轮继续,绝不拖慢 emit_done。
  - 清理原子:drain 后把"未成功/未处理"的行用【临时文件 + fsync + os.replace】
    重写(同 session_file 的原子替换范式),进程随时崩溃都不会留下半截 WAL。
  - 坏行容错:单行 JSON 损坏只跳过该行并计数,不阻塞其余回填。
  - 容量保护:WAL 超过 MEM_SPOOL_MAX_MB 时丢弃最老的一半并 error 告警,防磁盘打满。

线程安全:进程内一把锁串行化 append/drain(同一后端进程内多请求并发)。
本模块不直接依赖 redis/redis-py,回填动作由调用方传入 write_one 回调,便于单测。
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
_FILENAME = "events.wal.jsonl"
# processing 标记的陈旧阈值在 short_term 侧;此处只关心文件。


def _enabled() -> bool:
    return os.getenv("MEM_SPOOL_ENABLED", "1") not in ("0", "false", "False", "")


def _spool_dir() -> str:
    """WAL 目录。默认与 session-memory.md 同数据根 memories_data/spool,便于统一挂卷。"""
    env = str(os.getenv("MEM_SPOOL_DIR", "") or "").strip()
    if env:
        return env
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
    return os.path.join(project_root, "memories_data", "spool")


def _wal_path() -> str:
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
                    if isinstance(rec, dict) and rec.get("eid"):
                        records.append(rec)
                    else:
                        bad += 1
                except Exception:  # noqa: BLE001
                    bad += 1
    except FileNotFoundError:
        return [], 0
    return records, bad


def _atomic_rewrite(path: str, records: list[dict]) -> None:
    """把剩余记录原子重写回 WAL;records 为空则删除 WAL。"""
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    if not records:
        with contextlib.suppress(OSError):
            os.unlink(path)
        return
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp-wal-", suffix=".jsonl")
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
    """WAL 超体积上限时丢弃最老一半(只在 append 前调用),防磁盘被打满。"""
    try:
        if not os.path.exists(path) or os.path.getsize(path) <= _max_bytes():
            return
        records, _bad = _read_lines(path)
        keep = records[len(records) // 2:]
        logger.error("memory WAL 超过体积上限,丢弃最老 %d 条(保留 %d 条)",
                     len(records) - len(keep), len(keep))
        _atomic_rewrite(path, keep)
    except Exception as e:  # noqa: BLE001
        logger.info("WAL cap enforce skipped: %s: %s", type(e).__name__, str(e)[:120])


def append_record(record: dict) -> bool:
    """把一条事件追加到本地 WAL。成功 True;开关关闭/落盘失败 False(调用方仅记日志)。"""
    if not _enabled() or not record.get("eid"):
        return False
    path = _wal_path()
    line = _dump_line(record) + "\n"
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
        except Exception as e:  # noqa: BLE001  本地盘也不可写:最后兜底,只告警
            logger.error("memory WAL append failed: %s: %s",
                         type(e).__name__, str(e)[:120])
            return False


def pending_count() -> int:
    """WAL 中待回填条数(轻量);无文件返回 0。"""
    path = _wal_path()
    with _LOCK:
        records, _ = _read_lines(path)
        return len(records)


def drain(write_one: Callable[[dict], bool], *,
          max_items: Optional[int] = None,
          time_budget_s: Optional[float] = None) -> dict:
    """把 WAL 中的记录用 write_one 逐条推回 Redis。

    :param write_one: 回调,入参一条 record,返回 True=已落 Redis(成功或幂等跳过),
                      False/抛异常=本次失败、保留到 WAL 下轮再试。
    :param max_items: 单次最多处理条数(默认 MEM_SPOOL_DRAIN_LIMIT=100)。
    :param time_budget_s: 单次处理时间盒(秒,默认 MEM_SPOOL_DRAIN_BUDGET=0.2),
                      到点后剩余条目保留,下轮继续。
    :returns: {"processed", "sent", "remain", "bad"}。
    """
    if max_items is None:
        try:
            max_items = max(1, int(os.getenv("MEM_SPOOL_DRAIN_LIMIT", "100")))
        except Exception:  # noqa: BLE001
            max_items = 100
    if time_budget_s is None:
        try:
            time_budget_s = max(0.0, float(os.getenv("MEM_SPOOL_DRAIN_BUDGET", "0.2")))
        except Exception:  # noqa: BLE001
            time_budget_s = 0.2

    path = _wal_path()
    result = {"processed": 0, "sent": 0, "remain": 0, "bad": 0}
    with _LOCK:
        records, bad = _read_lines(path)
        result["bad"] = bad
        if bad:
            logger.info("memory WAL 跳过 %d 条损坏行", bad)
        if not records:
            # 清理可能存在的空文件
            with contextlib.suppress(OSError):
                if os.path.exists(path):
                    os.unlink(path)
            return result

        start = time.monotonic()
        kept: list[dict] = []
        processed = 0
        for rec in records:
            # 达到条数/时间盒:本条之后全部原样保留
            if processed >= max_items or \
                    (time_budget_s and time.monotonic() - start > time_budget_s):
                kept.append(rec)
                continue
            processed += 1
            try:
                ok = bool(write_one(rec))
            except Exception as e:  # noqa: BLE001
                ok = False
                logger.info("WAL replay one failed: %s: %s",
                            type(e).__name__, str(e)[:120])
            if ok:
                result["sent"] += 1
            else:
                kept.append(rec)  # 本轮回填失败,保留
        result["processed"] = processed
        result["remain"] = len(kept)
        try:
            _atomic_rewrite(path, kept)
        except Exception as e:  # noqa: BLE001  重写失败不致命:宁可下次重复处理(幂等)
            logger.info("WAL rewrite skipped: %s: %s", type(e).__name__, str(e)[:120])
        if result["sent"]:
            logger.info("memory WAL drained: sent=%d remain=%d",
                        result["sent"], result["remain"])
    return result
