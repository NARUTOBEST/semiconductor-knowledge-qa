# -*- coding: utf-8 -*-
"""会话级摘要文件 session-memory.md 的磁盘存储(节点二 Auto-Compact 用)。

- 每会话一个文件:<MEM_SESSION_DIR>/<username>/<safe>.session-memory.md。
  文件名用 scoped_thread_id 的 blake2b 短哈希(防路径穿越/特殊字符),按用户分目录,
  账号注销时整目录 rmtree。
- 文件 = YAML-lite front-matter(游标/元数据)+ 9 章节结构化摘要正文。
- 并发安全:同目录 ``<file>.lock`` 用 O_CREAT|O_EXCL 自旋锁(跨平台、无新依赖);
  写入走 "临时文件 + fsync + os.replace" 原子替换,读方永远看不到半成品。

游标(cursor_msg_id / last_tokens / last_tool_calls / version)权威存文件 front-matter,
state 字段仅作节点间传递/镜像。旁路:文件系统不可用时调用方降级(规则截断/不 compact)。
"""
from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import tempfile
import threading
import time
from typing import Any, Optional

logger = logging.getLogger("agent")

_FM_START = "---"
_FM_END = "---"
_LOCK_STALE_SECONDS = 60.0   # 锁文件超过该年龄视为进程崩溃遗留,可抢占
_FILE_SUFFIX = ".session-memory.md"
_REENTRANT = threading.local()   # .locks: {lock_path: 深度} 本线程已持有的锁(重入计数)

# 9 章节结构化摘要模板(附则)
NINE_SECTION_TITLES = [
    "会话目标",
    "已达成结论 / 决策",
    "用户身份与偏好",
    "关键事实与数据",
    "进行中任务与状态",
    "使用过的工具与副作用",
    "未解决问题 / 待办",
    "风险与约束",
    "下一步建议",
]


def empty_template_body() -> str:
    """9 章节空骨架(首次摘要 LLM 失败/规则兜底时用)。"""
    return "\n".join(f"## {i+1}. {t}\n" for i, t in enumerate(NINE_SECTION_TITLES))


def _session_dir() -> str:
    import config as C
    return str(getattr(C, "MEM_SESSION_DIR", "") or os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")),
        "memories_data", "sessions"))


def _lock_timeout() -> float:
    import config as C
    return float(getattr(C, "MEM_FILE_LOCK_TIMEOUT", 5.0))


def session_path(username: Optional[str], thread_id: str) -> str:
    """返回会话摘要文件绝对路径(不创建)。按用户名分目录,文件名取 scoped id 哈希。"""
    safe = hashlib.blake2b((thread_id or "").encode("utf-8", "ignore"),
                          digest_size=12).hexdigest()
    user_dir = username if username else "_anonymous"
    return os.path.join(_session_dir(), user_dir, safe + _FILE_SUFFIX)


@contextlib.contextmanager
def file_lock(path: str, timeout: Optional[float] = None):
    """跨平台自旋文件锁:同目录 ``<path>.lock`` 用 O_CREAT|O_EXCL 独占创建。

    超时(默认 MEM_FILE_LOCK_TIMEOUT)抛 TimeoutError;锁文件超 _LOCK_STALE_SECONDS
    视为崩溃遗留,抢占删除后重试。
    同线程可重入(嵌套时直接放行、不重复建锁文件)——整周期锁内再调 write()
    (其内部也加同一把锁)不会自等超时。
    """
    timeout = _lock_timeout() if timeout is None else timeout
    lock_path = path + ".lock"
    held = getattr(_REENTRANT, "locks", None)
    if held is None:
        held = _REENTRANT.locks = {}
    if held.get(lock_path):
        held[lock_path] += 1
        try:
            yield
        finally:
            held[lock_path] -= 1
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd = None
    deadline = time.monotonic() + timeout
    try:
        while True:
            try:
                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode("utf-8"))
                break
            except FileExistsError:
                # 崩溃遗留的陈旧锁:年龄超限则抢占
                try:
                    age = time.time() - os.path.getmtime(lock_path)
                    if age > _LOCK_STALE_SECONDS:
                        os.unlink(lock_path)
                        continue
                except OSError:
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError(f"session 文件锁等待超时: {lock_path}")
                time.sleep(0.05)
        held[lock_path] = 1
        yield
    finally:
        held.pop(lock_path, None)
        if fd is not None:
            with contextlib.suppress(OSError):
                os.close(fd)
        with contextlib.suppress(OSError):
            os.unlink(lock_path)


def _atomic_write(path: str, text: str) -> None:
    """临时文件 + fsync + os.replace 原子替换(同目录 rename 原子,Win/POSIX 均可)。"""
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp-sess-", suffix=".md")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def _render(meta: dict[str, Any], body: str) -> str:
    lines = [_FM_START]
    # recent_cursor_seq:摘要/compact 落盘时记录的【短期流水高水位 seq】,与摘要游标
    # 同一处写入;recent 预取只读 seq 大于该值的事件,保证摘要与近期原文严格互斥。
    # pending_remove_ids:本轮已落盘、待 RemoveMessage 的消息 id 列表(Req6 幂等)。
    for k in ("cursor_msg_id", "cursor_index", "last_tokens",
              "last_tool_calls", "recent_cursor_seq", "pending_remove_ids",
              "version", "updated_at"):
        v = meta.get(k, "")
        if isinstance(v, (list, tuple)):
            v = ",".join(str(x) for x in v)   # 待删 id 列表 -> 逗号串
        lines.append(f"{k}: {v}")
    lines.append(_FM_END)
    lines.append("")
    lines.append(body.rstrip())
    lines.append("")
    return "\n".join(lines)


def _parse(text: str) -> tuple[dict[str, Any], str]:
    """解析 front-matter + 正文。损坏/无 front-matter 时按空 meta、全文正文处理。"""
    if not text:
        return {}, ""
    ls = text.splitlines()
    if not ls or ls[0].strip() != _FM_START:
        return {}, text
    meta: dict[str, Any] = {}
    i = 1
    while i < len(ls) and ls[i].strip() != _FM_END:
        line = ls[i]
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
        i += 1
    body = "\n".join(ls[i + 1:]).lstrip("\n")
    return meta, body


def _to_int(v, default: int = 0) -> int:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


def _parse_id_list(v) -> list[str]:
    """front-matter 的逗号分隔 id 列表 -> [str];空/损坏返回 []。"""
    if isinstance(v, (list, tuple)):
        return [str(x).strip() for x in v if str(x).strip()]
    if not v:
        return []
    return [p.strip() for p in str(v).split(",") if p.strip()]


class SessionFile:
    """session-memory.md 读写(线程/多进程安全由 file_lock 保证)。"""

    def __init__(self, username: Optional[str], thread_id: str):
        self.username = username
        self.thread_id = thread_id
        self.path = session_path(username, thread_id)

    def exists(self) -> bool:
        return os.path.exists(self.path)

    def read(self) -> tuple[dict[str, Any], str]:
        """返回 (meta, body)。文件不存在/损坏返回 ({}, "")。"""
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return _parse(f.read())
        except FileNotFoundError:
            return {}, ""
        except Exception as e:  # noqa: BLE001  损坏按空处理(下轮重建)
            logger.info("session_file read failed: %s: %s",
                        type(e).__name__, str(e)[:120])
            return {}, ""

    def read_meta_typed(self) -> dict[str, Any]:
        """读 front-matter 并把数值字段转 int(缺失给默认)。"""
        meta, _ = self.read()
        return {
            "cursor_msg_id": str(meta.get("cursor_msg_id", "") or ""),
            "cursor_index": _to_int(meta.get("cursor_index"), 0),
            "last_tokens": _to_int(meta.get("last_tokens"), 0),
            "last_tool_calls": _to_int(meta.get("last_tool_calls"), 0),
            # 短期流水高水位 seq(Req1 游标互斥);逗号分隔的待删消息 id(Req6 幂等)。
            "recent_cursor_seq": _to_int(meta.get("recent_cursor_seq"), 0),
            "pending_remove_ids": _parse_id_list(meta.get("pending_remove_ids", "")),
            "version": _to_int(meta.get("version"), 0),
        }

    def lock(self, timeout: Optional[float] = None):
        """整周期互斥锁(读→处理→写全程持有),线程/多进程均有效。

        write() 内部已隐式加同一把锁(同线程重入放行);需要在"读游标→LLM→写"
        整段防并发覆盖时,用本方法显式包住全周期。持锁可跨 LLM 调用,等锁超时
        抛 TimeoutError,由调用方按降级处理。
        """
        return file_lock(self.path, timeout=timeout)

    def write(self, meta: dict[str, Any], body: str) -> None:
        """加锁 + 原子写。"""
        with file_lock(self.path):
            _atomic_write(self.path, _render(meta, body))

    def delete(self) -> None:
        with contextlib.suppress(OSError):
            os.unlink(self.path)
        with contextlib.suppress(OSError):
            os.unlink(self.path + ".lock")


def delete_user_files(username: str) -> None:
    """账号注销:整目录 rmtree(该用户全部会话摘要文件)。"""
    import shutil
    user_dir = os.path.join(_session_dir(), username or "_anonymous")
    with contextlib.suppress(OSError):
        shutil.rmtree(user_dir, ignore_errors=True)
