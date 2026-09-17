# -*- coding: utf-8 -*-
"""记忆工作管道:后台串行消费每轮的记忆维护任务。

职责(答案定稿后由主链路非阻塞 submit,全程不影响主链路):
  1. 在 daemon worker 里 invoke 独立记忆图(graph.build_memory_graph)——
     记忆沉淀(短期事实 + 长期升迁)、阈值触发的会话摘要/compact、兜底维护;
  2. compact 产出的 RemoveMessage 经调用方注入的 compact_applier 直接落到
     会话 checkpoint(两轮之间执行,避开与运行中图的 checkpoint 写并发);
  3. 提供 wait_idle 入口等待门:同一会话的下一轮请求在入口等待上一轮记忆
     处理完成(有界超时,超时放行——未完成的幂等重做留给下一轮)。

并发模型:进程内单 worker + 全局 FIFO + 跨进程会话锁(session_lock)。
同一会话的任务因 FIFO 天然串行;多 worker 部署(WORKERS>1)时,执行前再抢
Redis 会话锁(memlock:*,见 session_lock),同一会话的记忆任务在全部 worker
间全局互斥——抢不到就把任务重排队尾稍后再试(单 worker 模型不阻塞其他会话)。
Redis 不可用时锁 fail-open,退化为进程内 FIFO 串行(单 worker 时代语义)。
跨会话任务排队互不阻塞;极端积压由 wait_idle 的超时兜底,主链路永不被记忆拖死。
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any, Callable, Optional

import config as C

from . import session_lock
from .graph import build_memory_graph

logger = logging.getLogger("agent")

# 抢不到跨进程锁时的重试间隔(任务重排队尾后小睡,避免空转;远小于 wait_idle 超时)
_LOCK_RETRY_SLEEP = 0.25

logger = logging.getLogger("agent")


class MemoryJob:
    """一轮的记忆维护任务(原料快照;daemon 持有,与请求生命周期解耦)。"""

    __slots__ = ("username", "thread_id", "question", "answer",
                 "messages", "final_reason", "compact_applier")

    def __init__(self, *, username: Optional[str], thread_id: Optional[str],
                 question: str = "", answer: str = "",
                 messages: Optional[list] = None,
                 final_reason: str = "answer",
                 compact_applier: Optional[Callable[[str, list], None]] = None):
        self.username = username
        self.thread_id = thread_id
        self.question = question or ""
        self.answer = answer or ""
        self.messages = list(messages or [])
        self.final_reason = final_reason or "answer"
        # compact_applier(store_thread_id, remove_messages):由 ReAct 侧注入,
        # 把摘要节点产出的 RemoveMessage 应用到会话 checkpoint。
        self.compact_applier = compact_applier

    def key(self) -> tuple[str, str]:
        return (self.username or "", self.thread_id or "")


class MemoryPipeline:
    """按会话串行的后台记忆管道(进程内单例,见 get_pipeline)。"""

    def __init__(self):
        self._cv = threading.Condition()
        self._q: deque[MemoryJob] = deque()
        self._running_key: Optional[tuple[str, str]] = None
        self._worker: Optional[threading.Thread] = None

    # ---------------- 提交 ----------------
    def submit(self, job: MemoryJob) -> None:
        """非阻塞入队;惰性启动 worker。异常不外抛(记忆旁路)。"""
        try:
            with self._cv:
                self._q.append(job)
                if self._worker is None or not self._worker.is_alive():
                    self._worker = threading.Thread(
                        target=self._loop, daemon=True, name="memory-pipeline")
                    self._worker.start()
                self._cv.notify()
        except Exception as e:  # noqa: BLE001
            logger.info("memory pipeline submit failed: %s: %s",
                        type(e).__name__, str(e)[:120])

    # ---------------- 入口等待门 ----------------
    def wait_idle(self, username: Optional[str], thread_id: Optional[str],
                  timeout: Optional[float] = None) -> bool:
        """等该会话上一轮记忆处理完成。True=已空闲;False=超时放行(兜底)。

        timeout 缺省取 MEM_WAIT_IDLE_TIMEOUT;<=0 表示不等待。
        """
        if timeout is None:
            timeout = float(getattr(C, "MEM_WAIT_IDLE_TIMEOUT", 10.0))
        if timeout <= 0:
            return True
        key = (username or "", thread_id or "")
        deadline = time.monotonic() + timeout
        with self._cv:
            while self._busy(key) or session_lock.held(key):
                rem = deadline - time.monotonic()
                if rem <= 0:
                    logger.info("memory pipeline wait_idle timeout key=%s|%s",
                                key[0], key[1])
                    return False
                # 本进程有 notify 唤醒;跨进程锁变化只能轮询感知,封顶小间隔
                self._cv.wait(min(rem, _LOCK_RETRY_SLEEP))
        return True

    def _busy(self, key: tuple[str, str]) -> bool:
        if self._running_key == key:
            return True
        return any(j.key() == key for j in self._q)

    def pending_count(self, username: Optional[str] = None,
                      thread_id: Optional[str] = None) -> int:
        """待处理任务数(测试/观测);给定时只数该会话的。"""
        with self._cv:
            if username is None and thread_id is None:
                return len(self._q) + (1 if self._running_key else 0)
            key = (username or "", thread_id or "")
            n = sum(1 for j in self._q if j.key() == key)
            if self._running_key == key:
                n += 1
            return n

    # ---------------- worker ----------------
    def _loop(self) -> None:
        while True:
            with self._cv:
                while not self._q:
                    self._cv.wait()
                job = self._q.popleft()
                self._running_key = job.key()
            token = session_lock.acquire(job.key())
            if token is None and session_lock.held(job.key()):
                # 锁被其他 worker 持有:重排队尾稍后再试,不阻塞本进程其他会话。
                # (acquire 返回 None 也可能是 Redis 不可用,此时 held 必为
                # False——直接无锁执行,即 fail-open 退化。)
                with self._cv:
                    self._running_key = None
                    self._q.append(job)
                time.sleep(_LOCK_RETRY_SLEEP)
                continue
            try:
                _execute(job)
            except Exception as e:  # noqa: BLE001  单任务失败不影响后续
                logger.info("memory pipeline job failed: %s: %s",
                            type(e).__name__, str(e)[:160])
            finally:
                session_lock.release(job.key(), token)
                with self._cv:
                    self._running_key = None
                    self._cv.notify_all()  # 唤醒 wait_idle


def _execute(job: MemoryJob) -> dict[str, Any]:
    """执行一个记忆任务:跑独立记忆图 + compact 落 checkpoint + 兜底维护。"""
    state = {
        "username": job.username or "",
        "thread_id": job.thread_id or "",
        "question": job.question,
        "full_reply": job.answer,
        "messages": job.messages,
        "final_reason": job.final_reason,
    }
    graph = build_memory_graph()
    final = graph.invoke(state)
    removes = list(final.get("remove_messages") or [])
    if removes and job.compact_applier is not None:
        try:
            job.compact_applier(job.thread_id, removes)
        except Exception as e:  # noqa: BLE001  compact 失败幂等,下轮重做
            logger.info("memory pipeline compact apply failed: %s: %s",
                        type(e).__name__, str(e)[:160])
    return {"summarized": bool(final.get("summarized")),
            "compacted": bool(final.get("compacted")),
            "level": final.get("summary_level") or "",
            "removed": len(removes)}


# ---------------- 进程内单例 ----------------
_pipeline: Optional[MemoryPipeline] = None
_pipeline_lock = threading.Lock()


def get_pipeline() -> MemoryPipeline:
    global _pipeline
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                _pipeline = MemoryPipeline()
    return _pipeline


def reset_pipeline() -> None:
    """测试用:丢弃单例(已排队任务随之丢弃)。"""
    global _pipeline
    with _pipeline_lock:
        _pipeline = None
