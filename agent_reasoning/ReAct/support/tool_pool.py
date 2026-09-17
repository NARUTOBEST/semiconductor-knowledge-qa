# -*- coding: utf-8 -*-
"""工具 fan-out 专用的全局 daemon 线程池(进程级单例)。

为什么不用每轮新建 ``with ThreadPoolExecutor(...)``:
  with 退出时隐式 ``shutdown(wait=True)``,会 join 仍在运行的 worker。工具调用
  (多模态 VL 30s、图片下载等)若超过端到端硬预算,``future.result(timeout=)``
  超时后 with 退出仍被挂死的 worker 拖住 —— 导致 done 事件延迟、断连后
  generator 无法收尾、工具线程随并发请求累积。

本池:
  - 进程级单例、有界(max_workers),工作线程全部 **daemon** —— 进程退出自动
    消失,解释器退出不等待;
  - 任务超时后直接丢弃结果:``cancel()`` 只能取消仍在排队的任务,运行中的任务
    允许跑完但结果无人读取,**绝不 join**;
  - 空闲线程阻塞在队列上,无轮询开销。
"""
from __future__ import annotations

import queue
import threading
from concurrent.futures import Future
from typing import Any, Callable

import config as C


class _DaemonThreadPool:
    """固定数量 daemon 线程 + 无界任务队列的极简线程池。

    任务排队满负载时,多余任务在队列等待;调用方用 ``future.result(timeout=)``
    控制预算,排队过久同样超时快速失败(过载保护,优于无限堆积)。
    """

    def __init__(self, max_workers: int):
        self._q: "queue.Queue[tuple[Future, Callable, tuple, dict]]" = queue.Queue()
        self._max = max(1, int(max_workers))
        self._spawned = 0
        self._lock = threading.Lock()

    def submit(self, fn: Callable, *args: Any, **kwargs: Any) -> Future:
        fut: Future = Future()
        self._q.put((fut, fn, args, kwargs))
        # 懒扩张:线程数达到 max_workers 前,每提交一个任务补一个线程
        with self._lock:
            if self._spawned < self._max:
                self._spawned += 1
                idx = self._spawned
                threading.Thread(
                    target=self._worker,
                    name=f"tool-pool-{idx}",
                    daemon=True,
                ).start()
        return fut

    def _worker(self) -> None:
        while True:
            fut, fn, args, kwargs = self._q.get()
            # cancel() 只能取消仍在排队(PENDING)的任务;已取消则跳过执行。
            # 运行中的任务 cancel 无效,允许跑完(结果丢弃),不阻塞任何收尾。
            if not fut.set_running_or_notify_cancel():
                continue
            try:
                fut.set_result(fn(*args, **kwargs))
            except BaseException as e:  # noqa: BLE001 - 异常交回 future,与 ThreadPoolExecutor 行为一致
                fut.set_exception(e)


_pool: "_DaemonThreadPool | None" = None
_pool_lock = threading.Lock()


def get_pool() -> _DaemonThreadPool:
    """全局工具线程池单例。

    容量 = 单请求 fan-out 上限(TOOL_MAX_PARALLEL)× 全局并发槽位
    (RATE_LIMIT_GLOBAL_CONCURRENT),保证满载时各请求的工具调用都有线程;
    超出部分排队,由调用方的硬预算超时兜底。
    """
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                parallel = int(getattr(C, "TOOL_MAX_PARALLEL", 4) or 4)
                global_slots = int(getattr(C, "RATE_LIMIT_GLOBAL_CONCURRENT", 8) or 8)
                _pool = _DaemonThreadPool(max_workers=parallel * global_slots)
    return _pool
