# -*- coding: utf-8 -*-
"""raglite 快路径运行入口:用户消息落库 + run_path 包装(镜像 simple/runner.py)。"""
from __future__ import annotations

import time
import uuid
from typing import Callable, Optional

from agent_reasoning.ReAct.trace import TraceRecorder
from agent_reasoning.ReAct.support.runner import run_path
from memories.storage.thread_scope import scoped_thread_id
from .stream import raglite_answer_stream


def run_raglite(message: str,
                history: Optional[list[dict]] = None,
                *,
                thread_id: str,
                username: Optional[str] = None,
                session_id: Optional[str] = None,
                on_event: Optional[Callable[[dict], None]] = None,
                hard_deadline: Optional[float] = None,
                qc_feedback: Optional[str] = None,
                search_future=None,
                **_):
    """生成器:raglite 路径(1 次检索 + 1 次主模型流式作答)。

    经 run_path 包装,具备与 react 路径一致的短期流水落库、on_event 回调与
    异常兜底。``search_future`` 为服务层投机检索的 future(已并发跑起来的
    search_text 调用),传入则直接取结果,省一次串行检索。
    ``qc_feedback``(同层重做反馈)与 ``hard_deadline`` 与 react 同约定。
    """
    t0 = time.time()
    trace_id = str(uuid.uuid4())[:8]
    recorder = TraceRecorder(trace_id, t0, message)

    # 短期流水按用户隔离的存储键(防 thread_id IDOR;raglite 无 checkpoint)。
    store_thread_id = scoped_thread_id(thread_id, username)

    # 流开始前:落一条用户消息到短期流水(用按用户隔离的存储键)
    try:
        from memories.storage.short import short_term
        short_term.append_event(
            store_thread_id, "user_message", {"content": message},
            user_id=username, session_id=session_id,
        )
    except Exception:
        pass

    event_iterable = raglite_answer_stream(
        message, history,
        recorder=recorder, trace_id=trace_id, t0=t0,
        username=username, thread_id=store_thread_id,
        hard_deadline=hard_deadline, qc_feedback=qc_feedback,
        search_future=search_future,
    )
    yield from run_path(
        "raglite", event_iterable,
        thread_id=store_thread_id, username=username, session_id=session_id,
        trace_id=trace_id, on_event=on_event,
        on_error=lambda e: setattr(recorder, "final_reason", "error"),
    )
