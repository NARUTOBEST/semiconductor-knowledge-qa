# -*- coding: utf-8 -*-
"""simple 范式运行入口:用户消息落库 + run_path 包装。"""
from __future__ import annotations

import time
import uuid
from typing import Callable, Optional

from agent_reasoning.ReAct.trace import TraceRecorder
from agent_reasoning.ReAct.support.runner import run_path
from memories.storage.thread_scope import scoped_thread_id
from .stream import simple_answer_stream


def run_simple(message: str,
               history: Optional[list[dict]] = None,
               *,
               thread_id: str,
               username: Optional[str] = None,
               session_id: Optional[str] = None,
               on_event: Optional[Callable[[dict], None]] = None):
    """生成器:simple 路径(单轮直答,lite 模型,不绑工具)。

    经 run_path 包装,具备与 react 路径一致的短期流水落库、on_event 回调与
    异常兜底。不绑工具、不调旁路 LLM(不改写/不质检)。
    """
    t0 = time.time()
    trace_id = str(uuid.uuid4())[:8]
    recorder = TraceRecorder(trace_id, t0, message)

    # 短期流水按用户隔离的存储键(防 thread_id IDOR;simple 无 checkpoint)。
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

    event_iterable = simple_answer_stream(
        message, history,
        recorder=recorder, trace_id=trace_id, t0=t0,
        username=username, thread_id=store_thread_id,
    )
    yield from run_path(
        "simple", event_iterable,
        thread_id=store_thread_id, username=username, session_id=session_id,
        trace_id=trace_id, on_event=on_event,
        on_error=lambda e: setattr(recorder, "final_reason", "error"),
    )
