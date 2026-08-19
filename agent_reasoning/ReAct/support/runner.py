# -*- coding: utf-8 -*-
"""LangGraph Agent 的运行入口(同步生成器)。

本模块提供两类东西:
1. 通用包装器 ``run_path()``:把任意一条 SSE 事件流(simple / react / P&E)
   包上横切逻辑——短期流水落库、on_event 回调、异常兜底发 error、结束后触发
   长期升迁。三条路径共用,保证流水/升迁/异常行为一致。
2. ``run_agent_graph()``:medium ReAct 路径的具体实现——初始化 TraceRecorder、
   CoverageTracker、checkpointer,编译图,再把 ``graph.stream`` 交给 run_path。

simple 范式入口在 ``agent_reasoning.simple.run_simple``;
P&E 范式入口在 ``agent_reasoning.PE.run_plan_execute``。
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Callable, Iterable, Optional

from ..trace import TraceRecorder
from memories.storage import working_saver

from ..core.graph import build_graph
from .plan_grounding import CoverageTracker
from .llm import get_client, llm_create_with_retry
from memories.orchestration import persist_event, after_stream

logger = logging.getLogger("agent")


def run_path(path_name: str,
             event_iterable: Iterable[dict],
             *,
             thread_id: str,
             username: Optional[str] = None,
             session_id: Optional[str] = None,
             trace_id: Optional[str] = None,
             on_event: Optional[Callable[[dict], None]] = None,
             on_error: Optional[Callable[[Exception], None]] = None,
             on_finally: Optional[Callable[[], None]] = None):
    """通用事件流包装器:包裹 simple / react / P&E 任意一条事件流。

    :param path_name: 路径名(仅日志/审计用,如 "simple"/"react"/"plan_execute")。
    :param event_iterable: 被包装的 SSE 事件流(生成器)。
    :param thread_id/username/session_id: 短期流水与长期升迁的主键。
    :param trace_id: 异常兜底 error 事件使用的 trace 关联 id。
    :param on_event: 每个事件的额外回调(审计/指标),异常不影响主流。
    :param on_error: 流抛异常时的回调(路径用于置 recorder.final_reason 等)。
    :param on_finally: 流结束(含异常)后的清理回调(如关闭 CoverageTracker)。

    透传事件;异常时补发一条 ``error`` 事件并保证前端能收尾;最终在 finally 中
    触发一次长期升迁(幂等)。
    """
    trace_id = trace_id or ""
    promoted = False

    def _promote():
        nonlocal promoted
        if promoted:
            return
        promoted = True
        after_stream(thread_id, username, session_id)

    try:
        for ev in event_iterable:
            persist_event(ev, thread_id=thread_id,
                          user_id=username, session_id=session_id)
            if on_event is not None:
                try:
                    on_event(ev)
                except Exception:
                    pass
            yield ev
    except Exception as e:
        # 图/路径执行异常:发 error 事件,保证前端能收尾
        err = {"type": "error", "trace_id": trace_id,
               "message": f"内部错误: {type(e).__name__}: {str(e)[:160]}"}
        persist_event(err, thread_id=thread_id,
                      user_id=username, session_id=session_id)
        if on_event is not None:
            try:
                on_event(err)
            except Exception:
                pass
        if on_error is not None:
            try:
                on_error(e)
            except Exception:
                pass
        yield err
    finally:
        _promote()
        if on_finally is not None:
            try:
                on_finally()
            except Exception:
                pass


def run_agent_graph(message: str,
                    history: Optional[list[dict]] = None,
                    *,
                    thread_id: str,
                    username: Optional[str] = None,
                    session_id: Optional[str] = None,
                    max_steps: int = 6,
                    max_total_seconds: int = 60,
                    on_event: Optional[Callable[[dict], None]] = None):
    """生成器:yield SSE 事件 dict(medium ReAct 路径)。

    thread_id: 会话/任务标识(对应 LangGraph checkpoint thread_id)。
    username:  当前用户名(=user_id,长期记忆按此隔离)。
    on_event:  每个事件的额外回调(供上层审计/指标用),异常不影响主流。
    """
    t0 = time.time()
    trace_id = str(uuid.uuid4())[:8]
    recorder = TraceRecorder(trace_id, t0, message)

    # 计划步骤覆盖度追踪器(守护线程);plan_node 在 need_plan=True 时 set_plan+启动。
    # 简单问题永不喂入计划,线程空转等待 close,开销可忽略。
    def _llm_caller(**kwargs):
        # trace_id 由调用方(coverage tracker)传入,此处不要重复传
        return llm_create_with_retry(get_client(), **kwargs)

    coverage_tracker = CoverageTracker(trace_id, _llm_caller).start()

    config: dict[str, Any] = {
        "configurable": {
            "thread_id": thread_id,
            "user_id": username,
            "session_id": session_id,
            "trace_recorder": recorder,
            "coverage_tracker": coverage_tracker,
        }
    }
    inputs: dict[str, Any] = {
        "question": message,
        "history": history or [],
        "started_at": t0,
        "max_steps": max_steps,
        "max_total_seconds": max_total_seconds,
        # 每轮传入新 trace_id,覆盖 checkpoint 残留的旧值,
        # 保证 setup_node 与 recorder/异常兜底事件用同一个 id
        "trace_id": trace_id,
    }

    # 流开始前:落一条用户消息到短期流水
    try:
        from memories.storage.short import short_term
        short_term.append_event(
            thread_id, "user_message", {"content": message},
            user_id=username, session_id=session_id,
        )
    except Exception:
        pass

    with working_saver() as cp:
        graph = build_graph(checkpointer=cp)
        event_iterable = graph.stream(
            inputs, config=config, stream_mode="custom",
        )
        yield from run_path(
            "react", event_iterable,
            thread_id=thread_id, username=username, session_id=session_id,
            trace_id=trace_id, on_event=on_event,
            on_error=lambda e: setattr(recorder, "final_reason", "error"),
            on_finally=coverage_tracker.close,
        )
