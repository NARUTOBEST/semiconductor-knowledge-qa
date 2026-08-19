# -*- coding: utf-8 -*-
"""LangGraph Agent 的运行入口(同步生成器)。

run_agent_graph 对外 yield 的事件 dict 与旧 react_stream 完全一致,
是 chat.service 唯一需要调用的新版入口。内部:
  1. 初始化 TraceRecorder,放进 config["configurable"]
  2. 用 working_saver() 编译图(挂 PostgresSaver checkpoint)
  3. 流开始前补写 user_message 到短期流水
  4. graph.stream(stream_mode="custom"):每个事件先持久化短期、再回调 on_event、再 yield
  5. 结束/异常后在后台触发长期升迁
"""
from __future__ import annotations

import time
import uuid
from typing import Any, Callable, Optional

from ..trace import TraceRecorder
from memories.storage import working_saver

from ..core.graph import build_graph
from .plan_grounding import CoverageTracker
from .llm import get_client, llm_create_with_retry
from memories.orchestration import persist_event, after_stream


def run_agent_graph(message: str,
                    history: Optional[list[dict]] = None,
                    *,
                    thread_id: str,
                    username: Optional[str] = None,
                    session_id: Optional[str] = None,
                    max_steps: int = 6,
                    max_total_seconds: int = 60,
                    on_event: Optional[Callable[[dict], None]] = None):
    """生成器:yield SSE 事件 dict(结构与旧 react_stream 一致)。

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

    promoted = False

    def _promote():
        nonlocal promoted
        if promoted:
            return
        promoted = True
        after_stream(thread_id, username, session_id)

    with working_saver() as cp:
        graph = build_graph(checkpointer=cp)
        try:
            for ev in graph.stream(inputs, config=config, stream_mode="custom"):
                persist_event(ev, thread_id=thread_id,
                              user_id=username, session_id=session_id)
                if on_event is not None:
                    try:
                        on_event(ev)
                    except Exception:
                        pass
                yield ev
        except Exception as e:
            # 图执行异常:发 error + done,保证前端能收尾
            err = {"type": "error", "trace_id": trace_id,
                   "message": f"内部错误: {type(e).__name__}: {str(e)[:160]}"}
            persist_event(err, thread_id=thread_id,
                          user_id=username, session_id=session_id)
            if on_event is not None:
                try:
                    on_event(err)
                except Exception:
                    pass
            yield err
            recorder.final_reason = "error"
        finally:
            _promote()
            coverage_tracker.close()
