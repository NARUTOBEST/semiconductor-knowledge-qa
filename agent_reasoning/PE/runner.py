# -*- coding: utf-8 -*-
"""PE(Plan-and-Execute)范式运行入口。

必经规划 -> 交给 plan_execute_stream 跑步骤隔离执行 + synthesizer;
planner 失败或空步骤降级为普通 ReAct(medium)。经 run_path 包装。
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Callable, Optional

from agent_reasoning.ReAct.trace import TraceRecorder
from agent_reasoning.ReAct.support.runner import (
    run_path, run_agent_graph,
)
from agent_reasoning.ReAct.support.planning import generate_plan
from .plan_execute import plan_execute_stream

logger = logging.getLogger("agent")


def run_plan_execute(message: str,
                     history: Optional[list[dict]] = None,
                     *,
                     thread_id: str,
                     username: Optional[str] = None,
                     session_id: Optional[str] = None,
                     max_steps: int = 6,
                     max_total_seconds: int = 60,
                     on_event: Optional[Callable[[dict], None]] = None):
    """生成器:complex Plan-and-Execute 路径(阶段 5)。

    第一步必经规划(generate_plan force=True);planner 失败或产出空步骤时
    降级为普通 ReAct(发 status 告知用户,5.8)。规划成功则交给
    plan_execute_stream 跑步骤隔离执行 + synthesizer。经 run_path 包装,
    与其它路径共享短期流水 / 升迁 / 异常兜底。
    """
    t0 = time.time()
    trace_id = str(uuid.uuid4())[:8]
    recorder = TraceRecorder(trace_id, t0, message)

    # ---- 必经规划(5.1 / 5.2);失败降级普通 ReAct(5.8)----
    planner_error = None
    t_plan_start = time.time()
    try:
        steps, plan_err = generate_plan(message, force=True, trace_id=trace_id)
        if plan_err:
            planner_error = plan_err
            logger.warning("planner failed, fallback to react: %s", plan_err)
            steps = []
    except Exception as e:
        planner_error = f"{type(e).__name__}: {e}"
        logger.warning("planner exception, fallback to react: %s", e)
        steps = []
    planner_duration_ms = int((time.time() - t_plan_start) * 1000)

    if not steps:
        # planner 不可用:降级 medium ReAct,不阻断。直接委托 run_agent_graph
        # (它会自己落 user_message、包 run_path、新建 recorder,故此处不重复落)。
        yield {"type": "status", "trace_id": trace_id,
               "message": "检索规划不可用,将直接检索作答…"}
        yield from run_agent_graph(
            message, history,
            thread_id=thread_id, username=username, session_id=session_id,
            max_steps=max_steps, max_total_seconds=max_total_seconds,
            on_event=on_event,
        )
        return

    # 规划成功:落一条用户消息到短期流水(降级分支由 run_agent_graph 负责落)
    try:
        from memories.storage.short import short_term
        short_term.append_event(
            thread_id, "user_message", {"content": message},
            user_id=username, session_id=session_id,
        )
    except Exception:
        pass

    event_iterable = plan_execute_stream(
        message, history, steps,
        recorder=recorder, trace_id=trace_id, t0=t0,
        thread_id=thread_id, username=username,
        max_total_seconds=max_total_seconds,
        planner_error=planner_error,
        planner_duration_ms=planner_duration_ms,
    )
    yield from run_path(
        "plan_execute", event_iterable,
        thread_id=thread_id, username=username, session_id=session_id,
        trace_id=trace_id, on_event=on_event,
        on_error=lambda e: setattr(recorder, "final_reason", "error"),
    )
