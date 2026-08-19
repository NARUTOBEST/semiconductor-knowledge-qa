# -*- coding: utf-8 -*-
"""知识助手 -- 聊天服务入口(薄封装)。

本文件只做一件事:提供 react_stream(),把路由层请求委托给新版 LangGraph ReAct
实现(agent_reasoning.ReAct.support.runner.run_agent_graph)。具体的 ReAct 主循环、LLM 调用、grounding
检测等全部在 chat/react/ 包内,此处不再保留实现。

为兼容现有调用方(health/service.py、context management/query_rewrite.py 及
若干测试以 chat.service.get_client 等路径导入/打桩),这里重导出相关符号。
"""
from __future__ import annotations

import uuid
from typing import Callable, Optional

# ---- 新版 ReAct 实现(已独立到顶层包 agent_reasoning.ReAct)----
from agent_reasoning.ReAct.support.runner import run_agent_graph, run_simple
from agent_reasoning.ReAct.support.llm import get_client, llm_create_with_retry, LLM_RETRIES
from agent_reasoning.ReAct.support.answer_grounding import (
    verify_citations,
    check_faithfulness,
    grounding_check,
    yield_grounding_warnings,
)
from agent_reasoning.router import classify_complexity

__all__ = [
    "react_stream",
    "get_client",
    "llm_create_with_retry",
    "LLM_RETRIES",
    "verify_citations",
    "check_faithfulness",
    "grounding_check",
    "yield_grounding_warnings",
]


def react_stream(message,
                 history,
                 max_total_seconds: int = 60,
                 on_event: Optional[Callable[[dict], None]] = None,
                 *,
                 thread_id: Optional[str] = None,
                 username: Optional[str] = None,
                 session_id: Optional[str] = None,
                 max_steps: int = 6):
    """生成器:yield SSE 事件 dict。

    复杂度路由(阶段 3):先用 classify_complexity 判定 tier,发 ``tier`` 事件,
    再分发:
      - simple -> run_simple(单轮直答,lite 模型,不绑工具)
      - medium -> run_agent_graph(ReAct 循环)
      - complex -> 暂走 run_agent_graph(阶段 5 接入 Plan-and-Execute)

    thread_id 缺省时生成随机 uuid。
    """
    if thread_id is None:
        thread_id = str(uuid.uuid4())

    # 1. 复杂度分类(失败兜底 medium,不阻断)
    decision = classify_complexity(message, history or [])
    tier = decision["tier"]

    # 2. 告知前端所选 tier(前端可据此展示不同 UI/进度)
    yield {"type": "tier", "tier": tier,
           "confidence": decision.get("confidence", 0.0),
           "source": decision.get("source", "fallback")}

    # 3. 按 tier 分发(complex 在阶段 5 接入 P&E 前先走 medium ReAct)
    if tier == "simple":
        yield from run_simple(
            message, history,
            thread_id=thread_id, username=username, session_id=session_id,
            on_event=on_event,
        )
    else:
        yield from run_agent_graph(
            message, history,
            thread_id=thread_id, username=username, session_id=session_id,
            max_steps=max_steps, max_total_seconds=max_total_seconds,
            on_event=on_event,
        )
