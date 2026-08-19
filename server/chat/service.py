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
from agent_reasoning.ReAct.support.runner import run_agent_graph
from agent_reasoning.ReAct.support.llm import get_client, llm_create_with_retry, LLM_RETRIES
from agent_reasoning.ReAct.support.answer_grounding import (
    verify_citations,
    check_faithfulness,
    grounding_check,
    yield_grounding_warnings,
)

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
    """生成器:yield SSE 事件 dict,委托给 agent_reasoning.ReAct.run_agent_graph。

    保持与旧签名兼容(message, history, max_total_seconds, on_event);
    新增 thread_id/username/session_id/max_steps 由路由层传入(用于 checkpoint
    与长期记忆按用户隔离)。thread_id 缺省时生成随机 uuid。
    """
    if thread_id is None:
        thread_id = str(uuid.uuid4())

    yield from run_agent_graph(
        message,
        history,
        thread_id=thread_id,
        username=username,
        session_id=session_id,
        max_steps=max_steps,
        max_total_seconds=max_total_seconds,
        on_event=on_event,
    )
