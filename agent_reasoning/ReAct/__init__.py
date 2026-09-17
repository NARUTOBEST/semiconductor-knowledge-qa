# -*- coding: utf-8 -*-
"""ReAct 范式:LangGraph 编排实现。

包内分两层:
  core     ReAct 框架本身 —— State/节点/图编译(图骨架)
  support  辅助工具 —— LLM 客户端、工具线程池、运行入口

记忆层的装配与生命周期(短期落库、工作记忆压缩清理)在 memories.orchestration /
memories.storage.working.summarize;support.runner 通过 import 调用它们。

对外入口 `run_agent_graph(...)`,由 `chat.service` 调用。
为兼容历史导入路径(chat.service、测试),本包仍重导出常用符号。
"""
from .core.state import AgentState
from .core.graph import build_graph
from .support.llm import get_client, llm_create_with_retry, LLM_RETRIES
from .support.runner import run_agent_graph

__all__ = [
    "run_agent_graph",
    "AgentState",
    "build_graph",
    "get_client",
    "llm_create_with_retry",
    "LLM_RETRIES",
]
