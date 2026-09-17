# -*- coding: utf-8 -*-
"""强化 ReAct 工作流的辅助工具集:LLM 客户端、工具线程池、运行入口。

与 ..core(图骨架)解耦:core 的节点通过相对路径从本包取这些支撑能力。
"""
from .llm import get_client, llm_create_with_retry, LLM_RETRIES

# 注意:runner 是运行入口,会拉起 ..core.graph→nodes→本包的 llm,
# 不在 __init__ 里 eager import,避免包初始化阶段的循环导入;需要时用
# `from ..support.runner import run_agent_graph` 直接导入。

__all__ = ["get_client", "llm_create_with_retry", "LLM_RETRIES"]
