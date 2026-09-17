# -*- coding: utf-8 -*-
"""raglite 快路径:单一事实点 1 次检索 + 1 次主模型流式作答,不走 ReAct 循环。

与 simple(无检索闲聊)和 react(多步工具循环)互补,由路由器按问题形态分发:
领域关键词命中且非复杂标记(对比/流程/因果等)的单事实问题走本路径。
线上 tier 事件仍对外发 "react"(附 path=raglite),eval tier_ok / 前端零改动。
"""
from .runner import run_raglite

__all__ = ["run_raglite"]
