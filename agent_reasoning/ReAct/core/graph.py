# -*- coding: utf-8 -*-
"""LangGraph 图定义:把节点与边拼成 ReAct 图。

拓扑(agent↔tools 工具循环已抽到 core/loop.py 托管为单个 react 节点):
  START → setup → recall → rewrite → plan → build_messages → react
                                                          ↓
                                                       reflect ─(重生成)→ react
                                                          ↓
                                                       finalize → END

react 节点内部自循环到出答案/终态(answer/timeout/max_steps/error);
reflect 重生成时置 final_reason=None 并写入 reflect_feedback,回到 react。
"""
from langgraph.graph import START, END, StateGraph

from .state import AgentState
from . import nodes
from .loop import react_node


def _after_reflect(state: AgentState) -> str:
    # reflect 重试时置 final_reason=None 并写入 reflect_feedback
    if state.get("final_reason") is None and state.get("reflect_feedback"):
        return "react"
    return "finalize"


def build_graph(checkpointer=None):
    """编译 ReAct 图。checkpointer 为 None 时为无状态图(测试用)。"""
    b = StateGraph(AgentState)
    b.add_node("setup", nodes.setup_node)
    b.add_node("recall", nodes.recall_node)
    b.add_node("rewrite", nodes.rewrite_node)
    b.add_node("plan", nodes.plan_node)
    b.add_node("build_messages", nodes.build_messages_node)
    b.add_node("react", react_node)
    b.add_node("reflect", nodes.reflect_node)
    b.add_node("finalize", nodes.finalize_node)

    b.add_edge(START, "setup")
    b.add_edge("setup", "recall")
    b.add_edge("recall", "rewrite")
    b.add_edge("rewrite", "plan")
    b.add_edge("plan", "build_messages")
    b.add_edge("build_messages", "react")
    b.add_edge("react", "reflect")
    b.add_conditional_edges(
        "reflect", _after_reflect,
        {"react": "react", "finalize": "finalize"},
    )
    b.add_edge("finalize", END)

    return b.compile(checkpointer=checkpointer)
