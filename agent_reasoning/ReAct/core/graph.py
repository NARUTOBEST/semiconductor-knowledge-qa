# -*- coding: utf-8 -*-
"""LangGraph 图定义:把节点与边拼成 ReAct 图。

拓扑(1.2 将 reflect 拆成 coverage_check + grounding 两个独立节点):
  START → setup → recall → rewrite → plan → build_messages → react
                                                          ↓
                                                   coverage_check
                                                    ├─(回退)→ react
                                                    └─→ grounding
                                                          ├─(反思重生成)→ react
                                                          └─→ finalize → END

- react 节点内部自循环到出答案/终态(answer/timeout/max_steps/error)。
- coverage_check:计划步骤覆盖度判定,未覆盖且有预算则回退重检索。
- grounding:引用校验 + 忠实度检测,失败且有预算则反思重生成。
- 回退/重生成均置 final_reason=None 并写入 reflect_feedback,据此路由回 react。
"""
from langgraph.graph import START, END, StateGraph

from .state import AgentState
from . import nodes
from .loop import react_node


def _after_coverage(state: AgentState) -> str:
    # coverage 回退时置 final_reason=None 并写入 reflect_feedback -> 回到 react
    if state.get("final_reason") is None and state.get("reflect_feedback"):
        return "react"
    return "grounding"


def _after_grounding(state: AgentState) -> str:
    # grounding 反思重生成时置 final_reason=None 并写入 reflect_feedback -> 回到 react
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
    b.add_node("coverage_check", nodes.coverage_check_node)
    b.add_node("grounding", nodes.grounding_node)
    b.add_node("finalize", nodes.finalize_node)

    b.add_edge(START, "setup")
    b.add_edge("setup", "recall")
    b.add_edge("recall", "rewrite")
    b.add_edge("rewrite", "plan")
    b.add_edge("plan", "build_messages")
    b.add_edge("build_messages", "react")
    b.add_edge("react", "coverage_check")
    b.add_conditional_edges(
        "coverage_check", _after_coverage,
        {"react": "react", "grounding": "grounding"},
    )
    b.add_conditional_edges(
        "grounding", _after_grounding,
        {"react": "react", "finalize": "finalize"},
    )
    b.add_edge("finalize", END)

    return b.compile(checkpointer=checkpointer)
