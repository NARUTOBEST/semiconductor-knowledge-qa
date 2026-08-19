# -*- coding: utf-8 -*-
"""LangGraph 图定义:把节点与边拼成 ReAct 循环。

拓扑:
  START → setup → recall → rewrite → build_messages → agent
                                              ┌─ tools ─┘  (有 tool_calls)
                                              └─ finalize (终态: answer/timeout/max_steps/error)
  finalize → END

路由依据 state["final_reason"]:
  None         -> tools (agent 这一轮要求调工具)
  非 None      -> finalize (终态原因)
"""
from langgraph.graph import START, END, StateGraph

from .state import AgentState
from . import nodes


def _after_agent(state: AgentState) -> str:
    if state.get("final_reason"):
        return "reflect"
    return "tools"


def _after_reflect(state: AgentState) -> str:
    # reflect 重试时置 final_reason=None 并写入 reflect_feedback
    if state.get("final_reason") is None and state.get("reflect_feedback"):
        return "agent"
    return "finalize"


def build_graph(checkpointer=None):
    """编译 ReAct 图。checkpointer 为 None 时为无状态图(测试用)。"""
    b = StateGraph(AgentState)
    b.add_node("setup", nodes.setup_node)
    b.add_node("recall", nodes.recall_node)
    b.add_node("rewrite", nodes.rewrite_node)
    b.add_node("plan", nodes.plan_node)
    b.add_node("build_messages", nodes.build_messages_node)
    b.add_node("agent", nodes.agent_node)
    b.add_node("tools", nodes.tools_node)
    b.add_node("reflect", nodes.reflect_node)
    b.add_node("finalize", nodes.finalize_node)

    b.add_edge(START, "setup")
    b.add_edge("setup", "recall")
    b.add_edge("recall", "rewrite")
    b.add_edge("rewrite", "plan")
    b.add_edge("plan", "build_messages")
    b.add_edge("build_messages", "agent")
    b.add_conditional_edges(
        "agent", _after_agent,
        {"tools": "tools", "reflect": "reflect"},
    )
    b.add_edge("tools", "agent")
    b.add_conditional_edges(
        "reflect", _after_reflect,
        {"agent": "agent", "finalize": "finalize"},
    )
    b.add_edge("finalize", END)

    return b.compile(checkpointer=checkpointer)
