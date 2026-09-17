# -*- coding: utf-8 -*-
"""LangGraph 图定义:把节点与边拼成 ReAct 图。

拓扑:
  START → setup → build_messages → react → finalize → emit_done → END

- react 节点(loop.react_node)内部托管 agent↔tools 子图,自循环到出答案/终态。
- 记忆维护【已整体迁出主图】:finalize → emit_done 后图流即完即关,done 立即
  发、限流槽立即释放;每轮记忆任务由 runner 在流结束后提交给后台记忆管道
  (memories/orchestration/memory_loop/pipeline.py,独立记忆图 + 按会话串行),
  全程不占请求流。同会话下一轮请求在入口经 wait_idle 等上一轮记忆完成。
- build_compaction_graph():最小图,仅供后台把摘要节点产出的 RemoveMessage
  经 update_state 应用到会话 checkpoint(两轮之间执行,安全落压缩)。
"""
from langgraph.graph import START, END, StateGraph

from .state import AgentState
from . import nodes
from .loop import react_node


def build_graph(checkpointer=None):
    """编译 ReAct 图。checkpointer 为 None 时为无状态图(测试用)。"""
    b = StateGraph(AgentState)
    b.add_node("setup", nodes.setup_node)
    b.add_node("build_messages", nodes.build_messages_node)
    b.add_node("react", react_node)
    b.add_node("finalize", nodes.finalize_node)
    b.add_node("emit_done", nodes.emit_done_node)

    b.add_edge(START, "setup")
    b.add_edge("setup", "build_messages")
    b.add_edge("build_messages", "react")
    b.add_edge("react", "finalize")
    b.add_edge("finalize", "emit_done")
    b.add_edge("emit_done", END)

    return b.compile(checkpointer=checkpointer)


def build_compaction_graph(checkpointer=None):
    """最小图(单 no-op 节点),复用 AgentState 的 messages 通道语义。

    后台记忆管道经 graph.update_state(config, {"messages": [RemoveMessage...]})
    把 Auto-Compact 应用到会话 checkpoint——不经过任何 LLM/工具节点。
    """
    b = StateGraph(AgentState)
    b.add_node("noop", lambda state: {})
    b.add_edge(START, "noop")
    b.add_edge("noop", END)
    return b.compile(checkpointer=checkpointer)
