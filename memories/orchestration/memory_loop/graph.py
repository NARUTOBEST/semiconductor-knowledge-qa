# -*- coding: utf-8 -*-
"""独立记忆工作图(与 ReAct 主图完全解耦)。

拓扑(线性两节点,兜底收拢在节点一内):
  START → mem_consolidate ─┬→ retry ┐(条件边,同旧主图内联版)
                           ├→ degrade ─┐
                           └→ summary ─┤
                                       ↓
                              mem_summary(阈值不满足内部零成本早退)
                                       ↓
                                      END

mem_consolidate 先做降级兜底(WAL 回填 + PG spool 重放 + NULL 向量补嵌,
不挑门控,匿名/关闭记忆轮也推进故障欠账),再做本轮沉淀(门控不过则只兜底)。

- 由后台管道(pipeline.py)在答案定稿后 invoke,不占请求流;无 checkpointer
  (记忆权威在 Redis 事实表 / session-memory.md / PG,checkpoint 压缩经
  summary 节点产出的 remove_messages 由管道转交调用方注入)。
- 之所以保留图编排而非纯函数串联:节点/条件边结构与故障路径(retry/degrade)
  显式可见、可单测,与纯逻辑层(consolidate/session_summary/resilience)分层清晰。
"""
from typing import Any, TypedDict

from langgraph.graph import START, END, StateGraph

from . import nodes as MN


class MemoryState(TypedDict, total=False):
    """记忆图输入/输出(纯内存 state,不落 checkpoint)。"""
    # ---- 输入(管道 submit 时装配)----
    username: str
    thread_id: str
    question: str
    full_reply: str
    messages: list          # 本轮定稿后的完整消息快照(BaseMessage)
    final_reason: str
    # ---- 内部路由 ----
    mem_llm_fail_count: int
    mem_breaker_open: bool
    mem_degraded: bool
    mem_fact_id: str
    mem_consolidate_done: bool
    # ---- 输出 ----
    summarized: bool
    compacted: bool
    summary_level: str
    remove_messages: list   # [RemoveMessage];由管道转交 compact_applier 落 checkpoint


def build_memory_graph():
    """编译独立记忆图(无状态,每次 invoke 独立)。"""
    b = StateGraph(MemoryState)
    b.add_node("mem_consolidate", MN.consolidate_node)
    b.add_node("mem_consolidate_retry", MN.consolidate_retry_node)
    b.add_node("mem_consolidate_degrade", MN.consolidate_degrade_node)
    b.add_node("mem_summary", MN.summary_node)

    b.add_edge(START, "mem_consolidate")
    _cs_branches = {
        "summary": "mem_summary",
        "retry": "mem_consolidate_retry",
        "degrade": "mem_consolidate_degrade",
    }
    b.add_conditional_edges("mem_consolidate", MN.route_consolidate, _cs_branches)
    b.add_conditional_edges("mem_consolidate_retry", MN.route_consolidate, _cs_branches)
    b.add_edge("mem_consolidate_degrade", "mem_summary")
    b.add_edge("mem_summary", END)
    return b.compile()
