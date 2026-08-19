# -*- coding: utf-8 -*-
"""把 agent_node ↔ tools_node 的 ReAct 工具循环抽成可复用单元。

设计要点:
- ``build_react_graph()`` 编译一个只含 agent↔tools 两节点的子图(无 checkpointer),
  路由依据 ``final_reason``:有值(answer/timeout/max_steps/error)结束,否则继续 tools。
- ``react_loop()`` 是给上层(P&E 每步、其它路径)直接用的生成器:传入初始 messages 与
  运行参数,yield SSE 事件 dict,并通过 ``return`` 给出最终 state(调用方从
  ``StopIteration.value`` 或 ``react_node`` 内部取得)。
- 主图通过 ``react_node`` 把这个循环托管成「单个图节点」:在节点内运行子图,把子图的
  自定义事件转发到父流(保持扁平 SSE 契约),并把子图最终 state 以「reducer 友好的补丁」
  形式回写到父图。

为什么不用 LangGraph 原生嵌套子图:当前版本中子图的自定义事件不会冒泡到父图的
``stream_mode="custom"``(需 subgraphs=True 且带命名空间,会破坏现有扁平事件契约),
因此由 react_node 显式转发事件。
"""
from __future__ import annotations

import time
from typing import Any, Optional

from langgraph.graph import START, END, StateGraph

from .state import AgentState
from . import nodes


def _after_agent(state: AgentState) -> str:
    """agent 之后:有终态原因则结束子图,否则进入 tools。"""
    return "end" if state.get("final_reason") else "tools"


def build_react_graph(checkpointer=None):
    """编译 agent↔tools 循环子图(最终 state 由 values 流取得)。

    默认无 checkpointer;独立调用 react_loop 时可传入以支持步骤内断点续跑。
    """
    b = StateGraph(AgentState)
    b.add_node("agent", nodes.agent_node)
    b.add_node("tools", nodes.tools_node)
    b.add_edge(START, "agent")
    b.add_conditional_edges(
        "agent", _after_agent,
        {"tools": "tools", "end": END},
    )
    b.add_edge("tools", "agent")
    return b.compile(checkpointer=checkpointer)


def _run_react_stream(state: AgentState, configurable: dict[str, Any],
                      checkpointer=None):
    """运行子图,返回 (events_iter, final_state_getter)。

    用 ``stream_mode=["custom","values"]``:custom 用于转发 SSE 事件,
    values 最后一帧即最终 state(无 checkpointer 时无法 get_state)。

    :param checkpointer: 子图 checkpointer。react_node 固定传 None(父图已
        持久化其输出);独立调用 react_loop(P&E 每步)可按需传入以做步骤内断点续跑。
    """
    graph = build_react_graph(checkpointer=checkpointer)
    cfg: dict[str, Any] = {"configurable": dict(configurable or {})}
    final: dict[str, Any] = {}

    def events():
        nonlocal final
        for mode, data in graph.stream(
            state, config=cfg, stream_mode=["custom", "values"]
        ):
            if mode == "custom":
                yield data
            else:
                final = data
        # values 最后一帧是终态;兜底(空流)时至少返回入参 state
        if not final:
            final = dict(state)

    return events(), lambda: dict(final)


def react_loop(initial_messages: list,
               *,
               step_instruction: Optional[str] = None,
               max_steps: int = 6,
               max_total_seconds: int = 60,
               started_at: Optional[float] = None,
               checkpointer=None,
               configurable: Optional[dict[str, Any]] = None,
               bind_tools: bool = True,
               skip_rewrite: bool = False,
               skip_recall: bool = False,
               **state_kw):
    """独立运行一次 agent↔tools 循环(不依赖外层图)。

    :param initial_messages: 进入循环前的初始消息列表(system/history/user 等)。
    :param step_instruction: 可选,P&E 每步传入的步骤指令;会作为一条新 HumanMessage
        追加到 initial_messages 末尾(步骤间隔离)。
    :param max_steps/max_total_seconds: 有界执行上限。
    :param started_at: 请求开始时间戳;默认 now。跨步骤/断点续跑时应传入原始 t0,
        保证总预算正确。
    :param checkpointer: 子图 checkpointer(默认 None,无状态)。
    :param configurable: 透传给节点的运行时对象(trace_recorder/coverage_tracker/
        thread_id/user_id 等)。
    :param bind_tools: 是否给 LLM 绑定检索工具 schema。False(simple 直答)时不发
        tool_calls,模型只作答。
    :param skip_rewrite/skip_recall: 写入 state 的 tier 开关,供外层前置节点
        (rewrite_node/recall_node)按 tier 跳过;react_loop 自身只含 agent↔tools。
    :param state_kw: 其余 AgentState 字段(question/trace_id/full_reply/usage/
        collected_sources/retrieval_down/search_count 等)。
    :yield: SSE 事件 dict。
    :return: 最终 state dict(通过生成器 return 值,调用方用 ``.value`` 或
        ``StopIteration.value`` 获取)。
    """
    started_at = started_at if started_at is not None else time.time()
    messages = list(initial_messages)
    if step_instruction:
        from langchain_core.messages import HumanMessage
        messages.append(HumanMessage(content=step_instruction))

    state: dict[str, Any] = dict(state_kw)
    state.update({
        "messages": messages,
        "step": int(state.get("step") or 0),
        "started_at": started_at,
        "max_steps": max_steps,
        "max_total_seconds": max_total_seconds,
        "final_reason": None,
        "full_reply": state.get("full_reply", ""),
        "bind_tools": bind_tools,
        "skip_rewrite": skip_rewrite,
        "skip_recall": skip_recall,
    })

    evs, get_final = _run_react_stream(state, configurable or {})
    for ev in evs:
        yield ev
    # 终态 values 帧在最后一次 next 才被消费,循环结束后再取一次。
    return get_final()


def react_node(state: AgentState, config) -> dict[str, Any]:
    """主图节点:在一次节点调用内托管完整的 agent↔tools 循环。

    转发子图事件到父流(保持扁平 SSE 契约),并把子图终态以 reducer 友好的补丁
    回写父图。该节点执行期间不切换 LangGraph 节点,因此父图只需把它当作一个
    「会自己循环到出答案/终态」的黑盒节点。
    """
    from langgraph.config import get_stream_writer
    w = get_stream_writer()
    configurable = dict(config.get("configurable", {}))

    initial_messages = list(state.get("messages") or [])
    initial_usage = dict(state.get("usage") or {})

    evs, get_final = _run_react_stream(dict(state), configurable)
    for ev in evs:
        w(ev)  # 把子图事件扁平转发到父流
    # 注意:终态 values 帧是在生成器最后一次 next 时才被消费,循环体不会再执行,
    # 因此循环结束后必须再取一次 final(否则拿到的是上一帧/初始 state)。
    final = get_final()
    if not final:
        return {}

    # messages:只回写「本轮新增」的消息,避免把已存在于父图的初始消息重复 add。
    new_messages = final.get("messages", [])[len(initial_messages):]
    # usage:累加型 reducer,只回写增量(final 已在子图内从 initial 累加)。
    final_usage = final.get("usage") or {}
    delta_usage = {
        k: int(final_usage.get(k, 0)) - int(initial_usage.get(k, 0))
        for k in final_usage
    }
    delta_usage = {k: v for k, v in delta_usage.items() if v}

    patch: dict[str, Any] = {
        "messages": new_messages,
        "step": final.get("step", state.get("step", 0)),
        "final_reason": final.get("final_reason"),
        "full_reply": final.get("full_reply", ""),
        "retrieval_down": bool(final.get("retrieval_down")),
        "search_count": int(final.get("search_count") or 0),
        "tool_parse_errors": final.get("tool_parse_errors") or {},
    }
    # collected_sources 是合并 reducer,整体回写幂等(同 chunk_id 高分覆盖)。
    if final.get("collected_sources"):
        patch["collected_sources"] = final["collected_sources"]
    if delta_usage:
        patch["usage"] = delta_usage
    if final.get("error"):
        patch["error"] = final["error"]
    return patch
