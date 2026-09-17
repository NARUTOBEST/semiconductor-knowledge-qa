# -*- coding: utf-8 -*-
"""把 agent_node ↔ tools_node 的 ReAct 工具循环抽成可复用单元。

设计要点:
- ``build_react_graph()`` 编译一个只含 agent↔tools 两节点的子图(无 checkpointer),
  路由依据 ``final_reason``:有值(answer/timeout/max_steps/error)结束,否则继续 tools。
- ``react_loop()`` 是给上层/测试直接用的生成器:传入初始 messages 与运行参数,
  yield SSE 事件 dict,并通过 ``return`` 给出最终 state(调用方从
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


def _route_after_agent(state: AgentState) -> str:
    """agent 之后两路:有终态原因(answer/max_steps/timeout/error)→ end;否则→工具校验。

    recall_memory 已普通工具化(注册进 registry,Category.MEMORY),与检索三件套同走
    generation→runtime→execute→reflect 管线,不再有专用记忆节点/分叉。
    """
    if state.get("final_reason"):
        return "end"
    return "tools"


def _route_after_generation(state: AgentState) -> str:
    """生成阶段校验后:有合法调用→runtime 继续校验;全非法(错误 ToolMessage 已回灌)→回 agent。"""
    return "runtime" if (state.get("pending_tool_calls") or []) else "agent"


def _route_after_runtime(state: AgentState) -> str:
    """运行时校验后:有合法定型调用→execute 执行;全非法→回 agent(读纠错 ToolMessage 自纠)。"""
    return "execute" if (state.get("pending_tool_calls") or []) else "agent"


def build_react_graph(checkpointer=None):
    """编译 ReAct 工具循环子图(最终 state 由 values 流取得)。

    拓扑(工具调用错误处理按三阶段拆节点;机械重试在韧性中间件内,非图节点):

      START → agent ─(final_reason 有值)→ END
                    └→ validate_generation ─(有合法)→ validate_runtime ─(有合法)→ execute_tools
                          └(全非法)→ agent              └(全非法)→ agent        → reflect → agent

    - validate_generation:生成阶段兜底(结构坏/工具名幻觉 → 回灌错误,不执行);全非法直接回 agent。
    - validate_runtime:runtime 校验(JSON 解析/schema 参数/guard → 回灌错误);全非法直接回 agent。
      混合批(部分合法部分非法):合法的继续执行,非法的错误 ToolMessage 随流回 agent。
    - execute_tools:经韧性中间件(超时/熔断/限流/抖动/崩溃 透明重试)fan-out 执行;
      recall_memory 作为 Category.MEMORY 普通工具在此执行(失败经韧性链回灌显式错误)。
    - reflect:执行后决策(持久不可用降级摘工具 / 空结果低置信换词 hint)。
    决策节点都不终止图(终止由 agent 的 final_reason 决定),只回灌 ToolMessage/hint。

    默认无 checkpointer;独立调用 react_loop 时可传入以支持步骤内断点续跑。
    """
    from .validate_nodes import (
        validate_generation_node, validate_runtime_node, reflect_node,
    )
    b = StateGraph(AgentState)
    b.add_node("agent", nodes.agent_node)
    b.add_node("validate_generation", validate_generation_node)
    b.add_node("validate_runtime", validate_runtime_node)
    b.add_node("execute_tools", nodes.execute_tools_node)
    b.add_node("reflect", reflect_node)
    b.add_edge(START, "agent")
    b.add_conditional_edges(
        "agent", _route_after_agent,
        {"end": END, "tools": "validate_generation"},
    )
    # 校验回边(Req10):全非法(无合法调用)直接回 agent 自纠,不空转经过 execute。
    b.add_conditional_edges(
        "validate_generation", _route_after_generation,
        {"runtime": "validate_runtime", "agent": "agent"},
    )
    b.add_conditional_edges(
        "validate_runtime", _route_after_runtime,
        {"execute": "execute_tools", "agent": "agent"},
    )
    b.add_edge("execute_tools", "reflect")
    b.add_edge("reflect", "agent")
    return b.compile(checkpointer=checkpointer)


def _run_react_stream(state: AgentState, configurable: dict[str, Any],
                      checkpointer=None):
    """运行子图,返回 (events_iter, final_state_getter)。

    用 ``stream_mode=["custom","values"]``:custom 用于转发 SSE 事件,
    values 最后一帧即最终 state(无 checkpointer 时无法 get_state)。

    :param checkpointer: 子图 checkpointer。react_node 固定传 None(父图已
        持久化其输出);独立调用 react_loop 可按需传入以做断点续跑。
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
               max_steps: int = 6,
               max_total_seconds: int = 60,
               started_at: Optional[float] = None,
               checkpointer=None,
               configurable: Optional[dict[str, Any]] = None,
               bind_tools: bool = True,
               **state_kw):
    """独立运行一次 agent↔tools 循环(不依赖外层图)。

    :param initial_messages: 进入循环前的初始消息列表(system/history/user 等)。
    :param max_steps/max_total_seconds: 有界执行上限。
    :param started_at: 请求开始时间戳;默认 now。断点续跑时应传入原始 t0,
        保证总预算正确。
    :param checkpointer: 子图 checkpointer(默认 None,无状态)。
    :param configurable: 透传给节点的运行时对象(trace_recorder/thread_id/user_id 等)。
    :param bind_tools: 是否给 LLM 绑定检索工具 schema。False(simple 直答)时不发
        tool_calls,模型只作答。
    :param state_kw: 其余 AgentState 字段(question/trace_id/full_reply/usage/
        collected_sources/search_count 等)。
    :yield: SSE 事件 dict。
    :return: 最终 state dict(通过生成器 return 值,调用方用 ``.value`` 或
        ``StopIteration.value`` 获取)。
    """
    started_at = started_at if started_at is not None else time.time()
    messages = list(initial_messages)

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
        "tool_status": final.get("tool_status") or {},
        "search_count": int(final.get("search_count") or 0),
        "retrieval_max_score": float(final.get("retrieval_max_score") or 0.0),
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
