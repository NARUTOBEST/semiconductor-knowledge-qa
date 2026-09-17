# -*- coding: utf-8 -*-
"""ReAct 工具调用的三个【决策类】错误处理节点(贯穿工具调用三阶段)。

与「机械重试」中间件(support/tool_resilience,透明处理超时/熔断/限流/抖动)不同,
这三个节点处理需要【判断 / 反馈给模型 / 改变后续行为】的错误,把每个失败调用转成一条
带 tool_call_id 的 ToolMessage 回灌给 LLM 自纠,合法调用才继续向下:

  agent → validate_generation → validate_runtime → execute_tools → reflect → agent

  - validate_generation(阶段A 生成):tool_call 结构坏 / 缺 id / 工具名幻觉(不存在或
    disabled)→ 回灌错误,不进后续节点;合法调用放入 pending_tool_calls。
  - validate_runtime(阶段B 运行时):arguments JSON 解析失败 / schema 参数校验不通过
    (类型/enum/必填/超长)/ 安全 guard 拦截 → 回灌错误;合法参数定型后透传。
  - reflect(阶段D 二次推理决策):读 execute 的 tool_outcomes,对持久不可用(熔断/鉴权/
    崩溃,或重试类连续失败超阈值)标记 tool_status=down(下轮 agent 摘 schema + 降级常识),
    并对空结果/低置信注入换词再检索 hint。

节点签名 (state, config) -> dict(返回 state 更新);不直接终止图(终止由 agent 的
final_reason 决定)。失败明细优先走 messages 里的错误 ToolMessage。
"""
from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.config import get_stream_writer

import config as C  # noqa: E402
from tools import registry  # noqa: E402
from ..support.tool_validate import validate_arguments  # noqa: E402
from ..support.tool_errors import (  # noqa: E402
    Stage, Kind, ToolCallError, error_tool_message, MECHANICAL_KINDS,
)

logger = logging.getLogger("agent")

# 同一工具在【本次请求】内连续失败达到该次数后,标记其类别 down(剩余轮次摘除)。
# 熔断(circuit_open)/鉴权(auth)属确定性不可用,立即 down,不等阈值。
REQUEST_FAIL_STREAK_THRESHOLD = 2

LOW_CONFIDENCE_THRESHOLD = 0.01


# ==================== 节点①:生成阶段校验 ====================
def validate_generation_node(state, config) -> dict:
    """校验 LLM 刚产出的 tool_calls 的【结构与工具名】。

    非法调用(缺 name / name 非字符串 / 工具不存在 / disabled)生成错误 ToolMessage 回灌,
    不进入 runtime 校验与执行;合法调用整理为 pending_tool_calls 透传。
    """
    w = get_stream_writer()
    trace_id = state["trace_id"]
    step = state["step"]

    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", None) or []

    msgs: list[ToolMessage] = []
    pending: list[dict[str, Any]] = []

    for idx, tc in enumerate(tool_calls):
        name = tc.get("name")
        real_id = tc.get("id")
        # 缺 id 的调用无法与 ToolMessage 配对(LangGraph 消息校验要求 tool_call_id
        # 对应 AIMessage 上的某个 tool_call);这类格式错误改用 SystemMessage 反馈。
        tcid = real_id or f"generr-{step}-{idx}"

        def feedback(err: ToolCallError, label: str):
            """有真实 id 用 ToolMessage(可配对);无 id 退回 SystemMessage。"""
            if real_id:
                msgs.append(error_tool_message(tcid, err))
            else:
                msgs.append(SystemMessage(content=f"【工具调用格式错误】{err.message}"))
            _emit_failure(w, trace_id, step, tcid, label, err)

        # 结构/格式:name 缺失或非字符串
        if not name or not isinstance(name, str):
            feedback(
                ToolCallError(Stage.GENERATION, Kind.FORMAT,
                              "工具调用缺少有效的工具名 name 字段",
                              tool=str(name) if name else "?"),
                str(name or "?"))
            continue

        # 工具名幻觉:不存在或已禁用
        spec = registry.get(name)
        if spec is None or not spec.enabled:
            feedback(
                ToolCallError(Stage.GENERATION, Kind.UNKNOWN_TOOL,
                              f"可用工具为: {', '.join(sorted(registry.names()))}",
                              tool=name),
                name)
            logger.info("工具名幻觉拦截: %s(第 %d 步)", name, step)
            continue

        pending.append({"id": tcid, "name": name,
                        "args": tc.get("args") if isinstance(tc.get("args"), dict) else {}})

    if not pending:
        w({"type": "status",
           "message": "工具调用有误,已反馈模型修正…", "trace_id": trace_id, "step": step})

    return {"messages": msgs, "pending_tool_calls": pending}


# ==================== 节点②:运行时参数校验 ====================
def validate_runtime_node(state, config) -> dict:
    """对①放行的调用做【参数解析与 schema 校验】。

    JSON 解析失败(agent 累积流时已记入 tool_parse_errors)/ 类型·enum·必填·超长不符 /
    guard 拦截 → 生成错误 ToolMessage 回灌、不执行;合法参数经类型规范化后透传给 execute。
    """
    w = get_stream_writer()
    trace_id = state["trace_id"]
    step = state["step"]

    pending = state.get("pending_tool_calls") or []
    parse_errors = state.get("tool_parse_errors") or {}

    msgs: list[ToolMessage] = []
    valid: list[dict[str, Any]] = []

    for call in pending:
        tcid = call["id"]
        name = call["name"]
        spec = registry.get(name)

        # 1) arguments JSON 解析失败(机械解析在 agent 流组装期完成,此处做决策回灌)
        perr = parse_errors.get(tcid)
        if perr:
            err = ToolCallError(Stage.RUNTIME, Kind.JSON_PARSE, str(perr), tool=name)
            msgs.append(error_tool_message(tcid, err))
            _emit_failure(w, trace_id, step, tcid, name, err)
            continue

        # 2) schema 校验 + 类型规范化
        clean, errors, unknown = validate_arguments(spec, call.get("args") or {})
        if errors:
            err = ToolCallError(Stage.RUNTIME, Kind.SCHEMA_VIOLATION,
                                "; ".join(errors), tool=name)
            msgs.append(error_tool_message(tcid, err))
            _emit_failure(w, trace_id, step, tcid, name, err)
            logger.info("参数 schema 校验拦截 %s: %s", name, "; ".join(errors))
            continue

        # 3) 安全/业务 guard 钩子(可选;ToolSpec 上挂 guard(clean_args)-> 拦截原因 str 或 None)
        guard = getattr(spec, "guard", None)
        if callable(guard):
            try:
                block_reason = guard(clean)
            except Exception as e:  # guard 自身异常按拦截处理,保守不放行
                block_reason = f"校验器异常: {e}"
            if block_reason:
                err = ToolCallError(Stage.RUNTIME, Kind.BLOCKED, str(block_reason), tool=name)
                msgs.append(error_tool_message(tcid, err))
                _emit_failure(w, trace_id, step, tcid, name, err)
                continue

        if unknown:
            # 未知参数已被剔除(不进 clean);不额外发 ToolMessage(避免同一 tool_call_id 两条),
            # 仅状态提示,模型下轮可见调用未按预期生效。
            w({"type": "status",
               "message": f"工具 {name} 的无效参数 {unknown} 已忽略",
               "trace_id": trace_id, "step": step})

        valid.append({"id": tcid, "name": name, "args": clean})

    return {"messages": msgs, "pending_tool_calls": valid}


# ==================== 节点③:执行后决策(研判 / 退避 / 降级) ====================
def reflect_node(state, config) -> dict:
    """读 execute_tools 产出的 tool_outcomes,做基于错误/质量的决策。

    - 持久不可用:熔断/鉴权/崩溃 → 立即标该类别 down;超时/上游/限流在本请求连续失败
      达阈值 → 标 down(下轮 agent 健康度自适应摘 schema + 注入降级常识提示)。
    - 空结果 / 低置信检索且仍有步数 → 注入换关键词再检索的 system hint。
    错误 ToolMessage 已由 execute 节点按 tool_call_id 生成;本节点只追加决策 hint / 状态。
    """
    w = get_stream_writer()
    trace_id = state["trace_id"]
    step = state["step"]
    outcomes = state.get("tool_outcomes") or []

    tool_status = dict(state.get("tool_status") or {})
    streak = dict(state.get("tool_fail_streak") or {})
    requery = dict(state.get("tool_requery_count") or {})
    hint_msgs: list[SystemMessage] = []

    # ---- 失败研判:请求级连续失败退避 + 持久不可用降级 ----
    for oc in outcomes:
        if oc.get("ok"):
            streak[oc["name"]] = 0
            continue
        kind = oc.get("kind")
        name = oc["name"]
        cat = oc.get("category")
        streak[name] = streak.get(name, 0) + 1

        # 机械类错误(超时/上游/限流/崩溃/熔断/鉴权/预算超时)到达 reflect,说明韧性中间件
        # 已重试过仍失败(或熔断/鉴权确定性不可用)——即服务本请求内持久不可用,立即标记
        # 该类别 down(下轮 agent 摘 schema + 降级常识);非机械类(理论上不该出现在 outcome)
        # 走请求级连续失败阈值。
        immediate_down = kind in MECHANICAL_KINDS
        streak_down = streak[name] >= REQUEST_FAIL_STREAK_THRESHOLD
        if cat and (immediate_down or streak_down):
            tool_status[cat] = "down"
            w({"type": "status",
               "message": f"工具 {name} 持续失败,已暂时切换备用策略…",
               "trace_id": trace_id, "step": step})
            logger.info("reflect 标记 %s 类别 down(kind=%s, streak=%d)",
                        name, kind, streak[name])

    # ---- 空结果 / 低置信:换词再检索 hint(仅当本轮确实执行了工具)----
    # outcomes 为空说明所有调用都在校验阶段被拦(幻觉名/参数错),纠错 ToolMessage 已指引,
    # 不再误发"未检索到资料"提示。
    max_steps = int(state.get("max_steps") or 6)
    can_retry = bool(getattr(C, "REACT_ADAPTIVE_RETRIEVAL", True)) and step < max_steps - 1
    collected = state.get("collected_sources") or {}
    max_score = float(state.get("retrieval_max_score") or 0.0)
    any_empty = any(oc.get("empty") for oc in outcomes)

    # Req8:每工具换词计数。本轮涉及的检索类工具(产生来源者),换词提示按工具名累计,
    # 达 REACT_TOOL_REQUERY_MAX 后不再引导重试,改提示"按内部资料未覆盖作答"(不额外耗步)。
    max_requery = int(getattr(C, "REACT_TOOL_REQUERY_MAX", 2))
    retrieval_tools = sorted({
        oc["name"] for oc in outcomes
        if oc.get("produces_sources") or oc.get("category") == "retrieval"})

    def _exhausted() -> bool:
        """涉及的检索工具是否都已达换词上限(无工具可再引导重试)。"""
        if not retrieval_tools:
            return True
        return all(requery.get(n, 0) >= max_requery for n in retrieval_tools)

    def _bump_requery() -> None:
        for n in retrieval_tools:
            requery[n] = requery.get(n, 0) + 1

    _STOP_HINT = (
        "【多次检索未命中】已用多个关键词反复检索仍无对口资料,请【不要再调用检索工具】,"
        "直接基于已有信息与通用知识作答;若内部资料确实未覆盖该问题,请如实说明"
        "“内部资料未覆盖该内容”,不要在资料不足时强行下结论。")

    def _reword_hint(lead: str) -> str:
        return (
            lead +
            "若这是设备型号/报警代码/操作步骤/规格参数类问题,请【换用设备型号、报警代码、"
            "工序别名、故障现象等关键词】再调用 search_text/search_image 检索一次,"
            "或用 get_chunk 取命中块完整内容;不要在资料不足时直接下结论。"
            "若换关键词后仍无高相关结果,再按「内部资料未覆盖」如实说明。")

    if not outcomes:
        pass
    elif (not collected and can_retry) or \
            ((collected and max_score < C.RETRIEVAL_CONFIDENT_SCORE and can_retry) or any_empty):
        # 该分支统一处理"无资料/低置信/空结果"。达换词上限 -> 停止引导;否则注入换词 hint。
        if _exhausted() or not can_retry:
            hint_msgs.append(SystemMessage(content=_STOP_HINT))
            w({"type": "status",
               "message": "多次检索无结果,按内部资料未覆盖口径作答…",
               "trace_id": trace_id, "step": step})
        else:
            lead = ("【未检索到资料】本轮工具调用没有返回任何可用结果。"
                    if not collected else
                    f"【检索置信度提示】本轮检索到的资料最高相关分仅 {max_score:.2f},"
                    f"低于可信阈值 {C.RETRIEVAL_CONFIDENT_SCORE:.2f},很可能未命中对口资料。")
            hint_msgs.append(SystemMessage(content=_reword_hint(lead)))
            _bump_requery()
            w({"type": "status",
               "message": "未检索到资料,正在换关键词重新检索…",
               "trace_id": trace_id, "step": step})
    elif collected and max_score < LOW_CONFIDENCE_THRESHOLD:
        w({"type": "status",
           "message": "⚠️ 检索置信度较低,以下回答仅供参考,建议核实原始文档。",
           "trace_id": trace_id, "step": step})

    patch: dict[str, Any] = {
        "messages": hint_msgs,
        "tool_status": tool_status,
        "tool_fail_streak": streak,
        "tool_requery_count": requery,
        "tool_outcomes": [],   # 每轮清空(消费完毕)
    }
    return patch


def _emit_failure(w, trace_id, step, tcid, name, err: ToolCallError) -> None:
    """校验失败补发一条 tool_result(ok=false)SSE,前端可见失败原因。

    result_preview 与回灌 LLM 的 ToolMessage 用同一份面向模型的纠错文案(而非裸
    异常串),保证前端展示和模型看到的指引一致;error 字段保留原始 err.message。
    """
    try:
        guidance = error_tool_message(tcid, err).content
        w({"type": "tool_result", "trace_id": trace_id, "step": step,
           "tool_call_id": tcid, "name": name, "ok": False,
           "duration_ms": 0, "result_size": 0, "result_preview": guidance[:200],
           "error": err.message, "error_type": err.kind, "stage": err.stage})
    except Exception:
        pass
