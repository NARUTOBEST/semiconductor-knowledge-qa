# -*- coding: utf-8 -*-
"""LangGraph 节点:把原 react_stream 生成器拆成图节点。

每个节点签名 (state: AgentState, config: RunnableConfig) -> dict,返回 state 更新。
- SSE/审计事件通过 get_stream_writer() 发出(结构与旧 react_stream 完全一致)。
- 不可序列化的 TraceRecorder 走 config["configurable"]["trace_recorder"]。
- OpenAI 客户端用同包 llm.get_client() 单例,不进 state。

复用(不重写):rewrite_query / build_messages / dispatch / TraceRecorder /
truncate_tool_result / sources_from_result / grounding_check / meta_event。
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.config import get_stream_writer

# 业务模块由 agent_reasoning 包 __init__ 与项目根 sys.path 提供
import config as C  # noqa: E402
from tools import search_tools as _ALL_TOOLS, dispatch  # noqa: E402
from support.metrics import metrics  # noqa: E402
from query_rewrite import rewrite_query  # noqa: E402
from message_builder import build_messages  # noqa: E402
from context_management import truncate_tool_result  # noqa: E402
from ..trace import TraceRecorder  # noqa: E402
from ..utils.sources import sources_from_result  # noqa: E402
from ..utils.events import meta_event  # noqa: E402

logger = logging.getLogger("agent")

from memories.storage.long import recall_memories, format_memories_for_prompt  # noqa: E402

from .state import AgentState  # noqa: E402
from ..support.llm import (  # noqa: E402
    get_client, llm_create_with_retry, LLM_TIMEOUT, STREAM_TIMEOUT,
)
from ..support.answer_grounding import grounding_check  # noqa: E402
from ..support.plan_grounding import CoverageTracker, _JUDGE_TIMEOUT  # noqa: E402
from ..support.planning import (  # noqa: E402
    generate_plan, looks_complex,
    MAX_PLAN_STEPS, PLAN_MIN_REMAINING_SECONDS,
)
from memories.storage.working.summarize import (  # noqa: E402
    maybe_summarize, format_summary_block, schedule_pregeneration,
)

# ==================== 常量 ====================
_TOOL_SCHEMAS = _ALL_TOOLS
_TOOL_NAMES = {t["function"]["name"] for t in _ALL_TOOLS}
_SEARCH_TOOL_NAMES = _TOOL_NAMES & {"search_text", "search_image"}
MAX_STEPS = 6
MAX_TOTAL_SECONDS = 60
MAX_REFLECT = 1   # grounding 校验失败后最多反思重生成的次数
MAX_COVERAGE_ROLLBACKS = 1  # 计划步骤未覆盖时最多回退重检索的次数
_MAX_TOOL_RETRIES = 1  # 工具瞬时异常自动重试次数
LOW_CONFIDENCE_THRESHOLD = 0.01
ARGS_PREVIEW_LEN = 60
RESULT_PREVIEW_LEN = 500


# ==================== 辅助 ====================
def _args_preview(args, limit=ARGS_PREVIEW_LEN):
    text = json.dumps(args, ensure_ascii=False, default=str)
    if len(text) > limit:
        text = text[:limit] + "…"
    return text


def _tool_call_signature(tc: dict[str, Any]) -> str:
    """用于死循环检测:忽略 tool_call_id,仅按 tool name + 排序后的 args 签名。"""
    args = tc.get("args") or {}
    try:
        args_str = json.dumps(args, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        args_str = str(args)
    return f"{tc.get('name')}:{args_str}"


def _last_tool_calls(messages: list[BaseMessage]) -> list[dict] | None:
    """取上一步 agent 节点生成的 tool_calls。

    当前 step 的 AIMessage 尚未写回 messages,因此最近一条带 tool_calls
    的 AIMessage 就是上一轮 agent 实际发出去的调用。
    """
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            tcs = getattr(m, "tool_calls", None)
            if tcs:
                return tcs
    return None


def _same_tool_calls(a: list[dict], b: list[dict]) -> bool:
    """比较两组 tool_calls 是否等价(忽略 id,只比 name+args)。"""
    if len(a) != len(b):
        return False
    sigs_a = sorted(_tool_call_signature(x) for x in a)
    sigs_b = sorted(_tool_call_signature(x) for x in b)
    return sigs_a == sigs_b


def _dispatch_with_retry(name, args, recorder, trace_id, step, w):
    """工具调用包装:未捕获异常(网络/IO 瞬时故障)自动重试 N 次。

    业务层面返回的 {error: ...} 不重试(dispatch 已把参数错误等确定性
    失败包装成 error);只有真正抛异常才走重试。
    """
    last_exc = None
    for attempt in range(_MAX_TOOL_RETRIES + 1):
        try:
            return dispatch(name, args)
        except Exception as e:
            last_exc = e
            err_doc = recorder.record_error(step, "tool_dispatch", e)
            w({"type": "error_trace", "trace_id": trace_id, **err_doc})
            if attempt < _MAX_TOOL_RETRIES:
                wait = 0.5 * (attempt + 1)
                logger.info(
                    "tool %s dispatch failed (attempt %d/%d), retry in %.1fs: %s",
                    name, attempt + 1, _MAX_TOOL_RETRIES + 1, wait, e)
                w({"type": "status",
                   "message": f"工具 {name} 调用异常,{wait:.1f}s 后重试…",
                   "trace_id": trace_id, "step": step})
                time.sleep(wait)
            else:
                break
    return {
        "error": (f"{name} 调度异常(已重试{_MAX_TOOL_RETRIES}次): "
                  f"{type(last_exc).__name__}: {last_exc}")
    }


def _format_plan_block(task_plan: dict[str, Any] | None,
                       search_count: int = 0) -> str:
    """把 task_plan 渲染为注入 system message 的计划文本块(含已检索进度)。

    search_count 是本轮已执行的检索工具次数,作为粗粒度进度信号提示模型:
    已经检索过若干轮,应在新资料基础上继续推进计划剩余步骤,而非重复开头的检索。
    """
    plan = task_plan or {}
    if not plan.get("need_plan"):
        return ""
    steps = [str(s).strip() for s in plan.get("steps") or [] if str(s).strip()]
    if not steps:
        return ""
    lines = "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1))
    progress = ""
    if search_count > 0:
        progress = (f"\n\n进度提示:你已经完成 {search_count} 次检索,"
                    "请基于已获得的资料继续推进尚未完成的计划步骤,"
                    "不要重复已经做过的检索;若资料已足够,直接综合作答。")
    return ("## 任务计划\n"
            "请按以下计划逐步检索,完成一步再进行下一步:\n" + lines + progress)


def _recorder(config) -> TraceRecorder:
    return config["configurable"]["trace_recorder"]


def _tracker(config) -> CoverageTracker | None:
    """取异步覆盖度追踪器;不存在(简单问题/断点续跑跨进程)返回 None。"""
    return config["configurable"].get("coverage_tracker")


def _msgs_to_openai(messages: list[BaseMessage]) -> list[dict]:
    """把 LangChain BaseMessage 列表回转成 OpenAI messages dict。"""
    out = []
    for m in messages:
        if isinstance(m, SystemMessage):
            out.append({"role": "system", "content": m.content})
        elif isinstance(m, HumanMessage):
            out.append({"role": "user", "content": m.content})
        elif isinstance(m, ToolMessage):
            out.append({
                "role": "tool",
                "tool_call_id": m.tool_call_id,
                "content": m.content,
            })
        elif isinstance(m, AIMessage):
            d: dict[str, Any] = {"role": "assistant", "content": m.content or ""}
            if m.tool_calls:
                d["tool_calls"] = [
                    {"id": tc["id"], "type": "function",
                     "function": {"name": tc["name"],
                                  "arguments": json.dumps(tc["args"], ensure_ascii=False)}}
                    for tc in m.tool_calls
                ]
            out.append(d)
        else:
            out.append({"role": "user", "content": str(getattr(m, "content", ""))})
    return out


def _extract_usage(chunk):
    u = getattr(chunk, "usage", None)
    if not u:
        return None
    return {
        "prompt_tokens": getattr(u, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(u, "completion_tokens", 0) or 0,
        "total_tokens": getattr(u, "total_tokens", 0) or 0,
    }


# ==================== 节点 ====================
def setup_node(state: AgentState, config) -> dict:
    """初始化本轮运行期字段,发出初始 status。

    State 经 checkpoint 跨轮持久化,除 step/retrieval_down 外,
    full_reply / collected_sources / error 等运行期字段若不重置,
    会把上一轮的回答、来源、错误带入下一轮(回答跨轮串联)。
    """
    w = get_stream_writer()
    trace_id = state.get("trace_id") or str(uuid.uuid4())[:8]
    w({"type": "status", "message": "理解问题中…", "trace_id": trace_id})
    return {
        "trace_id": trace_id,
        "step": 0,
        "retrieval_down": False,
        "full_reply": "",
        # 哨兵键:通知 _merge_sources reducer 清空跨轮残留(见 state.py)
        "collected_sources": {"__reset__": True},
        "tool_parse_errors": {},
        "error": None,
        "final_reason": None,
        "grounding": None,
        "reflect_count": 0,
        "reflect_feedback": "",
        "task_plan": {"need_plan": False},
        "search_count": 0,
        "coverage_rollbacks": 0,
    }


def recall_node(state: AgentState, config) -> dict:
    """召回长期记忆(失败降级为空,不阻断)。skip_recall=True 时直接跳过。"""
    if state.get("skip_recall"):
        return {"recalled_memories": []}
    user_id = config["configurable"].get("user_id") or state.get("user_id")
    question = state["question"]
    mems: list = []
    if user_id:
        try:
            mems = recall_memories(user_id, question)
        except Exception:
            mems = []
    return {"recalled_memories": mems}


def rewrite_node(state: AgentState, config) -> dict:
    """查询改写(失败回退原问题)。skip_rewrite=True 时不调 LLM,直接用原问题。"""
    recorder = _recorder(config)
    trace_id = state["trace_id"]
    w = get_stream_writer()
    question = state["question"]
    if state.get("skip_rewrite"):
        sub_queries = [question]
        recorder.sub_queries = list(sub_queries)
        return {"sub_queries": sub_queries}
    history = state.get("history") or []
    try:
        sub_queries = rewrite_query(question, history)
    except Exception as e:
        err = recorder.record_error(0, "query_rewrite", e)
        w({"type": "error_trace", "trace_id": trace_id, **err})
        sub_queries = [question]
    recorder.sub_queries = list(sub_queries)
    return {"sub_queries": sub_queries}


def plan_node(state: AgentState, config) -> dict:
    """复杂问题规划:粗筛命中才调一次 LLM 生成多步检索计划。

    计划写入 state.task_plan,由 build_messages_node 拼进 system message,
    引导 agent "按计划逐步检索,完成一步再进行下一步"。
    任何失败(LLM 异常/JSON 解析失败/无步骤)都降级为 need_plan=False,
    不阻断主流程。
    """
    w = get_stream_writer()
    recorder = _recorder(config)
    trace_id = state["trace_id"]
    question = state["question"]
    sub_queries = state.get("sub_queries") or []

    # 粗筛(medium 条件触发,6.1):单一意图/短问题不规划,省一次 LLM 调用
    if not looks_complex(question, sub_queries):
        return {"task_plan": {"need_plan": False}}

    # 剩余总预算不足时跳过规划(规划 LLM 调用最长 30s,会挤占生成时间)
    max_total = int(state.get("max_total_seconds") or MAX_TOTAL_SECONDS)
    remaining = max_total - (time.time() - state["started_at"])
    if remaining < PLAN_MIN_REMAINING_SECONDS:
        w({"type": "status",
           "message": "剩余响应时间不足,跳过规划直接回答…",
           "trace_id": trace_id})
        return {"task_plan": {"need_plan": False}}

    w({"type": "status", "message": "问题较复杂,正在制定检索计划…",
       "trace_id": trace_id})

    # 6.1/6.3:LLM 调用 + JSON 解析逻辑统一交给共享函数(与 complex P&E 同口径)。
    # force=False:允许 LLM 判定单点事实题返回空(不规划,静默降级)。
    steps, plan_err = generate_plan(
        question, force=False, trace_id=trace_id,
        max_steps=MAX_PLAN_STEPS,
    )
    if plan_err is not None:
        # 记录真实异常到 trace(错误可降级、不阻断回答),并给前端可见提示
        err_doc = recorder.record_error(0, "plan", RuntimeError(plan_err))
        w({"type": "error_trace", "trace_id": trace_id, **err_doc})
        w({"type": "status",
           "message": "检索规划服务因临时异常暂不可用,将直接检索作答…",
           "trace_id": trace_id})
        return {"task_plan": {"need_plan": False}}
    if not steps:
        # LLM 判定无需规划(单点事实题)或产出空步骤 -> 直接作答
        return {"task_plan": {"need_plan": False}}

    w({"type": "plan", "trace_id": trace_id,
       "steps": steps, "question": question})
    w({"type": "status", "message": f"已生成 {len(steps)} 步检索计划",
       "trace_id": trace_id})
    recorder.record_plan(steps, question)  # 事后 trace 可见
    # 启动异步覆盖度追踪:守护线程随检索资料到达持续维护覆盖文档,
    # 供 coverage_check_node 在 grounding 前做"计划每一步是否被资料覆盖"的判定。
    tracker = _tracker(config)
    if tracker is not None:
        tracker.set_plan(steps, question)
    return {"task_plan": {"need_plan": True, "steps": steps}}


def build_messages_node(state: AgentState, config) -> dict:
    """构建 LLM messages(system+history+user),并注入长期记忆与对话摘要。

    断点续跑/多轮:system 用固定 id(按 id 更新而非重复追加);已有 messages 时
    只追加本轮新 user 问题,不重复灌前端 history。跨轮消息过长时先做摘要压缩
    (summarize.maybe_summarize),旧轮次从 checkpoint 删除并并入 summary。
    """
    question = state["question"]
    sub_queries = state.get("sub_queries")
    recalled = state.get("recalled_memories") or []
    existing = state.get("messages") or []
    trace_id = state.get("trace_id", "")
    SYSTEM_MSG_ID = "system-prompt"

    summary = state.get("summary") or ""
    patch: dict[str, Any] = {}

    if existing:
        # 跨轮:先按需压缩旧消息(优先用临界点前的异步预生成摘要,否则同步兜底)
        thread_id = config.get("configurable", {}).get("thread_id")
        summ = maybe_summarize(state, trace_id, thread_id=thread_id)
        remove_msgs: list[BaseMessage] = []
        if summ:
            remove_msgs = summ.get("messages") or []
            if summ.get("summary"):
                summary = summ["summary"]
                patch["summary"] = summary

        raw_sys = build_messages(question, [], sub_queries=sub_queries)[0]
        extra_blocks = [b for b in (
            _format_plan_block(state.get("task_plan"),
                               int(state.get("search_count") or 0)),
            format_summary_block(summary),
            format_memories_for_prompt(recalled),
        ) if b]
        sys_content = raw_sys["content"] + (
            "\n\n" + "\n\n".join(extra_blocks) if extra_blocks else "")
        patch_msgs: list[BaseMessage] = remove_msgs + [
            SystemMessage(content=sys_content, id=SYSTEM_MSG_ID)]
        last = existing[-1]
        already = isinstance(last, HumanMessage) and last.content == question
        if not already:
            patch_msgs.append(HumanMessage(content=question))
        patch["messages"] = patch_msgs
        return patch

    history = state.get("history") or []
    raw = build_messages(question, history, sub_queries=sub_queries)
    extra_blocks = [b for b in (
        _format_plan_block(state.get("task_plan"),
                           int(state.get("search_count") or 0)),
        format_summary_block(summary),
        format_memories_for_prompt(recalled),
    ) if b]
    if extra_blocks and raw and raw[0].get("role") == "system":
        raw[0]["content"] = raw[0]["content"] + "\n\n" + "\n\n".join(extra_blocks)

    msgs: list[BaseMessage] = []
    for d in raw:
        role = d["role"]
        content = d["content"]
        if role == "system":
            msgs.append(SystemMessage(content=content, id=SYSTEM_MSG_ID))
        elif role == "assistant":
            msgs.append(AIMessage(content=content))
        else:
            msgs.append(HumanMessage(content=content))
    return {"messages": msgs}


def agent_node(state: AgentState, config) -> dict:
    """核心 LLM 调用节点:流式读 token、累积 tool_calls、判断终止/继续。"""
    w = get_stream_writer()
    recorder = _recorder(config)
    trace_id = state["trace_id"]
    t0 = state["started_at"]

    prev_step = int(state.get("step", 0))
    max_steps = int(state.get("max_steps") or MAX_STEPS)
    max_total_seconds = int(state.get("max_total_seconds") or MAX_TOTAL_SECONDS)

    # 已完整执行完 max_steps 轮且仍需继续 -> 停止(复刻 range(1,MAX_STEPS+1) 语义)
    if prev_step >= max_steps:
        w({"type": "status", "message": "已达最大推理步数,输出当前结果。",
           "trace_id": trace_id, "step": max_steps})
        return {"step": max_steps, "final_reason": "max_steps"}

    step = prev_step + 1
    elapsed = time.time() - t0
    if elapsed > max_total_seconds:
        w({"type": "status",
           "message": f"响应超时({int(elapsed)}s),输出当前结果。",
           "trace_id": trace_id, "step": step})
        return {"step": step, "final_reason": "timeout"}

    step_doc = recorder.new_step(step)
    w({"type": "step_start", "trace_id": trace_id, "step": step,
       "elapsed_ms": int((time.time() - t0) * 1000)})
    w({"type": "status", "message": "思考中…", "trace_id": trace_id, "step": step})

    # ---- LLM 调用 ----
    # bind_tools=False(simple 直答路径)时不传 tools schema,模型只生成文本、不会发 tool_calls。
    bind_tools = bool(state.get("bind_tools", True))
    llm_kwargs: dict[str, Any] = dict(
        model=C.OPENAI_TEXT_MODEL,
        messages=_msgs_to_openai(state["messages"]),
        stream=True, stream_options={"include_usage": True},
        temperature=0.3, timeout=STREAM_TIMEOUT,
    )
    if bind_tools:
        llm_kwargs["tools"] = _TOOL_SCHEMAS
        llm_kwargs["tool_choice"] = "auto"
    client = get_client()
    t_llm = time.time()
    stream, err = llm_create_with_retry(
        client, trace_id=trace_id, **llm_kwargs,
    )
    if err is not None:
        err_doc = recorder.record_error(step, "llm_create", err)
        w({"type": "error_trace", "trace_id": trace_id, **err_doc})
        w({"type": "error", "message": f"模型请求失败: {str(err)[:160]}",
           "trace_id": trace_id, "step": step})
        recorder.finish_step(step_doc, "error")
        recorder.final_reason = "error"
        return {"step": step, "final_reason": "error",
                "error": {"phase": "llm_create", "message": str(err)[:300]}}

    # ---- 读流 ----
    content_buf = ""
    full_reply = state.get("full_reply", "")
    tc_acc: dict[int, dict] = {}
    finish_reason = None
    usage = None
    ttft_ms = None
    t_first_delta = None
    try:
        for chunk in stream:
            choice = chunk.choices[0] if getattr(chunk, "choices", None) else None
            if choice is not None:
                delta = choice.delta
                if getattr(choice, "finish_reason", None):
                    finish_reason = choice.finish_reason
                if getattr(delta, "content", None):
                    if t_first_delta is None:
                        t_first_delta = time.time()
                        ttft_ms = int((t_first_delta - t_llm) * 1000)
                    content_buf += delta.content
                    full_reply += delta.content
                    w({"type": "token", "delta": delta.content,
                       "trace_id": trace_id, "step": step})
                if getattr(delta, "tool_calls", None):
                    if t_first_delta is None:
                        t_first_delta = time.time()
                        ttft_ms = int((t_first_delta - t_llm) * 1000)
                    for tc in delta.tool_calls:
                        idx = tc.index if tc.index is not None else 0
                        slot = tc_acc.setdefault(
                            idx, {"id": "", "name": "", "arguments": ""})
                        if tc.id:
                            slot["id"] = tc.id
                        fn = getattr(tc, "function", None)
                        if fn:
                            if fn.name:
                                slot["name"] = fn.name
                            if fn.arguments:
                                slot["arguments"] += fn.arguments
            chunk_usage = _extract_usage(chunk)
            if chunk_usage:
                usage = chunk_usage
    except Exception as e:
        err_doc = recorder.record_error(step, "llm_stream", e)
        w({"type": "error_trace", "trace_id": trace_id, **err_doc})
        w({"type": "error", "message": f"流读取中断: {str(e)[:160]}",
           "trace_id": trace_id, "step": step})
        recorder.finish_step(step_doc, "error")
        recorder.final_reason = "error"
        return {"step": step, "final_reason": "error", "full_reply": full_reply,
                "error": {"phase": "llm_stream", "message": str(e)[:300]}}

    stream_duration_ms = int((time.time() - t_llm) * 1000)

    # ---- 组装 tool_calls ----
    # ---- 死循环检测:模型连续两轮回发相同工具调用 ----
    prev_tcs = _last_tool_calls(state["messages"] or [])

    parsed_tc = []
    lc_tool_calls = []
    for _, v in sorted(tc_acc.items()):
        raw_args = v["arguments"] or ""
        parse_error = None
        try:
            args = json.loads(raw_args) if raw_args.strip() else {}
        except Exception as e:
            parse_error = f"{type(e).__name__}: {e}"
            args = {}
        parsed_tc.append({
            "id": v["id"], "name": v["name"], "args": args,
            "args_len": len(raw_args), "args_parse_error": parse_error,
        })
        lc_tool_calls.append({"id": v["id"], "name": v["name"], "args": args})

    recorder.record_llm(
        step_doc, finish_reason=finish_reason, usage=usage,
        thought=content_buf,
        tool_calls=[{
            "id": tc["id"], "name": tc["name"],
            "args_preview": _args_preview(tc["args"]),
            "args_len": tc["args_len"],
            "args_parse_error": tc["args_parse_error"],
        } for tc in parsed_tc],
        stream_duration_ms=stream_duration_ms,
        time_to_first_token_ms=ttft_ms,
    )
    if usage:
        metrics.record_tokens(usage["prompt_tokens"], usage["completion_tokens"])
    w({"type": "llm_response", "trace_id": trace_id, "step": step,
       "finish_reason": finish_reason, "usage": usage,
       "thought_preview": content_buf[:RESULT_PREVIEW_LEN]
           + ("…" if len(content_buf) > RESULT_PREVIEW_LEN else ""),
       "thought_len": len(content_buf), "has_tool_calls": bool(parsed_tc),
       "tool_calls": [{
           "id": tc["id"], "name": tc["name"],
           "args_preview": _args_preview(tc["args"]),
           "args_len": tc["args_len"],
           "args_parse_error": tc["args_parse_error"],
       } for tc in parsed_tc],
       "stream_duration_ms": stream_duration_ms,
       "time_to_first_token_ms": ttft_ms})

    # 若本轮 tool_calls 与上一轮完全一致,判定为工具调用循环,
    # 直接报错终止,不再进入 tools_node 空转。
    if lc_tool_calls and prev_tcs and _same_tool_calls(parsed_tc, prev_tcs):
        err_msg = ("检测到工具调用循环:模型重复发出了与上一轮完全相同的工具调用,"
                   "已终止执行以避免死循环。")
        logger.warning("tool call loop detected: %s",
                       [_tool_call_signature(tc) for tc in parsed_tc])
        w({"type": "error", "message": err_msg,
           "trace_id": trace_id, "step": step})
        recorder.final_reason = "error"
        return {
            "step": step,
            "final_reason": "error",
            "error": {"phase": "agent", "message": err_msg},
            "messages": [AIMessage(content=content_buf)],
        }

    ai_msg = (AIMessage(content=content_buf, tool_calls=lc_tool_calls)
              if lc_tool_calls else AIMessage(content=content_buf))
    patch: dict[str, Any] = {
        "step": step,
        "messages": [ai_msg],
        "full_reply": full_reply,
    }
    if usage:
        patch["usage"] = usage
    if lc_tool_calls:
        patch["final_reason"] = None  # 还要工具循环,清掉上一轮残留
        patch["tool_parse_errors"] = {
            tc["id"]: tc["args_parse_error"]
            for tc in parsed_tc
            if tc["args_parse_error"]
        }
    else:
        patch["final_reason"] = "answer"
    return patch


def tools_node(state: AgentState, config) -> dict:
    """执行最后一条 AIMessage 上的 tool_calls,回写 ToolMessage 与 sources。"""
    w = get_stream_writer()
    recorder = _recorder(config)
    trace_id = state["trace_id"]
    step = state["step"]
    step_doc = recorder.steps[-1] if recorder.steps else recorder.new_step(step)

    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", None) or []
    retrieval_down = bool(state.get("retrieval_down"))
    parse_errors = state.get("tool_parse_errors") or {}
    new_sources: dict[str, dict] = {}
    tool_msgs: list[BaseMessage] = []

    for tc in tool_calls:
        name = tc["name"]
        args = tc["args"] if isinstance(tc["args"], dict) else {}
        tcid = tc["id"]

        w({"type": "tool_call", "trace_id": trace_id, "step": step,
           "tool_call_id": tcid, "name": name, "args": args,
           "args_preview": _args_preview(args),
           "args_parse_error": parse_errors.get(tcid)})
        w({"type": "status",
           "message": f"调用工具 {name}({_args_preview(args)})…",
           "trace_id": trace_id, "step": step})

        t_tool = time.time()
        parse_error = parse_errors.get(tcid)
        if parse_error:
            # 参数 JSON 解析失败的调用不真正执行:
            # ① 避免 args={} 兜底触发模型从未意图的调用(全可选参的工具会真实执行,
            #    假结果/假来源还会混入 collected_sources 影响 grounding)
            # ② 把失败原因回传给 LLM,让其下一步自行修正重发
            result = {"error": f"参数 JSON 解析失败({parse_error}),请修正后重新调用 {name}"}
        else:
            result = _dispatch_with_retry(name, args, recorder, trace_id, step, w)
        duration_ms = int((time.time() - t_tool) * 1000)

        tool_ok = not (isinstance(result, dict) and result.get("error"))
        tool_error = result.get("error") if isinstance(result, dict) else None
        # 参数解析失败不算检索服务故障(缺参导致的 dispatch 报错
        # 曾把健康的检索服务误标为 retrieval_down)
        if name in _SEARCH_TOOL_NAMES and not tool_ok and not parse_error:
            retrieval_down = True
        metrics.record_tool_call(name, success=tool_ok)
        recorder.record_tool(
            step_doc, tool_call_id=tcid, name=name, args=args,
            ok=tool_ok, duration_ms=duration_ms, result=result, error=tool_error)
        result_size = len(json.dumps(result, ensure_ascii=False, default=str))
        w({"type": "tool_result", "trace_id": trace_id, "step": step,
           "tool_call_id": tcid, "name": name, "ok": tool_ok,
           "duration_ms": duration_ms, "result_size": result_size,
           "result_preview": json.dumps(result, ensure_ascii=False, default=str)[:RESULT_PREVIEW_LEN]
               + ("…" if result_size > RESULT_PREVIEW_LEN else ""),
           "error": tool_error})

        if name in _SEARCH_TOOL_NAMES:
            for _s in sources_from_result(result):
                key = _s.get("chunk_id") or (_s["source_stem"] + _s["page"])
                if key not in new_sources or _s["score"] > new_sources[key]["score"]:
                    new_sources[key] = _s

        tool_msgs.append(ToolMessage(
            content=truncate_tool_result(result, C.CONTEXT_TOOL_RESULT_MAX_CHARS),
            tool_call_id=tcid,
        ))

    new_count = len(new_sources)
    # 本轮已执行的检索工具调用次数(粗粒度计划进度信号)
    searched = sum(
        1 for tc in tool_calls
        if tc.get("name") in _SEARCH_TOOL_NAMES and not parse_errors.get(tc.get("id")))
    search_count = int(state.get("search_count") or 0) + searched
    recorder.finish_step(step_doc, "tool_calls", new_sources_count=new_count)
    w({"type": "step_end", "trace_id": trace_id, "step": step,
       "decision": "tool_calls", "new_sources_count": new_count,
       "elapsed_ms": step_doc.get("elapsed_ms")})

    collected = state.get("collected_sources") or {}
    total_sources = collected | new_sources
    # 把新到达的检索来源喂给异步覆盖度追踪器(守护线程据此增量重建覆盖文档)
    tracker = _tracker(config)
    if tracker is not None and new_sources:
        tracker.update_sources(new_sources)
    if total_sources:
        metrics.record_search(hit=True)
        w({"type": "sources", "items": list(total_sources.values())[:6],
           "trace_id": trace_id, "step": step})
        max_score = max(s["score"] for s in total_sources.values())
        if max_score < LOW_CONFIDENCE_THRESHOLD:
            w({"type": "status",
               "message": "⚠️ 检索置信度较低，以下回答仅供参考，建议核实原始文档。",
               "trace_id": trace_id, "step": step})
        else:
            w({"type": "status", "message": "已检索到资料,继续组织答案…",
               "trace_id": trace_id, "step": step})
    else:
        metrics.record_search(hit=False)
        if retrieval_down:
            w({"type": "status",
               "message": "⚠️ 检索服务暂时不可用，以下回答可能缺乏文献支持，请注意核实。",
               "trace_id": trace_id, "step": step})
        else:
            w({"type": "status", "message": "未检索到相关资料,尝试基于通用知识回答…",
               "trace_id": trace_id, "step": step})

    return {
        "messages": tool_msgs,
        "collected_sources": new_sources,
        "retrieval_down": retrieval_down,
        "search_count": search_count,
    }


def _maybe_coverage_rollback(state: AgentState, config) -> dict | None:
    """计划步骤覆盖判定 + 回退。

    调用异步 CoverageTracker 的 LLM 判定(基于其持续维护的覆盖文档):
    若某计划步骤未被已检索资料覆盖且预算(回退次数/总时长/步数)未耗尽,
    返回一个 state patch,把 agent 回退到未覆盖的步骤重新检索;
    否则(全部覆盖 / 判定不可用 / 预算耗尽)返回 None,交由后续 grounding 处理。
    """
    w = get_stream_writer()
    recorder = _recorder(config)
    tracker = _tracker(config)
    if tracker is None:
        return None

    trace_id = state["trace_id"]
    step = state["step"]
    full_reply = state.get("full_reply", "")
    task_plan = state.get("task_plan") or {}
    steps = [str(s).strip() for s in task_plan.get("steps") or [] if str(s).strip()]
    if not steps:
        return None

    box = tracker.request_judgment(full_reply)
    if box is None:
        return None
    verdict = box.result(timeout=_JUDGE_TIMEOUT)
    if verdict is None:
        # 判定超时/LLM 不可用/解析失败:fail-open,不阻断主流程,
        # 但给前端一个可见提示;后续 grounding 仍会做忠实度校验兜底。
        w({"type": "status",
           "message": "计划忠实度检测因临时异常暂不可用,将直接进行来源忠实度校验…",
           "trace_id": trace_id, "step": step})
        return None

    recorder.record_coverage(verdict)
    uncovered = [i for i in verdict.get("uncovered_steps", [])
                 if isinstance(i, int) and 1 <= i <= len(steps)]
    if not uncovered:
        return None  # 所有步骤均已覆盖

    rollbacks = int(state.get("coverage_rollbacks") or 0)
    budget_left = (time.time() - state["started_at"]
                   < int(state.get("max_total_seconds") or MAX_TOTAL_SECONDS))
    max_steps = int(state.get("max_steps") or MAX_STEPS)
    if rollbacks >= MAX_COVERAGE_ROLLBACKS or not budget_left or step >= max_steps:
        # 预算耗尽:不再回退,带覆盖警示进入 grounding/收尾
        w({"type": "status",
           "message": ("⚠️ 部分计划步骤资料不足,但已达回退上限/预算,"
                       "将基于现有资料作答并提示核实。"),
           "trace_id": trace_id, "step": step})
        return None

    first = uncovered[0]
    target_step = steps[first - 1]
    reason = verdict.get("reason") or "该步骤未被现有检索资料覆盖"
    w({"type": "status",
       "message": f"计划第 {first} 步资料不足,回退重新检索:{target_step}",
       "trace_id": trace_id, "step": step})
    w({"type": "coverage", "trace_id": trace_id, "step": step,
       "uncovered_steps": uncovered, "reason": reason,
       "rollback_count": rollbacks + 1})
    instruction = (
        f"计划的第 {first} 步【{target_step}】尚未被已检索资料覆盖({reason})。"
        "请针对该步骤补充调用检索工具(search_text/search_image)查找相关资料,"
        "拿到该步骤的资料后再综合全部计划步骤作答;不要重复已经检索过的内容。"
    )
    return {
        "coverage_rollbacks": rollbacks + 1,
        "final_reason": None,   # 回到 agent 重新检索
        "full_reply": "",       # 旧回答作废,从重检索后重新累积
        "reflect_feedback": f"回退到计划第{first}步: {reason}",
        "messages": [HumanMessage(
            content=instruction,
            id=f"rollback-{trace_id}-{rollbacks}",
        )],
    }


def coverage_check_node(state: AgentState, config) -> dict:
    """计划步骤覆盖度判定 + 回退(原 reflect_node 的第①部分,1.2 拆出)。

    由异步 CoverageTracker 维护的覆盖文档 + LLM 判定:若某计划步骤未被已检索
    资料覆盖且预算未耗尽,回退到该步骤重新检索(rollback),回到 react 循环。
    不适用(无计划/非 answer 终态/timeout/max_steps)或判定通过/不可用时返回
    空 patch,交由后续 grounding_node 处理。
    """
    decision = state.get("final_reason") or "answer"
    full_reply = state.get("full_reply", "")
    # 仅对"正常作答完成且有计划"的情形判定;timeout/max_steps 无步可退,直接进 grounding。
    if not (full_reply.strip() and decision == "answer"
            and (state.get("task_plan") or {}).get("need_plan")):
        return {}
    return _maybe_coverage_rollback(state, config) or {}


def grounding_node(state: AgentState, config) -> dict:
    """引用校验 + 忠实度检测 + 反思重生成(原 reflect_node 的第②部分,1.2 拆出)。

    grounding 失败且预算允许时,反思重生成:置 final_reason=None 并写入
    reflect_feedback,回到 react 循环;否则把 grounding 结果写入 state 进入收尾。
    """
    w = get_stream_writer()
    recorder = _recorder(config)
    trace_id = state["trace_id"]
    step = state["step"]
    decision = state.get("final_reason") or "answer"
    full_reply = state.get("full_reply", "")
    collected_sources = state.get("collected_sources") or {}
    reflect_count = int(state.get("reflect_count") or 0)

    grounding = None
    retrieval_down = bool(state.get("retrieval_down"))
    if full_reply.strip() and decision in ("answer", "timeout", "max_steps"):
        if collected_sources:
            w({"type": "status", "message": "验证答案来源…", "trace_id": trace_id})
            try:
                grounding = grounding_check(full_reply, list(collected_sources.values()))
                metrics.record_grounding(grounding["passed"])
                recorder.record_grounding(grounding["passed"], grounding["warnings"])
                if not grounding["passed"]:
                    for warn in grounding["warnings"]:
                        w({"type": "status", "message": f"⚠️ {warn}", "trace_id": trace_id})
                w({"type": "grounding", "trace_id": trace_id, "step": step,
                   "passed": grounding["passed"], "warnings": grounding["warnings"]})
            except Exception:
                # 校验服务异常:不伪造"通过",带上可见警示;不吞日志
                logger.exception("grounding_check 执行异常,跳过本轮校验")
                grounding = {"passed": True,
                             "warnings": ["来源校验服务异常,本轮答案未完成验证"]}
                w({"type": "status", "message": "⚠️ 来源校验服务异常,本轮答案未完成验证",
                   "trace_id": trace_id})
        elif not retrieval_down:
            # 无任何检索来源且检索服务正常:模型未检索就作答,存在编造风险(缺口1)。
            # 标记不通过,触发反思,引导其先检索再答;retrieval_down 时不反思
            # (工具已警示用户,且反思也无法补检索)。
            grounding = {"passed": False,
                         "warnings": ["回答未基于任何检索资料,存在编造风险,请先检索再作答"]}
            metrics.record_grounding(False)
            recorder.record_grounding(False, grounding["warnings"])
            w({"type": "status",
               "message": "⚠️ 回答未基于检索资料,将重新检索后作答",
               "trace_id": trace_id})
            w({"type": "grounding", "trace_id": trace_id, "step": step,
               "passed": False, "warnings": grounding["warnings"]})

    # ---- 是否允许再生成一轮(有界:次数 + 总时长;步数由 react 的 max_steps 兜底)----
    can_retry = (
        decision == "answer"
        and grounding is not None and not grounding["passed"]
        and full_reply.strip()
        and reflect_count < MAX_REFLECT
        and time.time() - state["started_at"] < int(
            state.get("max_total_seconds") or MAX_TOTAL_SECONDS)
    )
    if can_retry:
        no_sources = not collected_sources and not retrieval_down
        feedback = ";".join(grounding["warnings"]) or "回答与检索资料不符"
        w({"type": "status",
           "message": f"回答未通过来源校验,第 {reflect_count + 1} 次反思重生成…",
           "trace_id": trace_id, "step": step})
        # 通知前端重置正在流式输出的消息内容(旧回答作废,只展示重生成结果)
        w({"type": "reflect", "trace_id": trace_id, "step": step,
           "reflect_count": reflect_count + 1, "feedback": feedback})
        if no_sources:
            instruction = (
                "你刚才未检索任何资料就直接作答,这不符合要求。"
                "请先调用检索工具(search_text/search_image)查找相关资料,"
                "再严格依据检索结果组织回答;若确实检索不到,请明确说明,"
                "不要凭空编造。"
            )
        else:
            instruction = (
                "你刚才的回答未通过来源忠实度校验,不要原样重复。"
                f"修正意见:{feedback}。"
                "若现有资料不足以支撑某个结论,请补充检索后再作答;"
                "确实检索不到时请明确说明,不要编造。"
            )
        return {
            "reflect_count": reflect_count + 1,
            "reflect_feedback": feedback,
            "grounding": grounding,
            "final_reason": None,   # 回到 react 再生成(可继续调工具补检索)
            "full_reply": "",       # 重生成从零累积,避免新旧答案串联
            "messages": [HumanMessage(
                content=instruction,
                id=f"reflect-{trace_id}-{reflect_count}",
            )],
        }

    return {"grounding": grounding}


def finalize_node(state: AgentState, config) -> dict:
    """收尾:meta + done。grounding 已由 reflect_node 完成并写入 state。"""
    w = get_stream_writer()
    recorder = _recorder(config)
    trace_id = state["trace_id"]
    t0 = state["started_at"]
    step = state["step"]
    decision = state.get("final_reason") or "answer"
    full_reply = state.get("full_reply", "")
    collected_sources = state.get("collected_sources") or {}

    grounding = state.get("grounding")
    grounding_passed = grounding["passed"] if grounding else None

    step_doc = recorder.steps[-1] if recorder.steps else None
    if step_doc is not None:
        recorder.finish_step(step_doc, decision,
                             new_sources_count=step_doc.get("new_sources_count", 0))
        w({"type": "step_end", "trace_id": trace_id, "step": step,
           "decision": decision,
           "new_sources_count": step_doc.get("new_sources_count", 0),
           "elapsed_ms": step_doc.get("elapsed_ms")})

    recorder.final_reason = decision
    w(meta_event(
        trace_id, t0, step, collected_sources,
        tokens=recorder.total_tokens,
        tools_count=sum(len(s["tools"]) for s in recorder.steps),
        grounding_passed=grounding_passed,
    ))
    if full_reply.strip():
        w({"type": "assistant_message", "trace_id": trace_id, "content": full_reply})
    w({"type": "done", "trace_id": trace_id, "trace": recorder.to_dict()})

    patch: dict[str, Any] = {
        "final_reason": decision,
        "grounding": grounding,
        "full_reply": full_reply,
    }
    if state.get("error"):
        patch["error"] = state["error"]

    # 临界前预生成:用"本轮结束后"的 messages(即下一轮所见 existing)计算下一轮
    # 将被删除的最老轮次,后台线程提前算摘要,避免下一轮同步调 LLM 阻塞。
    try:
        thread_id = config.get("configurable", {}).get("thread_id")
        schedule_pregeneration(
            state.get("messages") or [], state.get("summary") or "",
            thread_id, trace_id,
        )
    except Exception:
        pass  # 预生成失败不影响本轮收尾
    return patch
