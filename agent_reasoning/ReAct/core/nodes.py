# -*- coding: utf-8 -*-
"""LangGraph 节点:主链路「检索 → ReAct 工具调用循环 → 回答」。

图拓扑:START → setup → build_messages → react → finalize → END。
react 节点(loop.react_node)内部托管 agent↔tools 子图自循环到出答案/终态。

每个节点签名 (state: AgentState, config: RunnableConfig) -> dict,返回 state 更新。
- SSE/审计事件通过 get_stream_writer() 发出。
- 不可序列化的 TraceRecorder 走 config["configurable"]["trace_recorder"]。
- OpenAI 客户端用同包 llm.get_client() 单例,不进 state。
"""
from __future__ import annotations

import inspect
import json
import logging
import re
import time
import uuid

# 过程性叙述误当终答的检测(GLM 偶发:输出"我再检索一下…"这类中间独白却不再调工具,
# 文本无 tool_calls 即被判为 final answer)。命中则旁路重问一次(见 agent_node)。
_NARRATION_RE = re.compile(
    r"(我再?检索|我需要检索|再检索一下|预检索|换一?[个组批]?(不同|其他)?关键词|"
    r"未直接命中|让我(检索|查)|接下来(我|先|再)|先(检索|查|看)|继续(检索|查)|"
    r"我(继续|再)(查|找|看))")

# 记忆指令型问题("请记住…"类):答案是对指令的确认而非知识问答,零检索也正当。
_MEMORY_INSTR_RE = re.compile(r"记(住|下|着|一?下)|帮我记|记住一?个")
from concurrent.futures import TimeoutError as FuturesTimeout
from typing import Any, Optional

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
from tools import dispatch, registry, Category, ALL_CATEGORIES  # noqa: E402
from tools.base import cache_hit_var  # noqa: E402
from ..support.tool_circuit import circuit_snapshot  # noqa: E402
from ..support.tool_resilience import call_with_resilience  # noqa: E402
from ..support.tool_errors import Stage, Kind, ToolCallError, error_tool_message  # noqa: E402
from support.metrics import metrics  # noqa: E402
from message_builder import build_messages  # noqa: E402
from context_management import truncate_tool_result  # noqa: E402
from ..trace import TraceRecorder  # noqa: E402
from ..utils.sources import sources_from_result, final_citation_cards  # noqa: E402
from ..utils.events import meta_event  # noqa: E402

logger = logging.getLogger("agent")

from .state import AgentState  # noqa: E402
from ..support.llm import (  # noqa: E402
    get_client, llm_create_with_retry, STREAM_TIMEOUT, no_think_extra,
    arm_stream_watchdog,
)
from ..support.tool_pool import get_pool  # noqa: E402
from memories.storage.working.summarize import format_summary_block  # noqa: E402

# recall_memory 现为注册进 registry 的普通工具(Category.MEMORY);仅需其名字做匿名过滤。
try:
    from memories.orchestration import MEMORY_TOOL_NAME  # noqa: E402
except Exception:  # noqa: BLE001
    MEMORY_TOOL_NAME = "recall_memory"

# ==================== 常量 ====================
# 工具 schema 在模块加载时从 registry 取快照。注册发生在 tools 包 import 期,
# 此处取到的即全部已注册且启用工具(检索三件套)。
# 工具 schema 每次活取 registry.schemas()(不再 import 期快照):
# MCP 桥是后台线程异步注册的,晚连上的工具须下一请求即生效。

MAX_STEPS = 6
MAX_TOTAL_SECONDS = 60
LOW_CONFIDENCE_THRESHOLD = 0.01
ARGS_PREVIEW_LEN = 60
RESULT_PREVIEW_LEN = 500

# setup 每轮把各工具类别可用性复位为 "up";tools_node 按故障把某类别标 "down"。
_DEFAULT_TOOL_STATUS = {cat: "up" for cat in ALL_CATEGORIES}


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


def _no_source_status(tool_status: dict[str, str]) -> str:
    """本轮所有工具都没拿到来源时,按故障 category 给前端不同提示。

    tool_status[category] 为 "up"/"down"(跨轮粘性,setup 复位)。
    """
    if tool_status.get(Category.RETRIEVAL) == "down":
        return ("⚠️ 检索服务暂时不可用，以下回答可能缺乏文献支持，请注意核实。")
    return "未检索到相关资料,尝试基于通用知识回答…"


# 各工具类别故障时给模型的人类可读名称与替代策略(注入 system 提示,引导自适应)
_CATEGORY_HEALTH = {
    Category.RETRIEVAL: {
        "label": "本地知识库检索",
        "fallback": "基于通用知识谨慎作答,并明确告知用户本地知识库暂时不可用、"
                    "答案未经内部资料核实。",
    },
    # 记忆类独立隔离:故障只摘 recall_memory,检索三件套与主流程不受影响。
    Category.MEMORY: {
        "label": "记忆召回",
        "fallback": "不再调用 recall_memory;按当前对话与通用知识作答即可,"
                    "不要假设该用户的历史偏好。",
    },
}


def _unavailable_tools(tool_status: dict[str, str]) -> tuple[set[str], list[str]]:
    """汇总当前不可用的工具。

    合并两个信号:
      - 本轮内已故障的类别(tool_status[category] == "down",粘性);
      - 跨轮持久熔断中(breaker state == "open";half_open 仍可试探,不摘)。
    返回 (不可用工具名集合, 故障类别列表)。熔断器关闭/特性关闭时返回空。
    """
    down_cats = {cat for cat, st in (tool_status or {}).items() if st == "down"}
    open_tools: set[str] = set()
    try:
        if C.TOOL_HEALTH_ADAPT_ENABLED:
            for tname, tstate in circuit_snapshot().items():
                if tstate == "open":
                    open_tools.add(tname)
    except Exception:
        logger.warning("读取熔断快照异常,忽略健康度自适应", exc_info=True)
        open_tools = set()

    unavailable: set[str] = set(open_tools)
    for spec in registry.all():
        if spec.category in down_cats:
            unavailable.add(spec.name)
    # 熔断打开工具的类别:用 registry.get 取 spec(熔断工具可能当前 disabled,
    # 不在 registry.all() 的 enabled 集合里,但类别仍需提示)。
    open_cats = set()
    for tname in open_tools:
        spec = registry.get(tname)
        if spec is not None:
            open_cats.add(spec.category)
    return unavailable, sorted(down_cats | open_cats)


def _tool_health_block(unavailable: set[str], down_cats: list[str]) -> str:
    """构造注入 system message 的工具健康度自适应提示块。无故障时返回空串。"""
    if not down_cats:
        return ""
    lines = ["【工具健康度提示】以下工具/服务当前不可用,严禁再调用它们:"]
    for cat in down_cats:
        info = _CATEGORY_HEALTH.get(cat)
        if info is None:
            continue
        names = [s.name for s in registry.by_category(cat)
                 if s.name in unavailable] or [s.name for s in registry.by_category(cat)]
        lines.append(f"- {info['label']}(工具: {', '.join(names)})不可用。{info['fallback']}")
    lines.append("请直接采用上述替代策略继续,不要重复尝试已不可用的工具。")
    return "\n".join(lines)


def _recorder(config) -> TraceRecorder:
    return config["configurable"]["trace_recorder"]


def _hard_deadline(state) -> float | None:
    """端到端硬截止墙钟时间戳;未设置(如测试直连图)返回 None。

    由 service 层按整条请求(含升级/重做)计算后经 state 传入,使旁路 LLM/工具
    等待都受同一条端到端预算约束,而不是各花各的 tier 预算叠加成数分钟。
    """
    dl = state.get("hard_deadline")
    return float(dl) if dl else None


def _remaining_budget(state) -> Optional[float]:
    """距硬截止还剩多少秒;无硬截止时回退到 tier 相对预算(started_at+max_total)。

    返回值可能 <= 0(已超时),调用方应据此短路;旁路等待用
    ``min(自身超时, max(剩余, 下限))`` 钳制,保证绝不越过端到端硬预算。
    """
    max_total = int(state.get("max_total_seconds") or MAX_TOTAL_SECONDS)
    started = float(state.get("started_at") or time.time())
    dl = _hard_deadline(state)
    if dl is None:
        return max_total - (time.time() - started)
    return min(dl, started + max_total) - time.time()


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

    State 经 checkpoint 跨轮持久化,full_reply / collected_sources / error 等
    运行期字段若不重置,会把上一轮的回答、来源、错误带入下一轮(回答跨轮串联)。
    """
    w = get_stream_writer()
    trace_id = state.get("trace_id") or str(uuid.uuid4())[:8]
    w({"type": "status", "message": "理解问题中…", "trace_id": trace_id})
    return {
        "trace_id": trace_id,
        "step": 0,
        # 哨兵键通知 _merge_tool_status reducer 复位为全部 "up"(见 state.py)
        "tool_status": {"__reset__": True, **_DEFAULT_TOOL_STATUS},
        "full_reply": "",
        # 哨兵键:通知 _merge_sources reducer 清空跨轮残留(见 state.py)
        "collected_sources": {"__reset__": True},
        "tool_parse_errors": {},
        "pending_tool_calls": [],
        "tool_outcomes": [],
        "tool_fail_streak": {},
        "tool_requery_count": {},
        "error": None,
        "final_reason": None,
        "search_count": 0,
        "retrieval_max_score": 0.0,
    }


def build_messages_node(state: AgentState, config) -> dict:
    """构建 LLM messages(system+history+user),并注入对话摘要。

    断点续跑/多轮:system 用固定 id(按 id 更新而非重复追加);已有 messages 时
    只追加本轮新 user 问题,不重复灌前端 history。

    跨轮旧消息的压缩/摘要【不在本节点】:由后台记忆管道在上一轮结束后完成
    (RemoveMessage 经 update_state 删旧轮 + summary 落盘),故本轮所见 existing 已是
    压缩后结果,这里只需按固定 id 更新 system prompt(注入 summary 块)。

    记忆注入(Req1):本节点做一次【确定性、无 LLM】的预取——长期高置信偏好(带条目 id)
    + 游标后的近期对话原文,并入 system 摘要块(失败软降级,不阻断)。模型随后仍可显式
    调用 recall_memory 工具做扩量/翻页(结果按预取 id 去重)。
    """
    question = state["question"]
    existing = state.get("messages") or []
    # 冷启动 = checkpointer 无既有 messages(首轮 / 重启恢复 / 换设备)。此轮多轮上下文
    # 以服务端 Redis 短期流水为权威来源铺成结构化消息(见下方 else 分支),不再依赖前端
    # 重发 history;相应地让预取跳过②近期文本块,避免同段历史重复注入。
    cold_start = not existing
    SYSTEM_MSG_ID = "system-prompt"

    summary = state.get("summary") or ""

    # 确定性预取(无 LLM):长期偏好 + 游标后近期对话。失败软降级为空块。
    cfg = (config or {}).get("configurable", {}) or {}
    username = cfg.get("user_id") or state.get("user_id")
    thread_id = cfg.get("thread_id")
    prefetch_block = ""
    prefetch_ids: set = set()
    try:
        from memories.orchestration.long.prefetch import build_prefetch_block
        pf = build_prefetch_block(username, thread_id, question,
                                  include_recent=not cold_start)
        prefetch_block = pf.get("block") or ""
        prefetch_ids = pf.get("mem_ids") or set()
    except Exception as e:  # noqa: BLE001  预取全程旁路
        logger.info("prefetch skipped: %s: %s", type(e).__name__, str(e)[:120])
    # 预取条目 id 经 contextvar 传给 recall_memory 工具做长期条目去重(同一条目不重复注入)。
    # 近期对话只由本预取通道注入,工具不再拉短期。
    try:
        from tools.memory_tool import set_prefetched_ids
        set_prefetched_ids(prefetch_ids)
    except Exception:  # noqa: BLE001
        pass

    def _with_blocks(base_sys: str) -> str:
        for blk in (format_summary_block(summary), prefetch_block):
            if blk:
                base_sys += "\n\n" + blk
        return base_sys

    patch: dict[str, Any] = {}
    qc_feedback = (state.get("qc_feedback") or "").strip()

    if existing:
        # 跨轮:旧消息已由 memory-loop 压缩落盘;仅更新 system(注入 summary+预取块)+ 追加新问题
        raw_sys = build_messages(question, [])[0]
        sys_content = _with_blocks(raw_sys["content"])
        patch_msgs: list[BaseMessage] = [
            SystemMessage(content=sys_content, id=SYSTEM_MSG_ID)]
        last = existing[-1]
        already = isinstance(last, HumanMessage) and last.content == question
        if not already:
            patch_msgs.append(HumanMessage(content=question))
        if qc_feedback:
            # 质检/升级重做轮:显式注入系统质检反馈(user 槽位,标注非用户发言)
            patch_msgs.append(HumanMessage(content=qc_feedback))
        patch["messages"] = patch_msgs
        return patch

    # 冷启动种子多轮:以服务端 Redis 短期流水为权威来源(不再用前端重发的 history)。
    # 仅当 Redis 不可用/无流水(返回 [])时,才紧急退回前端 history 兜底,保证不丢上下文。
    seed_turns: list[dict] = []
    try:
        from memories.orchestration.short.recall import recent_dialogue_messages
        seed_turns = recent_dialogue_messages(thread_id, question, limit=10)
    except Exception as e:  # noqa: BLE001  旁路:取短期流水失败不阻断
        logger.info("seed recent_dialogue skipped: %s: %s",
                    type(e).__name__, str(e)[:120])
    if not seed_turns:
        seed_turns = [
            {"role": h.get("role"), "content": h.get("content")}
            for h in (state.get("history") or [])
            if h.get("role") in ("user", "assistant") and h.get("content")
        ][-10:]

    # system(注入 summary+预取块) + Redis 种子多轮 + 本轮问题(+质检反馈)
    sys_content = _with_blocks(build_messages(question, [])[0]["content"])
    msgs: list[BaseMessage] = [SystemMessage(content=sys_content, id=SYSTEM_MSG_ID)]
    for t in seed_turns:
        if t["role"] == "assistant":
            msgs.append(AIMessage(content=t["content"]))
        else:
            msgs.append(HumanMessage(content=t["content"]))
    msgs.append(HumanMessage(content=question))
    # 质检/升级反馈:history 夹带渠道在"种子以 Redis 流水为权威"时会被忽略,
    # 故显式注入(若流水兜底渠道已带同文则不重复)。
    if qc_feedback and not (seed_turns and seed_turns[-1].get("content") == qc_feedback):
        msgs.append(HumanMessage(content=qc_feedback))
    return {"messages": msgs}


def agent_node(state: AgentState, config) -> dict:
    """核心 LLM 调用节点:流式读 token、累积 tool_calls、判断终止/继续。"""
    w = get_stream_writer()
    recorder = _recorder(config)
    trace_id = state["trace_id"]
    t0 = state["started_at"]

    prev_step = int(state.get("step", 0))
    max_steps = int(state.get("max_steps") or MAX_STEPS)

    # 已完整执行完 max_steps 轮且仍需继续 -> 停止(复刻 range(1,MAX_STEPS+1) 语义)
    if prev_step >= max_steps:
        w({"type": "status", "message": "已达最大推理步数,输出当前结果。",
           "trace_id": trace_id, "step": max_steps})
        return {"step": max_steps, "final_reason": "max_steps"}

    step = prev_step + 1
    # 端到端硬预算:既看 tier 相对预算(started_at+max_total),也看 service 下发的
    # 硬截止(跨升级/重做共享)。任一耗尽即停止推理、输出当前结果。
    remaining = _remaining_budget(state)
    if remaining is not None and remaining <= 0:
        elapsed = time.time() - t0
        w({"type": "status",
           "message": f"响应超时({int(elapsed)}s),输出当前结果。",
           "trace_id": trace_id, "step": step})
        return {"step": step, "final_reason": "timeout"}

    step_doc = recorder.new_step(step)
    w({"type": "step_start", "trace_id": trace_id, "step": step,
       "elapsed_ms": int((time.time() - t0) * 1000)})
    w({"type": "status", "message": "思考中…", "trace_id": trace_id, "step": step})

    # ---- 工具健康度自适应 ----
    # 合并"本轮故障类别(tool_status)"与"跨轮持久熔断(breaker open)":
    # ① 把不可用工具从本轮 schema 摘掉(模型看不到、无法再碰壁调用);
    # ② 注入 system 提示告知故障与替代策略(检索挂了基于通用知识谨慎作答,或如实告知用户)。
    unavailable, down_cats = _unavailable_tools(state.get("tool_status") or {})
    health_block = _tool_health_block(unavailable, down_cats)
    if down_cats:
        logger.info("工具故障自适应: 不可用工具=%s 故障类别=%s",
                    sorted(unavailable), down_cats)
        w({"type": "status",
           "message": "部分工具暂时不可用,已切换备用策略…",
           "trace_id": trace_id, "step": step})

    # ---- LLM 调用 ----
    # bind_tools=False(simple 直答路径)时不传 tools schema,模型只生成文本、不会发 tool_calls。
    bind_tools = bool(state.get("bind_tools", True))
    openai_messages = _msgs_to_openai(state["messages"])

    # ---- 投机检索预注入(仅首轮) ----
    # 服务层在请求进入时已后台执行 search_text(原始问题)。react 首轮把它作为
    # 一次"已完成的检索"注入上下文:模型可直接引用作答(省一步检索+解码),
    # 也可继续补检。来源同步并入 collected_sources,保证 sources 事件照常发出。
    pre_patch: dict[str, Any] = {}
    pre_search = state.get("pre_search") if step == 1 else None
    if bind_tools and pre_search and getattr(C, "REACT_PRE_SEARCH", True):
        spec_pre = registry.get("search_text")
        pre_srcs = {}
        for _s in sources_from_result(pre_search, spec=spec_pre):
            key = (_s.get("chunk_id") or _s.get("url")
                   or (_s.get("source_stem", "") + _s.get("page", "")))
            if key:
                pre_srcs[key] = _s
        if pre_srcs:
            t_pre = time.time()
            pre_doc = recorder.new_step(0)
            recorder.record_tool(pre_doc, tool_call_id="pre-search-0",
                                 name="search_text", args={"query": state.get("question", "")},
                                 duration_ms=int((time.time() - t_pre) * 1000),
                                 ok=True, category="retrieval")
            recorder.finish_step(pre_doc, "tool_calls", new_sources_count=len(pre_srcs))
            preview = json.dumps(pre_search, ensure_ascii=False, default=str)
            w({"type": "tool_result", "trace_id": trace_id, "step": 0,
               "tool_call_id": "pre-search-0", "name": "search_text", "ok": True,
               "duration_ms": 0,
               "result_size": len(preview),
               "result_preview": preview[:RESULT_PREVIEW_LEN]
                   + ("…" if len(preview) > RESULT_PREVIEW_LEN else ""),
               "error": None, "error_type": None})
            w({"type": "sources", "items": list(pre_srcs.values())[:6],
               "trace_id": trace_id, "step": 0})
            pre_patch = {
                "collected_sources": pre_srcs,
                "search_count": 1,
                "retrieval_max_score": max(
                    (float(s.get("score") or 0.0) for s in pre_srcs.values()),
                    default=0.0),
            }
            pre_content = truncate_tool_result(
                pre_search, C.CONTEXT_TOOL_RESULT_MAX_CHARS, spec=spec_pre)
            openai_messages = openai_messages + [
                {"role": "assistant",
                 "content": "",
                 "tool_calls": [{"id": "pre-search-0", "type": "function",
                                 "function": {"name": "search_text",
                                              "arguments": json.dumps(
                                                  {"query": state.get("question", "")},
                                                  ensure_ascii=False)}}]},
                {"role": "tool", "tool_call_id": "pre-search-0",
                 "content": ("[系统预检索结果 —— 已按用户原始问题自动检索,可直接引用,"
                            "如不足再自行检索]\n" + pre_content)},
            ]

    if health_block:
        # 紧邻本次调用追加 system 提示,优先级高于早先 system,确保模型读到最新健康度。
        openai_messages = openai_messages + [{"role": "system", "content": health_block}]

    # 最后一轮(已达最大推理步数):强制不带工具,要求模型基于已检索资料直接给出完整最终
    # 答案。否则模型可能把全部轮次耗在工具调用上,触顶 final_reason=max_steps 时
    # full_reply 只剩中间"我来检索…我继续…"叙述、没有可交付的答案(实测 TMA 题偶发)。
    if step >= max_steps and bind_tools:
        bind_tools = False
        openai_messages = openai_messages + [{
            "role": "system",
            "content": ("【系统提示】已达到最大检索步数,请不要再调用任何工具。"
                        "请立即基于上面已检索到的资料,直接给出简洁、带引用的最终答案;"
                        "不要叙述检索过程,也不要说“我继续/我再查/接下来”之类的话。"
                        "【忠实性红线】只陈述资料能直接支撑的内容:不得推测故障原因、"
                        "不得虚构排查步骤或引用资料,不得附加资料外的安全提示或"
                        "“联系技术支持”类建议;资料未覆盖的部分明确说明一句即可。"),
        }]
        w({"type": "status",
           "message": "已达最大推理步数,正在整合已检索资料给出最终答案…",
           "trace_id": trace_id, "step": step})
    llm_kwargs: dict[str, Any] = dict(
        model=getattr(C, "TIER_MODEL_REACT", None) or C.OPENAI_TEXT_MODEL,  # tier 模型优先(网关别名 main;直连模式回退)
        messages=openai_messages,
        stream=True, stream_options={"include_usage": True},
        temperature=0.3, timeout=STREAM_TIMEOUT,
    )
    if not bind_tools:
        # 终答调用(含末步强制摘工具)限制解码长度,防跑飞的长答案拖垮整体延迟;
        # 工具调用步不设限——截断会破坏流式 tool-call JSON 的完整性。
        llm_kwargs["max_tokens"] = int(getattr(C, "REACT_ANSWER_MAX_TOKENS", 600))
        # 终答关思考省解码;忠实性由 grounding 后置校验兜底(见 support/grounding.py)
        # —— 实测 run17 开思考自检对忠实度无增益(0.8737→0.8818,噪声级)。
        llm_kwargs.update(no_think_extra())
    if bind_tools:
        # recall_memory 已注册进 registry(与检索三件套同源,检索三件套经 MCP 桥注册);
        # 匿名用户/记忆关闭时不下发(记忆按用户隔离),故障工具由 unavailable 摘除。
        username = (config.get("configurable", {}).get("user_id")
                    or state.get("user_id"))
        hidden = set(unavailable)
        if not (username and getattr(C, "LONG_MEM_ENABLED", True)):
            hidden.add(MEMORY_TOOL_NAME)
        available_schemas = [
            sch for sch in registry.schemas()
            if sch.get("function", {}).get("name") not in hidden]
        # 全部工具都不可用时不传 tools(空列表会被 API 拒绝),让模型纯文本作答并告知用户。
        if available_schemas:
            llm_kwargs["tools"] = available_schemas
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

    # 总时长看门狗:思考模型(GLM)流式 reasoning 持续到达会绕过 STREAM_TIMEOUT
    # 空闲超时,单步思考实测可拖 5 分钟+。到点强制断流,按"思考超时截断"处理:
    # 有部分正文 → 当正常流末走;空 → 后续按空回复自然降级(不得重试)。
    _wd_deadline = float(getattr(C, "REACT_STREAM_DEADLINE_S", 45))
    _wd_cancel, _wd_killed = arm_stream_watchdog(stream, _wd_deadline)

    # ---- 读流 ----
    # Req4 流式缓冲:仅当本轮【确实绑定了工具】(模型可能先吐字再改口调工具)时,把 content
    # 暂存不即时下发;流末确认无 tool_calls(终答落定)才回放放流,出现 tool_calls 则丢弃
    # 缓冲正文。未绑工具(simple/末步强制摘工具)不会改口,保持即时流式。
    stream_buffer = (bool(getattr(C, "REACT_STREAM_BUFFER_ENABLED", True))
                     and bool(llm_kwargs.get("tools")))
    # grounding 后置校验:终答必须整段到手才能逐句校验,强制缓冲(含末步强制摘工具)
    if not stream_buffer and getattr(C, "GROUNDING_CHECK", False) and not bind_tools:
        stream_buffer = True
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
                    if not stream_buffer:
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
        if not _wd_killed.is_set():
            err_doc = recorder.record_error(step, "llm_stream", e)
            w({"type": "error_trace", "trace_id": trace_id, **err_doc})
            w({"type": "error", "message": f"流读取中断: {str(e)[:160]}",
               "trace_id": trace_id, "step": step})
            recorder.finish_step(step_doc, "error")
            recorder.final_reason = "error"
            return {"step": step, "final_reason": "error", "full_reply": full_reply,
                    "error": {"phase": "llm_stream", "message": str(e)[:300]}}
        # 看门狗截断:落到底部统一处理(部分正文按正常流末走)
    finally:
        _wd_cancel()

    stream_duration_ms = int((time.time() - t_llm) * 1000)

    # 看门狗触发但流是以"正常结束"形式落地的(close 不总抛异常):同样丢弃
    # 半截 tool-call,并给出截断提示。空正文场景由下方救援兜底。
    if _wd_killed.is_set():
        if tc_acc:
            tc_acc.clear()
        w({"type": "status",
           "message": "模型思考超时,已按当前进度截断处理…",
           "trace_id": trace_id, "step": step})

    # 终答救援:GLM 对抽象问题的思考可烧光全部 max_tokens(finish=length,
    # 正文 0 token,流正常结束),或被看门狗截断到零正文。轻模型(doubao,
    # thinking 已由 llm 层策略关闭)非流式直答一次,2-3s 必有产出;
    # 走 GLM 重试只会再烧光一次(实测)。
    if not bind_tools and not content_buf.strip() and not tc_acc \
            and openai_messages:
        w({"type": "status", "message": "正在重试生成答案…",
           "trace_id": trace_id, "step": step})
        try:
            _resp, _rerr = llm_create_with_retry(
                client, trace_id=trace_id, retries=1,
                model=str(getattr(C, "MODEL_LIGHT", "")
                          or "doubao-seed-2.0-lite"),
                messages=openai_messages,
                stream=False, temperature=0.3,
                max_tokens=int(getattr(C, "REACT_ANSWER_MAX_TOKENS", 2000)),
            )
            if _rerr is None:
                content_buf = (_resp.choices[0].message.content or "").strip()
                if content_buf:
                    full_reply += content_buf
        except Exception:  # noqa: BLE001  救援失败按空答案走下游降级
            content_buf = ""

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

    # Req4:缓冲回放——流末已落定。无 tool_calls(终答)→ 把本步缓冲正文以 token 事件一次性
    # 回放放流;有 tool_calls(还要检索)→ 丢弃缓冲,正文不下发(避免"先上屏再改口")。
    if stream_buffer and content_buf:
        if not lc_tool_calls:
            w({"type": "status", "message": "正在整理答案…",
               "trace_id": trace_id, "step": step})
            # 裸答熔断:全程没有任何检索来源时,模型自答零锚定(检索宕机/全空 miss),
            # 不放行——整段替换为人工翻阅手册引导,并跳过 grounding(引导文本无需校验)。
            _n_src = len(state.get("collected_sources") or {}) + \
                len(((pre_patch or {}).get("collected_sources") or {}))
            _naked_fused = False
            if _n_src == 0 and getattr(C, "REACT_NAKED_ANSWER_FUSE", True):
                # 上下文豁免:跨轮 checkpoint/摘要、recall_memory 工具、记忆指令型
                # 问题("请记住…")——凭会话上下文或确认指令作答是正当的,不走熔断。
                # 熔断只针对"全新单轮问题 + 全程零检索来源"的参数化自答(run22 场景)。
                _humans, _mem_tool = 0, False
                for _m in (state.get("messages") or []):
                    _mt = type(_m).__name__
                    if _mt == "HumanMessage":
                        _humans += 1
                    elif _mt == "ToolMessage" and \
                            getattr(_m, "name", "") == MEMORY_TOOL_NAME:
                        _mem_tool = True
                _ctx_exempt = (
                    bool(state.get("summary")) or _humans >= 2 or _mem_tool
                    or bool(_MEMORY_INSTR_RE.search(str(state.get("question") or ""))))
                if not _ctx_exempt:
                    from ..support.grounding import _guidance_text
                    w({"type": "status",
                       "message": "未检索到任何可用资料,已转为人工核查指引",
                       "trace_id": trace_id, "step": step})
                    content_buf = _guidance_text([])
                    _naked_fused = True
            elif (step < max_steps
                  and getattr(C, "REACT_NARRATION_SALVAGE", True)
                  and _NARRATION_RE.search(content_buf or "")):
                # 过程性叙述误当终答:不带工具重问一次(非流式,复用 Req4 一次性回放),
                # 仍叙述则接受原文(有 max_steps 强制答兜底,不会无限重问)。
                w({"type": "status",
                   "message": "检测到过程性叙述,正在重新生成最终答案…",
                   "trace_id": trace_id, "step": step})
                try:
                    _client = get_client()
                    _r2 = _client.chat.completions.create(
                        model=getattr(C, "TIER_MODEL_REACT", None) or C.OPENAI_TEXT_MODEL,
                        messages=openai_messages + [
                            {"role": "assistant", "content": content_buf},
                            {"role": "system",
                             "content": ("上一条回复是检索过程叙述,不是最终答案。"
                                         "请不要再叙述过程,立即基于上面已检索到的资料"
                                         "给出简洁、带引用的最终答案;资料未覆盖的部分"
                                         "明确说明一句即可。")},
                        ],
                        stream=False, temperature=0.3,
                        max_tokens=int(getattr(C, "REACT_ANSWER_MAX_TOKENS", 600)),
                        timeout=STREAM_TIMEOUT,
                    )
                    _txt2 = ((_r2.choices[0].message.content
                              if _r2.choices else "") or "").strip()
                    if _txt2 and not _NARRATION_RE.search(_txt2):
                        if full_reply.endswith(content_buf):
                            full_reply = full_reply[:-len(content_buf)] + _txt2
                        else:
                            full_reply += _txt2
                        content_buf = _txt2
                except Exception as e:  # 重问失败静默,保留原文走 grounding
                    logger.info("narration salvage retry failed: %s", str(e)[:120])
            # grounding 后置校验:逐句判"能否被检索资料蕴含",删无支撑句再下发。
            # 校验调用套熔断器:CLOSED 走超时+重试,连续失败→OPEN 降级放行原文,
            # 冷却结束 HALF_OPEN 试探恢复(见 support/grounding.py)。
            if not _naked_fused and getattr(C, "GROUNDING_CHECK", False):
                from ..support.grounding import grounding_filter
                filtered, ginfo = grounding_filter(
                    content_buf,
                    list((state.get("collected_sources") or {}).values()),
                    trace_id=trace_id)
                if ginfo.get("error") == "LLM调用失败":
                    w({"type": "status",
                       "message": "grounding 校验暂不可用(LLM调用失败),已降级放行原文",
                       "trace_id": trace_id, "step": step})
                elif ginfo.get("action") == "guidance":
                    # 置信度不达标:模型答案不下发,替换为人工翻阅手册引导
                    w({"type": "status",
                       "message": f"答案置信度不足({ginfo.get('confidence')}),已替换为人工核查指引",
                       "trace_id": trace_id, "step": step})
                elif ginfo.get("removed"):
                    w({"type": "status",
                       "message": f"已按检索资料过滤 {ginfo['removed']} 句无支撑内容",
                       "trace_id": trace_id, "step": step})
                content_buf = filtered
            w({"type": "token", "delta": content_buf,
               "trace_id": trace_id, "step": step})
        else:
            logger.info("stream buffer discarded %d chars (tool_calls present)",
                        len(content_buf))

    ai_msg = (AIMessage(content=content_buf, tool_calls=lc_tool_calls)
              if lc_tool_calls else AIMessage(content=content_buf))
    patch: dict[str, Any] = {
        "step": step,
        "messages": [ai_msg],
        "full_reply": full_reply,
    }
    if pre_patch:
        # 预检索来源/计数并入本步状态(sources 事件已在注入时发出)
        patch.update(pre_patch)
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


def _is_empty_result(result) -> bool:
    """判断工具结果是否为空(供 reflect 的空结果换词决策)。"""
    if result is None:
        return True
    if isinstance(result, (list, tuple, dict, str)):
        return len(result) == 0
    return False


def execute_tools_node(state: AgentState, config) -> dict:
    """执行经 validate_generation/validate_runtime 校验放行的 pending_tool_calls。

    本节点只做【编排】:发 tool_call 事件 → 全局 daemon 线程池 fan-out(每个调用经
    韧性中间件 call_with_resilience,透明处理超时/熔断/限流/抖动/崩溃)→ 端到端硬预算
    有界等待(单/多调用统一走 future.result,超时 budget_timeout 占位)→ 回主线程合并
    来源、记 metrics/trace、构造 ToolMessage(成功截断 / 失败回灌纠错文案)、产出
    tool_outcomes 供 reflect 决策。

    「机械重试」在中间件;「是否摘工具/降级/换词」由 reflect 节点研判,本节点不做决策。
    """
    w = get_stream_writer()
    recorder = _recorder(config)
    trace_id = state["trace_id"]
    step = state["step"]
    step_doc = recorder.steps[-1] if recorder.steps else recorder.new_step(step)

    valid = state.get("pending_tool_calls") or []
    new_sources: dict[str, dict] = {}
    tool_msgs: list[BaseMessage] = []
    outcomes: list[dict] = []

    # 主线程先发 tool_call / "调用中"事件(仅对真正要执行的合法调用)。
    for call in valid:
        name = call["name"]
        args = call.get("args") or {}
        tcid = call["id"]
        w({"type": "tool_call", "trace_id": trace_id, "step": step,
           "tool_call_id": tcid, "name": name, "args": args,
           "args_preview": _args_preview(args)})
        w({"type": "status",
           "message": f"调用工具 {name}({_args_preview(args)})…",
           "trace_id": trace_id, "step": step})

    hard_deadline = _hard_deadline(state)

    # 记忆工具的身份/预取 id 经 ContextVar 传给 handler;worker 线程不继承主线程
    # contextvar,故在主线程取值、于每个 worker 内显式设置(Req2 去重 + 身份隔离)。
    mem_username = (config.get("configurable", {}).get("user_id")
                    or state.get("user_id"))
    mem_thread_id = config.get("configurable", {}).get("thread_id")
    mem_prefetch_ids: set = set()
    try:
        from tools.memory_tool import (
            set_memory_ctx, set_prefetched_ids, get_prefetched_ids)
        # 主线程(build_messages_node 预取时设置)读取预取 id,再下发到 worker 线程。
        mem_prefetch_ids = get_prefetched_ids()
    except Exception:  # noqa: BLE001
        set_memory_ctx = set_prefetched_ids = None  # type: ignore

    def _execute(call: dict) -> dict:
        """worker 内执行单个调用(经韧性中间件)。不触碰 writer/recorder/metrics。"""
        name = call["name"]
        args = call.get("args") or {}
        tcid = call["id"]
        t_tool = time.time()
        circuits: list[dict] = []
        spec = registry.get(name)

        # 记忆工具:在本 worker 线程注入身份与预取 id(handler 经 contextvar 读取)。
        if name == MEMORY_TOOL_NAME and set_memory_ctx is not None:
            set_memory_ctx(mem_username, mem_thread_id)
            set_prefetched_ids(mem_prefetch_ids)

        def _on_event(ev):
            if ev.get("type") == "circuit":
                circuits.append(ev)

        cache_hit_var.set(False)  # 复位;handler 命中缓存时置 True(同线程可见)
        try:
            result, err = call_with_resilience(
                name, args, spec, deadline=hard_deadline,
                on_event=_on_event, invoke=dispatch)
        except Exception as e:  # 兜底:中间件理论上不抛,防止 worker 异常扩散
            result, err = None, ToolCallError(
                Stage.EXECUTION, Kind.CRASH, f"{type(e).__name__}: {e}", tool=name)
        cache_hit = bool(cache_hit_var.get())
        return {"call": call, "name": name, "args": args, "tcid": tcid,
                "spec": spec, "result": result, "err": err,
                "cache_hit": cache_hit, "circuits": circuits,
                "duration_ms": int((time.time() - t_tool) * 1000)}

    def _timeout_record(call: dict) -> dict:
        """硬预算耗尽仍未返回 -> budget_timeout 占位(保序,回灌 LLM)。"""
        return {"call": call, "name": call["name"], "args": call.get("args") or {},
                "tcid": call["id"], "spec": registry.get(call["name"]),
                "result": None,
                "err": ToolCallError(Stage.EXECUTION, Kind.BUDGET_TIMEOUT,
                                     "等待超过端到端时限,已跳过", tool=call["name"]),
                "cache_hit": False, "circuits": [], "duration_ms": 0}

    # 单个调用 / TOOL_MAX_PARALLEL<=1:同步串行执行(零线程开销,兼容串行测试);
    # 硬预算由韧性中间件内部按 deadline 钳制重试。多个调用且允许并发:走全局 daemon 池
    # fan-out + future.result(剩余硬预算)有界等待;超时的以 budget_timeout 占位,
    # 运行中的任务不 join(daemon 池,结果丢弃)。
    parallel = getattr(C, "TOOL_MAX_PARALLEL", 4)
    executed: list[dict] = []
    if not valid:
        executed = []
    elif len(valid) <= 1 or parallel <= 1:
        executed = [_execute(call) for call in valid]
    else:
        pool = get_pool()
        futures = {pool.submit(_execute, call): call for call in valid}
        for call in valid:  # 按 pending 原序对齐结果
            fut = next(f for f, t in futures.items() if t is call)
            wait_s = None
            rem = _remaining_budget(state)
            if rem is not None:
                wait_s = max(0.0, rem)
            try:
                executed.append(fut.result(timeout=wait_s))
            except FuturesTimeout:
                w({"type": "status",
                   "message": f"⚠️ 工具 {call['name']} 等待超过端到端时限,已跳过该调用",
                   "trace_id": trace_id, "step": step})
                logger.warning("tool %s 等待超过硬预算,跳过", call["name"])
                executed.append(_timeout_record(call))
                fut.cancel()  # 排队中的可取消;运行中的允许跑完(结果丢弃),不 join

    # 结果回主线程:发 circuit/tool_result、记 metrics/trace、合并来源、构造 ToolMessage。
    for r in executed:
        name = r["name"]
        args = r["args"]
        tcid = r["tcid"]
        spec = r["spec"]
        err = r["err"]
        result = r["result"]
        duration_ms = r["duration_ms"]
        cache_hit = r["cache_hit"]
        produces_sources = bool(spec and spec.produces_sources)

        for ev in r["circuits"]:
            w({"type": "circuit", "trace_id": trace_id, "step": step,
               "name": ev.get("name"), "state": ev.get("state")})

        tool_ok = err is None
        kind = err.kind if err else None
        is_empty = tool_ok and _is_empty_result(result)
        metrics.record_tool_call(
            name, success=tool_ok, category=spec.category if spec else None,
            duration_ms=duration_ms,
            error_type=kind or ("empty" if is_empty else None),
            cache_hit=cache_hit)
        recorder.record_tool(
            step_doc, tool_call_id=tcid, name=name, args=args,
            ok=tool_ok, duration_ms=duration_ms,
            result=result if tool_ok else (err.to_dict() if err else result),
            error=err.message if err else None,
            category=spec.category if spec else None,
            error_type=kind, cache_hit=cache_hit)

        if tool_ok:
            result_size = len(json.dumps(result, ensure_ascii=False, default=str))
            preview = json.dumps(result, ensure_ascii=False, default=str)
        else:
            result_size = len(err.message)
            preview = err.message
        w({"type": "tool_result", "trace_id": trace_id, "step": step,
           "tool_call_id": tcid, "name": name, "ok": tool_ok,
           "duration_ms": duration_ms, "result_size": result_size,
           "result_preview": preview[:RESULT_PREVIEW_LEN]
               + ("…" if result_size > RESULT_PREVIEW_LEN else ""),
           "error": err.message if err else None, "error_type": kind})

        if tool_ok and produces_sources:
            for _s in sources_from_result(result, spec=spec):
                # 来源主键:web 用 url,doc 用 chunk_id/source_stem+page
                key = (_s.get("chunk_id")
                       or _s.get("url")
                       or (_s.get("source_stem", "") + _s.get("page", "")))
                if not key:
                    continue
                if key not in new_sources or _s.get("score", 0) > new_sources[key].get("score", 0):
                    new_sources[key] = _s

        # 回灌 LLM 的 ToolMessage:成功截断结果;失败用面向模型的纠错文案。
        if tool_ok:
            tool_msgs.append(ToolMessage(
                content=truncate_tool_result(
                    result, C.CONTEXT_TOOL_RESULT_MAX_CHARS, spec=spec),
                tool_call_id=tcid))
        else:
            tool_msgs.append(error_tool_message(tcid, err))

        outcomes.append({
            "name": name, "tcid": tcid,
            "category": spec.category if spec else None,
            "ok": tool_ok, "kind": kind, "empty": is_empty,
            "produces_sources": produces_sources,
        })

    new_count = len(new_sources)
    # 本轮真正执行成功的检索类调用次数(get_chunk 不计入新检索)。
    searched = sum(1 for oc in outcomes if oc["produces_sources"] and oc["ok"])
    search_count = int(state.get("search_count") or 0) + searched
    recorder.finish_step(step_doc, "tool_calls", new_sources_count=new_count)
    w({"type": "step_end", "trace_id": trace_id, "step": step,
       "decision": "tool_calls", "new_sources_count": new_count,
       "elapsed_ms": step_doc.get("elapsed_ms")})

    collected = state.get("collected_sources") or {}
    total_sources = collected | new_sources
    max_score = max(
        (float(s.get("score") or 0.0) for s in total_sources.values()),
        default=0.0)

    if total_sources:
        metrics.record_search(hit=True)
        w({"type": "sources", "items": list(total_sources.values())[:6],
           "trace_id": trace_id, "step": step})
    else:
        metrics.record_search(hit=False)

    return {
        "messages": tool_msgs,
        "collected_sources": new_sources,
        "search_count": search_count,
        "retrieval_max_score": max_score,
        "tool_outcomes": outcomes,
    }


def finalize_node(state: AgentState, config) -> dict:
    """收尾:meta + assistant_message + 最终来源卡片。

    【不发 done】:done 由 emit_done_node 统一发射;记忆维护已迁出主图
    (后台记忆管道在流结束后处理),本节点不再承担任何记忆职责。
    """
    w = get_stream_writer()
    recorder = _recorder(config)
    trace_id = state["trace_id"]
    t0 = state["started_at"]
    step = state["step"]
    decision = state.get("final_reason") or "answer"
    full_reply = state.get("full_reply", "")
    collected_sources = state.get("collected_sources") or {}

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
    ))

    # Req5 终态分支:按 final_reason 决定发什么。
    #   answer            :正常答案卡 + 最终引用;
    #   max_steps/timeout :答案不完整,发 assistant_message 带 incomplete:true(含已完成部分);
    #   error             :只发 error 事件,不发答案卡/来源(错误信息已由 agent 节点发出)。
    is_error = decision == "error"
    is_incomplete = decision in ("max_steps", "timeout")
    if not is_error and full_reply.strip():
        msg: dict[str, Any] = {"type": "assistant_message",
                               "trace_id": trace_id, "content": full_reply}
        if is_incomplete:
            msg["incomplete"] = True
            msg["incomplete_reason"] = decision
            msg["note"] = ("已达推理步数/时间上限,以下为基于已检索资料的部分结果,"
                           "可能不完整。")
        w(msg)
        # 定稿后补发"按书去重+引用优先"的最终来源卡片,与答案实际引用对齐。
        try:
            cards = final_citation_cards(
                full_reply, list(collected_sources.values()), k=6)
            if cards:
                w({"type": "sources", "trace_id": trace_id, "step": step, "items": cards})
        except Exception:
            pass  # 卡片重排失败不影响收尾
    patch: dict[str, Any] = {
        "final_reason": decision,
        "full_reply": full_reply,
        "retrieval_max_score": float(state.get("retrieval_max_score") or 0.0),
    }
    if state.get("error"):
        patch["error"] = state["error"]
    return patch


def emit_done_node(state: AgentState, config) -> dict:
    """图流最后一帧:统一发射 done(finalize 之后、图即完即关)。

    done 是最后一帧:图流到此结束,runner 随即把本轮记忆原料(快照)交给
    后台记忆管道,不再占用请求流。内容与原 finalize 的 done 同源:trace +
    本轮检索相关分/次数(service 层质检门据此扣留 done)。
    """
    w = get_stream_writer()
    recorder = _recorder(config)
    trace_id = state["trace_id"]
    final_reason = state.get("final_reason") or "answer"
    w({"type": "done", "trace_id": trace_id, "trace": recorder.to_dict(),
       # Req5:done 强制带终态原因(answer/max_steps/timeout/error),前端/service 据此区分。
       "final_reason": final_reason,
       # 本轮最高检索相关分 + 检索次数:供 service 层质检门判低置信(simple 直答不产生)
       "retrieval_max_score": float(state.get("retrieval_max_score") or 0.0),
       "search_count": int(state.get("search_count") or 0)})
    # 后台记忆管道原料快照(runner 经 configurable 注入 holder,流结束后取走提交);
    # 无 holder(测试直调节点)则跳过。运行时对象走 configurable,不进 state。
    try:
        holder = ((config or {}).get("configurable") or {}).get("mem_snapshot")
        if isinstance(holder, dict):
            holder["messages"] = list(state.get("messages") or [])
            holder["full_reply"] = state.get("full_reply") or ""
            holder["final_reason"] = final_reason
    except Exception:  # noqa: BLE001  快照失败不影响 done
        pass
    return {}