# -*- coding: utf-8 -*-
"""工作记忆压缩:跨轮 messages 摘要(纯 token 预算 + 后台实时刷新)。

属于工作记忆层(存 state 快照的同层策略),放在 memories/storage/working/。
模型(类似 Claude Code 的短期记忆压缩):
  - 不按轮数兜底,上下文预算完全按 token。
  - 始终保留一段近期原文窗口(KEEP_RECENT_TOKENS,按完整轮次切,不拆半轮)。
  - 累积原文超过 SUMMARY_START_TOKENS 时,fork 后台子代理开始/刷新摘要
    (增量:旧摘要 + 新进入"旧区"的内容一起浓缩),并在随后每轮持续刷新——摘要实时更新。
  - 累积原文达到 COMPACT_TRIGGER_TOKENS 时,在发 LLM 之前把摘要带回、用
    RemoveMessage 删掉近期窗口之外的全部旧原文,只留摘要 + 近期原文。
  - 压缩后原文从 KEEP_RECENT_TOKENS 重新增长,重复整个流程,保证不溢出 40k 上下文。

三个阈值基于模型原生上限 40960(Qwen3-14B-AWQ max_position_embeddings;vLLM 以
--max-model-len 40960 启动):输出预留 8k、固定输入开销 4k(system+tools+长期召回+
当前问题),对话历史预算 ~28.7k。最坏情况(压缩前一轮)输入 ≈ 22k 原文 + 1.8k 摘要
+ 4k 固定 = 27.8k,加 8k 输出 = 35.8k,留 ~5k 估错裕量;token 估算偏保守高估。

延迟优化:摘要在临界点前由 daemon 线程池预生成,调用方(react/finalize_node)用"本轮
结束后"的 messages 算边界(=下一轮 build_messages 所见 existing),缓存键为待删消息
id 集合,故压缩时可零 LLM 延迟直接取;未命中/未完成则同步兜底。
"""
from __future__ import annotations

import collections
import concurrent.futures
import json
import logging
import os
import sys
import threading
import time
from typing import Any, Optional

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)

# 本文件位于 memories/storage/working/;项目根在上三级
_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
if os.path.join(_ROOT, "config") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "config"))
import config as C  # noqa: E402
from .._llm import chat_completion_with_fallback  # noqa: E402

logger = logging.getLogger("agent")

# ---- 上下文预算(模型原生上限 40960,取自 Qwen3-14B-AWQ config.max_position_embeddings;
#      vLLM 须以 --max-model-len 40960 启动,二者对齐) ----
MODEL_CONTEXT_TOKENS = 40_960
OUTPUT_RESERVE_TOKENS = 8_192        # 输出预留(长回答 + 工具循环)
FIXED_OVERHEAD_TOKENS = 4_096        # system + tools + 长期召回 + 当前问题
CONVERSATION_BUDGET = MODEL_CONTEXT_TOKENS - OUTPUT_RESERVE_TOKENS - FIXED_OVERHEAD_TOKENS  # 28_672

KEEP_RECENT_TOKENS = 6_000           # 始终保留的近期原文窗口(按整轮切)
SUMMARY_START_TOKENS = 13_000        # 原文超此值开始/刷新后台摘要
COMPACT_TRIGGER_TOKENS = 22_000      # 原文达此值则压缩(摘要替换旧原文)
SUMMARY_MAX_TOKENS = 1_800           # 摘要目标上限
SUMMARY_MAX_CHARS = 2_800            # 摘要硬截断(CJK ~1 token/字,兜底防失控)

# 单次喂给摘要 LLM 的消息条数硬上限(防御性)。
SUMMARIZE_BATCH = 80
_LLM_RETRIES = 3
# 主请求线程内同步摘要兜底的单次读超时(秒):刻意短,宁可降级不压缩也不长等。
_SYNC_SUMMARIZE_TIMEOUT = 8.0

_SUMMARY_PROMPT = """你是对话摘要器。下面是一段多轮对话(可能含工具调用与工具结果)。
请渐进式地更新摘要:在已有摘要基础上,把新增对话内容浓缩进去,保留后续问答可能用到的
关键事实、用户偏好、已讨论的实体、结论与待办。不要罗列流水账,用简洁中文输出,
控制在约 {max_tokens} token 以内。只输出摘要正文,不要任何前缀或解释。

已有摘要:
{prev_summary}

新增对话内容:
{transcript}
"""


# ---------------------------------------------------------------- LLM
def _llm_summarize(prev_summary: str, transcript: str,
                   trace_id: str = "", *,
                   retries: int = _LLM_RETRIES,
                   timeout: Optional[float] = None) -> Optional[str]:
    """调 LLM 增量更新摘要;失败返回 None。

    :param retries: 主模型重试次数。后台预生成用默认(多次重试,延迟不敏感);
                    主请求线程的同步兜底应传 1(单次,避免长等)。
    :param timeout: 单次请求读超时(秒);None 用客户端默认 20s。同步兜底应传短超时。
    """
    prompt = _SUMMARY_PROMPT.format(
        max_tokens=SUMMARY_MAX_TOKENS,
        prev_summary=prev_summary or "(无)",
        # 取尾部:增量刷新时更早的旧区已在 prev_summary 里,尾部才是新进入旧区、
        # 尚未被摘要的轮次;首次刷新时旧区最小,尾部即全部。
        transcript=transcript[-16000:],
    )
    # 主模型重试耗尽后自动切备用模型(见 memories/storage/_llm.py)
    resp, err = chat_completion_with_fallback(
        trace_id=f"summarize-{trace_id}" if trace_id else "summarize",
        retries=retries,
        model=C.MODEL_LIGHT,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        **({"timeout": timeout} if timeout else {}),
    )
    if err is not None:
        logger.info(json.dumps({
            "trace_id": trace_id, "event": "summarize_llm_fail",
            "error": f"{type(err).__name__}: {err}",
        }, ensure_ascii=False))
        return None
    text = (resp.choices[0].message.content or "").strip()
    return text or None


# ---------------------------------------------------------------- token 估算
def _estimate_tokens(text: str) -> int:
    """粗估 token:CJK ~1 token,空白分词 ~1.3 token/词(保守高估)。"""
    if not text:
        return 0
    cjk = sum(1 for c in text if "一" <= c <= "鿿")
    words = len(text.split())
    return int(cjk + words * 1.3)


def _message_tokens(m: BaseMessage) -> int:
    """单条对话消息进 LLM 的 token 估算(含 role 开销);system 不计入对话累积。"""
    if isinstance(m, SystemMessage):
        return 0
    overhead = 4
    if isinstance(m, AIMessage):
        tcs = getattr(m, "tool_calls", None) or []
        tc_text = json.dumps(
            [{"name": tc.get("name"), "args": tc.get("args")} for tc in tcs],
            ensure_ascii=False,
        ) if tcs else ""
        return overhead + _estimate_tokens(str(m.content or "")) + _estimate_tokens(tc_text)
    if isinstance(m, ToolMessage):
        return overhead + _estimate_tokens(str(m.content or ""))
    return overhead + _estimate_tokens(str(m.content or ""))


def _convo_without_system(messages: list[BaseMessage]) -> list[BaseMessage]:
    return [m for m in messages if not isinstance(m, SystemMessage)]


def _convo_tokens(convo: list[BaseMessage]) -> int:
    return sum(_message_tokens(m) for m in convo)


# ---------------------------------------------------------------- 边界算法
def _keep_index_by_tokens(convo: list[BaseMessage],
                          keep_recent_tokens: int) -> int:
    """返回"保留段起始下标":从该下标到结尾的完整轮次 token 总量 <= 预算。

    按 HumanMessage 切轮,从最后一轮向前累加,在不超预算的前提下尽量多保留近期轮次;
    即使最后一轮本身超预算,也至少保留最后一轮(不把当前轮拆半)。
    """
    human_idx = [i for i, m in enumerate(convo) if isinstance(m, HumanMessage)]
    if not human_idx:
        return len(convo)

    spans = []
    for j, start in enumerate(human_idx):
        end = human_idx[j + 1] if j + 1 < len(human_idx) else len(convo)
        cost = sum(_message_tokens(convo[k]) for k in range(start, end))
        spans.append((start, cost))

    accum = 0
    keep_from = spans[-1][0]  # 至少保留最后一轮
    for start, cost in reversed(spans):
        if accum + cost <= keep_recent_tokens:
            accum += cost
            keep_from = start
        else:
            break
    return keep_from


def calculate_compaction_keep_index(
    messages: list[BaseMessage],
    keep_recent_tokens: int = KEEP_RECENT_TOKENS,
    compact_trigger: int = COMPACT_TRIGGER_TOKENS,
) -> Optional[int]:
    """压缩边界。原文(非 system)总量 >= compact_trigger 时返回保留段起始下标,
    ``convo[:keep_from]`` 即应被摘要替换的旧原文;否则 None。"""
    convo = _convo_without_system(messages)
    if _convo_tokens(convo) < compact_trigger:
        return None
    keep_from = _keep_index_by_tokens(convo, keep_recent_tokens)
    return keep_from if keep_from > 0 else None


def calculate_pregen_keep_index(
    messages: list[BaseMessage],
    keep_recent_tokens: int = KEEP_RECENT_TOKENS,
    summary_start: int = SUMMARY_START_TOKENS,
) -> Optional[int]:
    """后台摘要边界。原文总量 >= summary_start 时返回与压缩一致的保留段起始下标
    (即此刻"旧区"已足够大,值得提前摘要);否则 None。

    保留段窗口与 calculate_compaction_keep_index 完全一致,保证预生成与真实压缩
    算出的待删消息集合相同。
    """
    convo = _convo_without_system(messages)
    if _convo_tokens(convo) < summary_start:
        return None
    keep_from = _keep_index_by_tokens(convo, keep_recent_tokens)
    return keep_from if keep_from > 0 else None


def _removed_messages(messages: list[BaseMessage],
                      keep_from: Optional[int]) -> list[BaseMessage]:
    if keep_from is None:
        return []
    return _convo_without_system(messages)[:keep_from][-SUMMARIZE_BATCH:]


# ---------------------------------------------------------------- 文本化
def _msg_to_text(m: BaseMessage) -> str:
    if isinstance(m, SystemMessage):
        return ""
    if isinstance(m, HumanMessage):
        return f"用户: {m.content}"
    if isinstance(m, ToolMessage):
        content = str(m.content)
        if len(content) > 800:
            content = content[:800] + "…"
        return f"工具结果[{getattr(m, 'tool_call_id', '')}]: {content}"
    if isinstance(m, AIMessage):
        parts = []
        if m.content:
            parts.append(str(m.content))
        tcs = getattr(m, "tool_calls", None) or []
        for tc in tcs:
            args = json.dumps(tc.get("args", {}), ensure_ascii=False)
            if len(args) > 300:
                args = args[:300] + "…"
            parts.append(f"调用工具 {tc.get('name')}({args})")
        return "助手: " + " | ".join(parts) if parts else ""
    return ""


def _build_transcript(old: list[BaseMessage]) -> str:
    return "\n".join(t for t in (_msg_to_text(m) for m in old) if t)


# ---------------------------------------------------------------- 预生成缓存
# thread_id -> {"key": (frozenset[removed_ids], prev_summary), "future": Future}
# 有界 LRU:长生命周期进程里活跃 thread 数不可控,不加上限会随会话数线性泄漏。
# 命中/写入时移到末尾;超限时淘汰最久未访问的条目(其后台 future 若仍在跑则
# 放弃结果——daemon 线程会自然结束,不影响主流程)。
_PREGEN_MAX_ENTRIES = max(64, int(os.getenv("SUMMARY_PREGEN_CACHE_MAX", "512")))
_pregen: "collections.OrderedDict[str, dict[str, Any]]" = collections.OrderedDict()
_pregen_lock = threading.Lock()


def _removed_ids(old: list[BaseMessage]) -> frozenset[str]:
    return frozenset(m.id for m in old if getattr(m, "id", None))


def _submit_pregen(prev_summary: str, transcript: str, trace_id: str):
    fut: concurrent.futures.Future = concurrent.futures.Future()

    def _run():
        try:
            result = _llm_summarize(prev_summary, transcript, trace_id)
            if not fut.done():
                fut.set_result(result)
        except Exception as e:  # noqa: BLE001
            if not fut.done():
                fut.set_exception(e)

    threading.Thread(target=_run, daemon=True, name="summary-pregen").start()
    return fut


def _consume_pregen(thread_id: Optional[str], removed_ids: frozenset[str],
                    prev_summary: str) -> Optional[str]:
    if not thread_id:
        return None
    with _pregen_lock:
        entry = _pregen.get(thread_id)
        if not entry or entry["key"] != (removed_ids, prev_summary or ""):
            return None
        _pregen.move_to_end(thread_id)  # LRU 命中
        fut = entry["future"]
    if not fut.done():
        return None  # 仍在跑:本次同步兜底,不等待
    try:
        result = fut.result(0)
    except Exception:
        result = None
    with _pregen_lock:
        _pregen.pop(thread_id, None)
    return result


def schedule_pregeneration(messages: list[BaseMessage], prev_summary: str,
                            thread_id: Optional[str], trace_id: str = "") -> None:
    """临界前实时刷新摘要:原文达 SUMMARY_START 后,后台增量摘要"旧区"内容。

    每轮由调用方(react/finalize_node)用"本轮消息已落 state"的 messages 调用一次,
    为下一轮压缩预热。待删集合变化(新轮进入旧区)时重新提交,使摘要保持最新。
    未达阈值则清缓存。
    """
    if not thread_id:
        return
    keep_from = calculate_pregen_keep_index(messages)
    if keep_from is None:
        with _pregen_lock:
            _pregen.pop(thread_id, None)
        return

    old = _removed_messages(messages, keep_from)
    if not old:
        return
    removed_ids = _removed_ids(old)
    transcript = _build_transcript(old)
    key = (removed_ids, prev_summary or "")

    with _pregen_lock:
        entry = _pregen.get(thread_id)
        if entry is not None and entry["key"] == key:
            _pregen.move_to_end(thread_id)
            return  # 同一批内容已在算/已算好,不重复
        _pregen[thread_id] = {
            "key": key,
            "future": _submit_pregen(prev_summary or "", transcript, trace_id),
        }
        _pregen.move_to_end(thread_id)
        # 超上限淘汰最久未访问条目
        while len(_pregen) > _PREGEN_MAX_ENTRIES:
            _pregen.popitem(last=False)


# ---------------------------------------------------------------- 真实压缩
def maybe_summarize(state: dict[str, Any], trace_id: str = "",
                    thread_id: Optional[str] = None) -> dict[str, Any]:
    """原文达 COMPACT_TRIGGER 时压缩:用摘要替换旧原文,返回 state 更新。

    优先消费后台实时刷新的摘要(零 LLM 延迟);未命中则同步兜底。
    返回 {"summary": str, "messages": [RemoveMessage,...]};无需压缩返回 {}。
    """
    messages: list[BaseMessage] = state.get("messages") or []
    keep_from = calculate_compaction_keep_index(messages)
    if keep_from is None:
        return {}

    old = _removed_messages(messages, keep_from)
    if not old:
        return {}
    removed_ids = _removed_ids(old)
    prev_summary = state.get("summary") or ""

    new_summary = _consume_pregen(thread_id, removed_ids, prev_summary)
    from_pregen = new_summary is not None
    if new_summary is None:
        # 同步兜底运行在用户请求线程:只做单次、短超时(主+备各至多 1 次),
        # 绝不在此多次重试长等(否则 LLM 故障时可把请求挂起 ~2 分钟)。
        # 失败即降级返回 {}——不删除原文,下一轮后台预生成还有机会补上。
        new_summary = _llm_summarize(
            prev_summary, _build_transcript(old), trace_id,
            retries=1, timeout=_SYNC_SUMMARIZE_TIMEOUT)
    if not new_summary:
        return {}  # 摘要失败:不删除,降级原样保留

    remove = [RemoveMessage(id=m.id) for m in old if getattr(m, "id", None)]
    logger.info(json.dumps({
        "trace_id": trace_id, "event": "messages_compacted",
        "removed": len(remove),
        "removed_tokens": sum(_message_tokens(m) for m in old),
        "keep_recent_tokens": KEEP_RECENT_TOKENS,
        "from_pregen": from_pregen,
        "summary_len": len(new_summary),
    }, ensure_ascii=False))
    return {"summary": new_summary, "messages": remove}


def format_summary_block(summary: Optional[str]) -> str:
    """把摘要拼成注入 system prompt 的文本块(无摘要返回空串);硬截断防失控。"""
    if not summary:
        return ""
    text = summary.strip()
    if len(text) > SUMMARY_MAX_CHARS:
        text = text[:SUMMARY_MAX_CHARS] + "…"
    return "## 此前对话摘要\n" + text
