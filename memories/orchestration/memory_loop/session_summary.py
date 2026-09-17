# -*- coding: utf-8 -*-
"""memory-loop 节点二【会话摘要与 Auto-Compact】。

制品:每会话一个 session-memory.md(9 章节正文 + front-matter 游标,权威存盘,
见 storage/working/session_file.py)。游标(cursor_msg_id/last_tokens/...)以文件
front-matter 为准,state 不存游标。

- 首摘:文件不存在且累计原文 tokens > MEM_SUMMARY_FIRST_TOKENS(1万) -> 生成 9 章节,
  游标 = 末条消息。
- 更新:文件存在;自游标起 条件A(Δtokens≥5000 且 Δ工具调用≥3)或
  条件B(本轮纯文本无工具调用 且 Δtokens≥2000) -> 增量合并、游标推进。
- 三级降级:① 角色化"子代理"LLM;② 失败 -> 短 prompt 直连 API(单次);
  ③ 仍失败 -> 规则截断(不调 LLM,把最后 N 个完整轮次原文写入文件附录;不 compact)。
- Auto-Compact(水位动作,与摘要解耦):原文 tokens ≥ W − reserve 时,用摘要替换
  游标前旧原文(RemoveMessage),保留近期 10K–40K(默认 20K)tokens,边界落在完整
  轮次、不切断 tool_use/tool_result 配对、保留区≥5 条文本消息;游标重置到替换点。

全部旁路:任何异常 -> 降级/放弃本轮,绝不外抛。
"""
from __future__ import annotations

import logging
import time
from typing import Any, Optional

import config as C
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    ToolMessage,
)

from ...storage._llm import chat_completion_with_fallback
from ...storage.working import summarize as S
from ...storage.working.session_file import (
    SessionFile,
    empty_template_body,
)
from ..short.recall import recent_high_watermark
from . import prompts

logger = logging.getLogger("agent")

ROUTE_DONE = "done"
ROUTE_DEGRADE = "degrade"


# ---------------------------------------------------------------- 计数工具
def _tool_calls_of(m: BaseMessage) -> int:
    if isinstance(m, AIMessage):
        return len(getattr(m, "tool_calls", None) or [])
    return 0


def _is_text_msg(m: BaseMessage) -> bool:
    """Human/AI 且有非空文本内容(ToolMessage/纯工具调用 AI 不算)。"""
    if isinstance(m, (HumanMessage, AIMessage)) and str(getattr(m, "content", "") or "").strip():
        return True
    return False


def _cursor_index(convo: list[BaseMessage], meta: dict[str, Any]) -> int:
    """游标在 convo 中的下标(其【后】为新消息);找不到返回 -1(全部视为新)。"""
    cid = meta.get("cursor_msg_id") or ""
    if cid:
        for i, m in enumerate(convo):
            if getattr(m, "id", None) == cid:
                return i
    idx = int(meta.get("cursor_index") or 0)
    return idx if 0 < idx < len(convo) else -1


def _turn_tool_calls(convo: list[BaseMessage]) -> int:
    """本轮(最后一个 HumanMessage 起)的工具调用数;用于条件B"纯文本轮"判定。"""
    last_human = -1
    for i in range(len(convo) - 1, -1, -1):
        if isinstance(convo[i], HumanMessage):
            last_human = i
            break
    if last_human < 0:
        return 0
    return sum(_tool_calls_of(m) for m in convo[last_human:])


# ---------------------------------------------------------------- LLM(三级降级)
def _llm_subagent(transcript: str, prev: str,
                  deadline: Optional[float] = None) -> Optional[str]:
    resp, err = chat_completion_with_fallback(
        trace_id="mem-summary-subagent",
        retries=int(getattr(C, "MEM_SUMMARY_LLM_RETRIES", 1)),
        model=getattr(C, "MODEL_LIGHT", None),
        messages=[
            {"role": "system", "content": prompts.SUBAGENT_SYS},
            {"role": "user", "content": prompts.build_user_prompt(transcript, prev)},
        ],
        temperature=0.2,
        timeout=float(getattr(C, "MEM_SUMMARY_LLM_TIMEOUT", 3.0)),
        deadline=deadline,
    )
    if err is not None or resp is None:
        return None
    try:
        text = (resp.choices[0].message.content or "").strip()
    except Exception:
        return None
    return text or None


def _llm_direct(transcript: str, prev: str,
                deadline: Optional[float] = None) -> Optional[str]:
    resp, err = chat_completion_with_fallback(
        trace_id="mem-summary-direct",
        retries=1,
        model=getattr(C, "MODEL_LIGHT", None) or "",
        messages=[
            {"role": "system", "content": prompts.DIRECT_SYS},
            {"role": "user", "content": prompts.build_user_prompt(transcript, prev)},
        ],
        temperature=0.2,
        timeout=float(getattr(C, "MEM_SUMMARY_LLM_TIMEOUT", 3.0)),
        deadline=deadline,
    )
    if err is not None or resp is None:
        return None
    try:
        text = (resp.choices[0].message.content or "").strip()
    except Exception:
        return None
    return text or None


def _rule_body(transcript: str) -> str:
    """级别3:不调 LLM。9 章节空骨架 + 附录"近期对话原文"(本轮/新段原文)。"""
    body = empty_template_body()
    body += "\n\n## 附:近期对话原文(规则降级,未做 LLM 浓缩)\n\n" + (transcript or "").strip()
    return body


def _summarize(transcript: str, prev: str,
               deadline: Optional[float] = None) -> tuple[Optional[str], str]:
    """返回 (body, level):level ∈ subagent/direct/rule。经济模式直接规则。

    deadline(整链预算)透传给两级 LLM:预算耗尽时调用立即失败,
    自然落到 level=rule(零 LLM 成本,摘要能力保留)。
    """
    if not getattr(C, "ECONOMY_MODE", False):
        body = _llm_subagent(transcript, prev, deadline=deadline)
        if body:
            return _cap(body), "subagent"
        body = _llm_direct(transcript, prev, deadline=deadline)
        if body:
            return _cap(body), "direct"
    return _cap(_rule_body(transcript)), "rule"


def _cap(text: str) -> str:
    n = int(getattr(C, "MEM_SUMMARY_MAX_CHARS", 4000))
    text = text.strip()
    return text if len(text) <= n else text[:n] + "…"


# ---------------------------------------------------------------- Auto-Compact 边界
def _compact_trigger() -> int:
    w = int(getattr(C, "MEM_MODEL_CONTEXT_TOKENS", 40960))
    reserve = max(int(getattr(C, "MEM_COMPACT_RESERVE_MIN", 13000)),
                  int(w * float(getattr(C, "MEM_COMPACT_RESERVE_RATIO", 0.10))))
    return w - reserve


def _keep_budget() -> int:
    lo = int(getattr(C, "MEM_KEEP_RECENT_MIN", 10000))
    hi = int(getattr(C, "MEM_KEEP_RECENT_MAX", 40000))
    default = int(getattr(C, "MEM_KEEP_RECENT_DEFAULT", 20000))
    return max(lo, min(hi, default))


def _tool_pair_intact(segment: list[BaseMessage]) -> bool:
    """段内每个 AIMessage.tool_call 的 id 都有配对 ToolMessage 且在段内。"""
    pending: set[str] = set()
    answered: set[str] = set()
    for m in segment:
        if isinstance(m, AIMessage):
            for tc in (getattr(m, "tool_calls", None) or []):
                if tc.get("id"):
                    pending.add(tc["id"])
        elif isinstance(m, ToolMessage):
            tcid = getattr(m, "tool_call_id", None)
            if tcid:
                answered.add(tcid)
    return pending.issubset(answered)


def _compact_keep_from(convo: list[BaseMessage]) -> Optional[int]:
    """返回保留段起始下标(其前为待摘要替换的旧原文);不该/不能压缩返回 None。"""
    total = S._convo_tokens(convo)
    if total < _compact_trigger():
        return None
    keep_from = S._keep_index_by_tokens(convo, _keep_budget())
    if keep_from <= 0:
        return None
    # 配对约束:替换段不得切断 tool_use/tool_result;不满足则前移到该轮起点
    while keep_from > 0 and not _tool_pair_intact(convo[:keep_from]):
        # 前移到上一个 HumanMessage 轮次起点
        prev_human = 0
        for i in range(keep_from - 1, -1, -1):
            if isinstance(convo[i], HumanMessage):
                prev_human = i
                break
        if prev_human == 0 and not isinstance(convo[0], HumanMessage):
            return None
        keep_from = prev_human
        if keep_from <= 0:
            return None
    # 保留区至少 MEM_MIN_TEXT_MESSAGES 条文本消息
    kept_text = sum(1 for m in convo[keep_from:] if _is_text_msg(m))
    if kept_text < int(getattr(C, "MEM_MIN_TEXT_MESSAGES", 5)):
        return None
    return keep_from


# ---------------------------------------------------------------- 主逻辑
def run_session_maintenance(username: Optional[str], thread_id: Optional[str],
                            messages: list[BaseMessage],
                            *, final_reason: Optional[str] = "answer",
                            deadline: Optional[float] = None
                            ) -> dict[str, Any]:
    """节点二同步逻辑。

    deadline:整链时间预算(monotonic),透传给摘要 LLM(耗尽自动落 rule 级降级)。
    返回 {"route": done/degrade, "summary": str(状态回灌串), "remove": [RemoveMessage],
          "summarized": bool, "compacted": bool, "level": str}。
    最外层不抛。

    同会话互斥:读游标→摘要(LLM)→写文件整周期持 per-session 文件锁。原子替换只保证
    不写坏,两个并发周期会互相覆盖摘要游标/重复生成待删 id(管理工具、未来放开同用户
    并发、多 worker 时可达)。同会话请求本就该串行——后一条的回答应建立在前一条记忆之上;
    等锁超时按"本轮不摘要"降级,绝不阻塞问答主链路。
    """
    empty = {"route": ROUTE_DONE, "summary": "", "remove": [],
             "summarized": False, "compacted": False, "level": ""}
    try:
        if not getattr(C, "MEM_LOOP_ENABLED", True) or not username or not thread_id:
            return empty
        convo = S._convo_without_system(messages or [])
        if not convo:
            return empty

        sf = SessionFile(username, thread_id)
        with sf.lock(timeout=float(getattr(C, "MEM_SESSION_MAINT_LOCK_TIMEOUT", 30.0))):
            return _run_locked(username, thread_id, convo, sf, deadline=deadline)
    except TimeoutError:
        logger.info("memory-loop session maintenance: lock timeout, degrade")
        return {"route": ROUTE_DEGRADE, "summary": "", "remove": [],
                "summarized": False, "compacted": False, "level": "lock_timeout"}
    except Exception as e:  # noqa: BLE001  节点二绝不冒泡
        logger.info("memory-loop session maintenance failed: %s: %s",
                    type(e).__name__, str(e)[:120])
        return {"route": ROUTE_DEGRADE, "summary": "", "remove": [],
                "summarized": False, "compacted": False, "level": "error"}


def _run_locked(username: Optional[str], thread_id: str, convo: list[BaseMessage],
                sf: SessionFile, *, deadline: Optional[float] = None
                ) -> dict[str, Any]:
    """锁内主体:读游标→按阈值摘要/写文件→Auto-Compact 先写后删。异常由调用方兜底降级。"""
    try:
        meta_typed = sf.read_meta_typed()
        exists = sf.exists()
        meta, prev_body = sf.read()
        total_tokens = S._convo_tokens(convo)
        cur_i = _cursor_index(convo, meta) if exists else -1
        new_seg = convo[cur_i + 1:] if cur_i >= 0 else convo
        new_tokens = S._convo_tokens(new_seg)
        new_tools = sum(_tool_calls_of(m) for m in new_seg)

        # Req14:新增轮次阈值——游标后新增的【用户轮次】达 MEM_SUMMARY_MIN_NEW_TURNS 才考虑摘要。
        new_turns = sum(1 for m in new_seg if isinstance(m, HumanMessage))
        min_turns = int(getattr(C, "MEM_SUMMARY_MIN_NEW_TURNS", 2))

        do_first = (not exists) and total_tokens > int(
            getattr(C, "MEM_SUMMARY_FIRST_TOKENS", 10000))
        cond_a = (exists and new_tokens >= int(
            getattr(C, "MEM_SUMMARY_UPDATE_TOKEN_TOOL", 5000))
            and new_tools >= int(getattr(C, "MEM_SUMMARY_UPDATE_TOOL_CALLS", 3)))
        cond_b = (exists and _turn_tool_calls(convo) == 0
                  and new_tokens >= int(getattr(C, "MEM_SUMMARY_UPDATE_TOKEN_TEXT", 2000)))
        # 轮次阈值作为更新触发的并列条件之一(cond_a/cond_b 还需轮次达标;首摘不受轮次限制)。
        turn_ok = do_first or new_turns >= min_turns

        body = prev_body
        level = ""
        summarized = False
        if (do_first or cond_a or cond_b) and turn_ok:
            transcript = S._build_transcript(new_seg if exists else convo)
            body, level = _summarize(transcript, prev_body if exists else "",
                                     deadline=deadline)
            summarized = body is not None
            # 写文件(游标推进到末尾 + 短期流水高水位);失败 -> 降级:不 compact、不动 messages
            try:
                end = convo[-1]
                new_meta = {
                    "cursor_msg_id": getattr(end, "id", "") or "",
                    "cursor_index": len(convo) - 1,
                    "last_tokens": total_tokens,
                    "last_tool_calls": sum(_tool_calls_of(m) for m in convo),
                    # 摘要游标与短期流水游标【同一处写入】:recent 预取只读该 seq 之后,
                    # 保证摘要覆盖范围与近期原文窗口严格互斥。
                    "recent_cursor_seq": recent_high_watermark(thread_id),
                    "pending_remove_ids": [],
                    "version": int(getattr(C, "MEM_SUMMARY_VERSION", 1)),
                    "updated_at": int(time.time()),
                }
                sf.write(new_meta, body or empty_template_body())
            except Exception as e:  # noqa: BLE001  文件不可写
                logger.info("memory-loop session file write failed: %s: %s",
                            type(e).__name__, str(e)[:120])
                return {"route": ROUTE_DEGRADE, "summary": "", "remove": [],
                        "summarized": False, "compacted": False, "level": "file_fail"}

        # ---- Auto-Compact(水位动作)----
        # Req6【先写后删幂等】:只有在摘要文件 + 游标 + 待删 id 列表全部落盘成功后,
        # 才返回 RemoveMessage。重跑同轮时,已记录在 front-matter 的 pending_remove_ids
        # 视为已处理——不重复生成 RemoveMessage(RemoveMessage 不存在的 id 本身无害,
        # 但幂等去重可避免跨轮重复删/重复计数)。
        remove: list[RemoveMessage] = []
        compacted = False
        summary_out = ""
        already_pending = set(meta_typed.get("pending_remove_ids") or [])
        keep_from = _compact_keep_from(convo)
        if keep_from is not None:
            # 规则降级(level=rule)不 compact:只保留近期原文附录,不删消息,下轮再试
            if level == "rule":
                logger.info("memory-loop compact skipped: rule-level summary")
            else:
                seg = convo[:keep_from]
                # 待删 id:有 id 且【未在上一轮落盘的 pending 列表里】(幂等去重)。
                pending_ids = [m.id for m in seg
                               if getattr(m, "id", None) and m.id not in already_pending]
                summary_out = body or prev_body
                if pending_ids and summary_out:
                    # 1) 先落盘:游标重置到替换点 + 待删 id 列表 + 流水高水位。
                    try:
                        anchor = convo[keep_from]
                        kept_tokens = S._convo_tokens(convo[keep_from:])
                        sf.write({
                            "cursor_msg_id": getattr(anchor, "id", "") or "",
                            "cursor_index": keep_from,
                            "last_tokens": kept_tokens,
                            "last_tool_calls": sum(
                                _tool_calls_of(m) for m in convo[keep_from:]),
                            "recent_cursor_seq": recent_high_watermark(thread_id),
                            "pending_remove_ids": pending_ids,
                            "version": int(getattr(C, "MEM_SUMMARY_VERSION", 1)),
                            "updated_at": int(time.time()),
                        }, summary_out)
                    except Exception as e:  # noqa: BLE001
                        # 落盘失败:绝不返回 RemoveMessage(先写后删),下轮按文件游标重建。
                        logger.info("memory-loop compact cursor write failed: %s: %s",
                                    type(e).__name__, str(e)[:120])
                        pending_ids = []
                    # 2) 落盘成功后才生成 RemoveMessage。
                    if pending_ids:
                        remove = [RemoveMessage(id=mid) for mid in pending_ids]
                        compacted = True
        return {"route": ROUTE_DONE, "summary": summary_out, "remove": remove,
                "summarized": summarized, "compacted": compacted, "level": level}
    except Exception as e:  # noqa: BLE001  节点二绝不冒泡
        logger.info("memory-loop session maintenance failed: %s: %s",
                    type(e).__name__, str(e)[:120])
        return {"route": ROUTE_DEGRADE, "summary": "", "remove": [],
                "summarized": False, "compacted": False, "level": "error"}
