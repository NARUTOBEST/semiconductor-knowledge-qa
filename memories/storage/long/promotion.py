# -*- coding: utf-8 -*-
"""记忆升迁流水线(任务结束后异步触发)。

流程(对照 memory-system-design):
  短期 session_events 流水 -> 过滤噪声 -> LLM 萃取事实 -> 去重 -> 长期 long_term_memories

- 只升迁有意义的对话内容(user/assistant 消息);status/token 等噪声事件丢弃。
- 去重由长期库唯一索引 (user_id, md5(content)) 保证;重复事实静默跳过。
- LLM 调用失败不抛到主流程,返回空列表(升迁是旁路,不阻塞业务)。
- 异步由调用方决定(本模块提供同步函数,可用 asyncio.to_thread / BackgroundTasks 包裹)。
"""
import json
import logging
import re
from typing import Any, Optional

import sys, os
# 本文件位于 memories/storage/long/;项目根在上三级
_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
if os.path.join(_ROOT, "config") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "config"))
import config as C  # noqa: E402

from ..short.short_term import short_term  # noqa: E402
from .._llm import chat_completion_with_fallback  # noqa: E402
from .long_term import long_term  # noqa: E402

logger = logging.getLogger("agent")

# 升迁连续失败达到此次数后,强制推进水位丢弃这批事件,避免永久性 LLM 故障
# 导致 backlog 无限增长、每轮重放越来越长的历史。
MAX_PROMOTION_FAILURES = 5


# 参与升迁的事件类型;其余(status/token/step_*/meta 等过程噪声)丢弃
_CONTENT_EVENTS = {"user_message", "assistant_message", "tool_result"}

_EXTRACT_PROMPT = """你是记忆萃取器。从下面这段对话中,抽取值得长期记住的关于【用户】的事实或偏好,
用于后续个性化推理。要求:
1. 只抽取明确陈述的事实/偏好(如用户身份、领域、设备型号、习惯、明确结论),不要臆测;
2. 每条一句中文,自包含、可脱离上下文理解;
3. 去掉寒暄、提问语气、过程性内容;
4. 最多 8 条;没有值得记忆的内容就返回空数组。

只返回 JSON 数组,例如 ["事实1","事实2"],不要任何其他文字。

对话:
{transcript}
"""

# 升迁守门员:萃取前先用极便宜的一次调用判断本轮是否值得升迁。
# 保守策略——拿不准一律 yes,宁可多萃取一次也不漏记事实。
_GATE_PROMPT = """判断下面这段对话是否包含值得长期记住的关于【用户】的事实或偏好(如身份、领域、设备型号、使用习惯、明确的结论或决定)。

- 纯寒暄、问候、致谢、简短确认、无实质内容 → 输出 no
- 含任何值得记住的信息 → 输出 yes
- 拿不准时一律输出 yes

只输出一个英文单词 yes 或 no,不要任何其他文字。

对话:
{transcript}
"""


def _build_transcript(events: list[dict[str, Any]]) -> str:
    lines = []
    for ev in events:
        etype = ev.get("event_type")
        if etype not in _CONTENT_EVENTS:
            continue
        payload = ev.get("payload") or {}
        content = payload.get("content") or payload.get("text") or ""
        if not content:
            continue
        role = {"user_message": "用户", "assistant_message": "助手",
                "tool_result": "工具"}.get(etype, etype)
        lines.append(f"{role}: {str(content)[:1000]}")
    return "\n".join(lines)


def _extract_facts(transcript: str) -> Optional[list[str]]:
    """调 LLM 萃取事实。

    返回 None 表示 LLM/解析失败(调用方不应推进水位,留待下次重试);
    返回 [] 表示成功但确实没有值得记忆的事实。
    """
    if not transcript.strip():
        return []
    # 主模型重试耗尽后自动切备用模型(见 memories/storage/_llm.py)
    resp, err = chat_completion_with_fallback(
        trace_id="promotion-extract",
        model=C.OPENAI_TEXT_MODEL,
        messages=[{"role": "user",
                   "content": _EXTRACT_PROMPT.format(transcript=transcript)}],
        temperature=0.0,
        max_tokens=400,
        timeout=20,
    )
    if err is not None:
        logger.warning("promotion extract LLM failed: %s: %s", type(err).__name__, err)
        return None
    text = (resp.choices[0].message.content or "").strip()

    # 剥离可能的 ```json 代码块
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        facts = json.loads(text)
        if not isinstance(facts, list):
            return []
        return [str(f).strip() for f in facts if str(f).strip()]
    except Exception:
        return None


def _should_promote(transcript: str) -> Optional[bool]:
    """守门员:用一次极轻量调用判断本轮对话是否值得升迁。

    返回 True=值得升迁(继续跑萃取);False=无价值(跳过萃取);
    None=LLM 调用失败(调用方不应推进水位,留待下次重试)。
    响应无法解析时 fail-open 返回 True,交给萃取器裁决,避免漏记事实。
    """
    if not transcript.strip():
        return False
    # 守门员同样享受主/备模型切换
    resp, err = chat_completion_with_fallback(
        trace_id="promotion-gate",
        model=C.LONG_MEMORY_GATE_MODEL,
        messages=[{"role": "user",
                   "content": _GATE_PROMPT.format(transcript=transcript)}],
        temperature=0.0,
        max_tokens=4,
        timeout=20,
    )
    if err is not None:
        logger.warning("promotion gate LLM failed: %s: %s", type(err).__name__, err)
        return None
    text = (resp.choices[0].message.content or "").strip().lower()

    if text.startswith("no"):
        return False
    if text.startswith("yes"):
        return True
    # 非预期输出:fail-open,不丢事实
    logger.warning("promotion gate returned non yes/no (%r), treat as promote", text)
    return True


def promote_thread(thread_id: str, user_id: str, *,
                   session_id: Optional[str] = None,
                   embed: bool = True) -> list[int]:
    """增量升迁一个 thread 的新短期流水到长期记忆。返回写入的记忆 id 列表。

    以 promotion_watermark.last_seq 为界,只升迁 seq 更大的新事件;
    成功(含无新事实)后推进水位到这批事件的最大 seq。LLM 失败不推进,下次重试,
    但连续失败达到 MAX_PROMOTION_FAILURES 后强制推进水位(丢弃这批无法升迁的事件),
    避免永久性 LLM 故障导致 backlog 无限增长。
    长期库 (user_id, md5(content)) 唯一索引是最终去重兜底。
    """
    after_seq = short_term.get_watermark(thread_id)
    events = short_term.list_events_after(thread_id, after_seq)
    if not events:
        return []

    max_seq = max(int(ev.get("seq") or 0) for ev in events)
    transcript = _build_transcript(events)

    # 新增事件全是被白名单过滤的噪声:无内容可萃取,直接推进水位,不调 LLM
    if not transcript.strip():
        short_term.advance_watermark(thread_id, max_seq)
        return []

    # 守门员:先判断本轮是否值得升迁,避免为闲聊轮次付完整萃取 LLM 的开销
    gate = _should_promote(transcript)
    if gate is None:
        return _handle_failure(thread_id, after_seq, max_seq,
                               "gate LLM unavailable")
    if gate is False:
        short_term.advance_watermark(thread_id, max_seq)
        return []

    facts = _extract_facts(transcript)

    # 萃取失败:记录失败计数,达上限则强制推进水位
    if facts is None:
        return _handle_failure(thread_id, after_seq, max_seq,
                               "extract LLM failed")

    ids: list[int] = []
    for fact in facts:
        try:
            mid = long_term.add_memory(
                user_id=user_id, content=fact,
                memory_type="fact", thread_id=thread_id,
                meta={"promoted_from": thread_id, "session_id": session_id},
                embed=embed,
            )
            if mid is not None:
                ids.append(mid)
        except Exception as e:
            logger.warning("promotion write long-term failed: %s: %s",
                           type(e).__name__, e)

    # 即便没有萃取到事实也推进水位——这批事件已被处理,避免每轮重放
    short_term.advance_watermark(thread_id, max_seq)
    logger.info("promotion thread=%s events=%d (seq>%d) facts=%d written=%d -> %d",
                thread_id, len(events), after_seq, len(facts), len(ids), max_seq)
    return ids


def _handle_failure(thread_id: str, after_seq: int, max_seq: int,
                    reason: str) -> list[int]:
    """记录一次升迁失败;连续失败达上限则强制推进水位,避免 backlog 无限增长。"""
    fail_count = short_term.record_promotion_failure(
        thread_id, max_fail=MAX_PROMOTION_FAILURES)
    if fail_count >= MAX_PROMOTION_FAILURES:
        logger.error(
            "promotion thread=%s failed %d consecutive times (%s); "
            "force-advancing watermark %d -> %d to stop replay",
            thread_id, fail_count, reason, after_seq, max_seq)
        short_term.advance_watermark(thread_id, max_seq)
    else:
        logger.warning(
            "promotion thread=%s %s; watermark stays at %d "
            "(fail %d/%d, will retry)",
            thread_id, reason, after_seq, fail_count, MAX_PROMOTION_FAILURES)
    return []
