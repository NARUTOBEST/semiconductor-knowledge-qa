# -*- coding: utf-8 -*-
"""长期记忆【抽取】:会话结束后后台用 LLM 抽取用户偏好,写入 PG。

旁路 daemon 线程(仿 summarize.schedule_pregeneration):应答结束后触发,不阻塞、不影响
聊天。LLM 抽取偏好事实 -> BGE-m3 向量化 -> long_term.upsert_memory(结构化/语义去重)
-> 汇总刷新 user_profile。任一步(PG / 嵌入 / LLM)失败都静默降级。

只抽【长期稳定】信息:语言/语气/格式偏好、身份角色、专业水平、关注领域、明确约束;
不抽"ALD 是什么"这类一次性问答事实。
"""
from __future__ import annotations

import json
import logging
import re
import threading
from typing import Optional

import config as C

from ...storage._llm import chat_completion_with_fallback
from ...storage.long import long_term, embed_texts

logger = logging.getLogger("agent")

# Req13 指令注入模式:抽取到的"偏好"若实为对 AI 的指令/越狱(而非用户事实),丢弃不持久化。
_INJECTION_PATTERNS = re.compile(
    r"(忽略(以上|之前|上述).{0,6}(规则|指令|提示|要求|内容)|"
    r"ignore\s+(all\s+)?(previous|above|prior)|"
    r"disregard\s+(previous|above)|"
    r"你现在是|从现在起你是|扮演|进入(开发者|管理员|越狱)模式|"
    r"system\s*prompt|系统提示词|开发者指令|"
    r"不要遵守|无需遵守|忘掉(你的|之前|所有)(规则|设定|限制))",
    re.IGNORECASE,
)


def _is_injection(text: str) -> bool:
    """抽取内容是否为指令注入/越狱文本(而非用户偏好事实)。"""
    if not getattr(C, "MEMORY_INJECTION_GUARD", True):
        return False
    return bool(_INJECTION_PATTERNS.search(text or ""))

_ALLOWED_CATEGORIES = {
    "language", "role", "expertise", "topic_interest",
    "tone", "format", "constraint", "fact",
}
# 归入"稳定设置"的类别(写 display_prefs)
_PREF_CATEGORIES = {"language", "tone", "format", "constraint"}
# 归入"关注领域/身份"的类别(写 top_interests)
_INTEREST_CATEGORIES = {"role", "expertise", "topic_interest"}

_EXTRACTION_SYS = (
    "你是用户长期偏好抽取器。从下面这轮【用户提问 + 助手回答】中,抽取关于该用户的"
    "【长期稳定】信息,用于跨会话个性化。只抽取:回答语言/语气/格式偏好、用户身份与角色、"
    "专业水平、长期关注的设备/工艺/领域、用户明确提出的长期约束。"
    "不要抽取一次性的问答事实(如某个术语的定义、某次具体故障)。\n"
    "输出严格的 JSON 数组(不要输出任何其他文字、不要 markdown 代码块),每个元素为:\n"
    '{"category": "language|role|expertise|topic_interest|tone|format|constraint|fact",'
    ' "key": "稳定槽位名(如 response_language/focus_domain,没有可留 null)",'
    ' "content": "一句面向 AI 的、可直接用于个性化的中文偏好陈述",'
    ' "importance": 0.0到1.0的数字}\n'
    "若本轮没有可长期记住的用户信息,输出 []。"
)


def schedule_extraction(*, username: str, thread_id: str = None,
                        user_message: str = "", assistant_message: str = "") -> None:
    """[兼容保留] 提交 daemon 线程做偏好抽取,异常静默。

    长期记忆写入已收口到 memory-loop 沉淀节点(consolidate_turn);此 daemon 包装
    仅为可能的旧调用方保留,新链路不要使用。
    """
    if not getattr(C, "LONG_MEM_ENABLED", True) or not username:
        return
    threading.Thread(
        target=_safe_consolidate,
        args=(username, thread_id, user_message, assistant_message),
        daemon=True, name="long-mem-extract",
    ).start()


def _safe_consolidate(username, thread_id, user_message, assistant_message) -> None:
    try:
        consolidate_turn(username, thread_id,
                         user_message=user_message, assistant_message=assistant_message)
    except Exception as e:  # noqa: BLE001  旁路任务,任何异常都不影响主流程
        logger.info("long-memory extraction skipped: %s: %s",
                    type(e).__name__, str(e)[:160])


def _format_dialogue(user_message: str, assistant_message: str) -> str:
    parts = []
    if user_message:
        parts.append("用户提问:" + str(user_message)[:2000])
    if assistant_message:
        parts.append("助手回答:" + str(assistant_message)[:2000])
    return "\n".join(parts)


def consolidate_turn(username: str, thread_id: str = None, *,
                     user_message: str = "", assistant_message: str = "",
                     deadline: Optional[float] = None) -> dict:
    """升迁门(同步):抽取本轮对话中的长期偏好,重要性达标的升迁写入 PG。

    由 memory-loop 沉淀节点调用(节点负责 retry/熔断/降级)。
    deadline:整链时间预算(monotonic 时间戳),透传给 LLM 调用做硬帽。
    - LLM 调用/解析【失败】时抛异常(调用方据此 retry→degrade);
    - LLM 成功但无达标偏好时返回 {"items": n, "promoted": 0}(不抛);
    - 关闭/匿名/无内容返回 {"promoted": 0, "skipped": ...}(不抛、不调 LLM)。

    升迁门控:仅 importance ≥ MEM_PROMOTE_IMPORTANCE 的条目写长期表(事实性/可复述性
    由抽取 prompt 保证——只抽长期稳定信息)。返回 {"items": 抽取条数, "promoted": 升迁条数}。
    """
    if not getattr(C, "LONG_MEM_ENABLED", True) or not username:
        return {"items": 0, "promoted": 0, "skipped": "disabled"}
    dialogue = _format_dialogue(user_message, assistant_message)
    if not dialogue:
        return {"items": 0, "promoted": 0, "skipped": "empty"}

    items = _call_extract_llm(dialogue, deadline=deadline)
    if items is None:
        raise RuntimeError("promotion-gate LLM 调用失败")  # 调用方 retry/degrade

    threshold = float(getattr(C, "MEM_PROMOTE_IMPORTANCE", 0.6))
    promoted_items = [it for it in items
                      if _clamp_float(it.get("importance"), 0.5) >= threshold]
    if not promoted_items:
        return {"items": len(items), "promoted": 0}

    _upsert_items(username, thread_id, promoted_items)
    logger.info("long-memory promoted %d/%d prefs for user=%s",
                len(promoted_items), len(items), username)
    return {"items": len(items), "promoted": len(promoted_items)}


def _upsert_items(username: str, thread_id, items: list[dict]) -> None:
    """把抽取条目向量化并 upsert 到长期表 + 汇总画像。

    PG 不可用(ping 失败)时整批进本地 spool,恢复后由记忆兜底维护重放,
    升迁数据不因 PG 宕机丢失(见 storage/long/spool.py)。
    """
    from ...storage.long import pg as _pg
    from ...storage.long import spool as _spool
    if not _pg.ping_pg():
        _spool.append_record(username, thread_id, items)
        logger.info("long-memory PG 不可用,%d 条升迁暂存本地 spool", len(items))
        return
    contents = [it["content"] for it in items]
    vectors = embed_texts(contents)  # 失败返回 None;向量缺失则只结构化存储
    display_prefs: dict = {}
    interests: list[str] = []
    for i, it in enumerate(items):
        category = it.get("category")
        if category not in _ALLOWED_CATEGORIES:
            category = "fact"
        content = (it.get("content") or "").strip()
        if not content:
            continue
        # Req13:写入侧指令过滤——指令性/越狱内容不是用户偏好,丢弃不持久化。
        if _is_injection(content):
            logger.info("long-memory injection content filtered out: %s", content[:60])
            continue
        key = (it.get("key") or "").strip() or None
        importance = _clamp_float(it.get("importance"), 0.5)
        embedding = vectors[i] if vectors and i < len(vectors) else None
        long_term.upsert_memory(
            username, category=category, key=key, content=content,
            embedding=embedding, importance=importance, source_thread=thread_id)
        if category in _PREF_CATEGORIES and key:
            display_prefs[key] = content
        elif category in _INTEREST_CATEGORIES:
            interests.append(content)
    long_term.upsert_profile(
        username,
        display_prefs=display_prefs or None,
        top_interests=interests or None,
    )


def replay_spooled(*, limit: int = None, time_budget_s: float = None) -> dict:
    """把本地 spool 里因 PG 宕机暂存的升迁批重放进 PG(有界、幂等)。

    由记忆兜底维护(resilience)在每轮收尾调用;PG 仍不可用时 0 成本跳过。
    """
    from ...storage.long import pg as _pg
    from ...storage.long import spool as _spool
    if not getattr(C, "LONG_MEM_ENABLED", True):
        return {"skipped": "long_mem_disabled"}
    if not _pg.ping_pg():
        return {"skipped": "pg_down"}
    return _spool.drain(lambda rec: (
        _upsert_items(rec["username"], rec.get("thread_id"),
                      [it for it in (rec.get("items") or [])
                       if isinstance(it, dict)]) or True),
        max_items=limit, time_budget_s=time_budget_s)


def _call_extract_llm(dialogue: str, deadline: Optional[float] = None):
    """调 LLM 抽取并解析为 list[dict];LLM/解析失败返回 None,成功(含空)返回 list。

    retries 显式取 1:重试应对瞬时抖动,一次足够;服务持续挂死由整链预算
    (deadline)+跨请求熔断兜底,不做模型内多次串行重试(旧默认 3 次重试会把
    一次尝试拖到 ~52s)。
    """
    resp, err = chat_completion_with_fallback(
        model=getattr(C, "MODEL_LIGHT", None) or "",
        messages=[
            {"role": "system", "content": _EXTRACTION_SYS},
            {"role": "user", "content": dialogue},
        ],
        temperature=0.0,
        max_tokens=600,
        timeout=float(getattr(C, "MEM_GATE_LLM_TIMEOUT", 3.0)),
        retries=1,
        deadline=deadline,
    )
    if err is not None or resp is None:
        return None
    try:
        text = resp.choices[0].message.content or ""
    except Exception:
        return None
    return _parse_json_array(text)


def _parse_json_array(text: str) -> list[dict]:
    """从 LLM 输出里容错解析出 JSON 数组(去 markdown 围栏/截取首个数组)。"""
    if not text:
        return []
    s = text.strip()
    # 去掉 ```json ... ``` 围栏,取围栏内内容
    if "```" in s:
        import re
        m = re.search(r"```(?:json)?\s*(.*?)```", s, re.S)
        if m:
            s = m.group(1).strip()
    try:
        data = json.loads(s)
    except Exception:
        # 兜底:截取第一个 [ 到最后一个 ]
        i, j = s.find("["), s.rfind("]")
        if i < 0 or j <= i:
            return []
        try:
            data = json.loads(s[i:j + 1])
        except Exception:
            return []
    if isinstance(data, dict):  # 模型偶尔返回 {"preferences": [...]}
        for v in data.values():
            if isinstance(v, list):
                data = v
                break
    if not isinstance(data, list):
        return []
    return [d for d in data if isinstance(d, dict)]


def _clamp_float(v, default: float) -> float:
    try:
        f = float(v)
        return max(0.0, min(1.0, f))
    except Exception:
        return default
