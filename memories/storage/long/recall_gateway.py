# -*- coding: utf-8 -*-
"""记忆召回网关。

统一入口(对照 memory-system-design):
  结构化过滤(按 user_id / memory_type) -> 向量检索 -> 重排 -> token 裁剪 -> 返回

- 向量检索走 long_term.vector_search(BGE-m3,pgvector 余弦)。
- 重排复用 RAG 的 BGE-reranker(若可用);失败则直接用向量分数。
- token 裁剪按字符估算(中文保守 1 token ≈ 1.5 字符),不精确但简单稳定,
  避免无 tiktoken 时崩溃;目标不超过 config.LONG_MEMORY_RECALL_MAX_TOKENS。
- 返回结构化 list[dict],由调用方决定如何拼进 messages;format_memories_for_prompt
  提供现成的文本块。
"""
import logging
from typing import Any, Optional

import sys, os
# 本文件位于 memories/storage/long/;项目根在上三级
_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
for _p in (os.path.join(_ROOT, "RAG"), os.path.join(_ROOT, "config"), _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import config as C  # noqa: E402

from .long_term import long_term  # noqa: E402

logger = logging.getLogger("agent")


def _estimate_tokens(text: str) -> int:
    # 粗略:中文/日文等 1 字 ~1 token;英文按空白分词。取两者上界偏保守。
    cjk = sum(1 for ch in text if '一' <= ch <= '鿿')
    words = len(text.split())
    return int(cjk + words * 1.3)


def _rerank(query: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """用 BGE-reranker 重排;不可用或失败时保持向量分数顺序。"""
    if not candidates:
        return candidates
    try:
        import embed  # type: ignore
        reranker = embed.get_reranker()
        docs = [str(c.get("content", "")) for c in candidates]
        scores = reranker.rerank(query, docs)
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)
        candidates.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
    except Exception as e:
        # 重排是增强项,失败不阻断召回
        logger.warning("recall rerank unavailable, using vector score: %s: %s",
                       type(e).__name__, e)
    return candidates


def recall_memories(user_id: str, query: str, *,
                    k: Optional[int] = None,
                    memory_types: Optional[list[str]] = None,
                    max_tokens: Optional[int] = None) -> list[dict[str, Any]]:
    """召回与当前 query 相关的长期记忆,经重排和 token 裁剪后返回。

    返回字段:id / memory_type / content / score / rerank_score(可选) / meta。
    """
    k = k or C.LONG_MEMORY_RECALL_TOP_K
    max_tokens = max_tokens or C.LONG_MEMORY_RECALL_MAX_TOKENS

    # 1. 结构化过滤 + 向量检索(召回 k*3 供重排)
    try:
        candidates = long_term.vector_search(
            user_id, query, k=max(k * 3, k), memory_types=memory_types)
    except Exception as e:
        logger.warning("recall long-term vector_search failed: %s: %s",
                       type(e).__name__, e)
        return []

    # 2. 重排
    ranked = _rerank(query, candidates)

    # 3. token 裁剪:按相关性从高到低累加,超预算停止
    selected: list[dict[str, Any]] = []
    used = 0
    for item in ranked:
        content = str(item.get("content", ""))
        cost = _estimate_tokens(content)
        if selected and used + cost > max_tokens:
            break
        selected.append(item)
        used += cost
        if len(selected) >= k:
            break
    return selected


def format_memories_for_prompt(memories: list[dict[str, Any]]) -> str:
    """把召回结果格式化为可直接插入 system prompt 的文本块。"""
    if not memories:
        return ""
    lines = ["以下是关于该用户的长期记忆,供回答时参考(不要向用户复述这些记忆本身):"]
    for i, m in enumerate(memories, 1):
        lines.append(f"{i}. {m.get('content', '')}")
    return "\n".join(lines)
