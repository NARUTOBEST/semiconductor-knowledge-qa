# -*- coding: utf-8 -*-
"""召回网关单测(mock 向量检索/重排,不连真实 PG/模型)。

覆盖:重排降级、vector_search 异常返回空、token 裁剪、k 截断、format 输出。
"""
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memories.storage.long import recall_gateway  # noqa: E402

# recall_gateway 内部 `from .long_term import long_term` 拿到单例实例
long_term = recall_gateway.long_term


def _candidates(n=3):
    return [{"id": i, "content": f"记忆内容{i} " * 10, "memory_type": "fact",
             "score": 0.9 - i * 0.1, "meta": {}} for i in range(n)]


def test_recall_returns_ranked_and_truncated(monkeypatch):
    cands = _candidates(5)
    monkeypatch.setattr(long_term, "vector_search",
                        lambda *a, **k: cands)
    # 重排按 content 长度给分(稳定可预测):越短分越高,验证确实重排了
    def _fake_rerank(query, docs):
        return [float(len(d)) for d in docs]
    fake_embed = types.SimpleNamespace(
        get_reranker=lambda: types.SimpleNamespace(rerank=_fake_rerank))
    monkeypatch.setitem(sys.modules, "embed", fake_embed)

    out = recall_gateway.recall_memories("alice", "问题", k=3, max_tokens=10_000)
    assert len(out) <= 3
    # 重排后第一条应是最短 content(长度最小)
    assert out[0]["id"] == 0
    assert "rerank_score" in out[0]


def test_rerank_failure_falls_back_to_vector_order(monkeypatch):
    cands = _candidates(3)
    monkeypatch.setattr(long_term, "vector_search", lambda *a, **k: cands)

    # embed 模块导入即抛异常 -> 走降级,保持向量分顺序,不报错
    import builtins
    real_import = builtins.__import__

    def _boom(name, *a, **k):
        if name == "embed":
            raise RuntimeError("reranker unavailable")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _boom)

    out = recall_gateway.recall_memories("alice", "问题", k=5, max_tokens=10_000)
    assert [m["id"] for m in out] == [0, 1, 2]  # 原始顺序
    assert "rerank_score" not in out[0]


def test_vector_search_exception_returns_empty(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("pg down")
    monkeypatch.setattr(long_term, "vector_search", _boom)
    assert recall_gateway.recall_memories("alice", "x") == []


def test_token_budget_truncates(monkeypatch):
    cands = _candidates(10)
    monkeypatch.setattr(long_term, "vector_search", lambda *a, **k: cands)
    # max_tokens 很小,只能容纳第一条
    out = recall_gateway.recall_memories("alice", "问题", k=10, max_tokens=5)
    assert len(out) == 1
    assert out[0]["id"] == 0


def test_empty_candidates(monkeypatch):
    monkeypatch.setattr(long_term, "vector_search", lambda *a, **k: [])
    assert recall_gateway.recall_memories("alice", "问题") == []


def test_format_memories_for_prompt():
    assert recall_gateway.format_memories_for_prompt([]) == ""
    block = recall_gateway.format_memories_for_prompt(
        [{"content": "用户用 Win11"}, {"content": "用户做芯片验证"}])
    assert "1. 用户用 Win11" in block
    assert "2. 用户做芯片验证" in block
    assert "长期记忆" in block


def test_estimate_tokens_cjk_and_words():
    # cjk 字数 + 词数*1.3(无空格中文整串被 split 算 1 词,取两者上界)
    assert recall_gateway._estimate_tokens("你好世界") == int(4 + 1 * 1.3)
    assert recall_gateway._estimate_tokens("hello world") == int(2 * 1.3)
