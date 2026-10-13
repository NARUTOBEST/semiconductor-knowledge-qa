# -*- coding: utf-8 -*-
"""混合重排(Qwen3 主路 + bge-m3 副路)单测:融合数学、单路降级、
双路皆败回退召回序、进程内双路接线(不依赖模型/网络)。"""
import urllib.error

import pytest

import config as C
import mcp_servers.retrieval.engine_api as E
from agent_reasoning.ReAct.support.tool_circuit import (
    CircuitOpenError, CircuitState,
)


def _pts(n=4):
    return [type("P", (), {"payload": {"content": f"d{i}"},
                           "score": 1.0 - i * 0.1})() for i in range(n)]


# ==================== _blend_score_sets 融合数学 ====================
def test_blend_two_paths_minmax_weighted():
    # 主路 [0.2,0.8]->归一[0,1]*0.6=[0,0.6];副路 [0.6,0.4]->[1,0]*0.4=[0.4,0]
    assert E._blend_score_sets([[0.2, 0.8], [0.6, 0.4]], [0.6, 0.4]) == \
        [0.4, 0.6]


def test_blend_single_path_passthrough():
    assert E._blend_score_sets([None, [0.6, 0.4]], [0.6, 0.4]) == [0.6, 0.4]


def test_blend_both_none_returns_none():
    assert E._blend_score_sets([None, None], [0.6, 0.4]) is None


def test_blend_constant_path_contributes_neutral_half():
    # 恒分路无区分度,贡献中性 0.5(不能拉爆另一路)
    # 主路恒 0.5 -> 0.5*0.6=0.3;副路 [0.1,0.9]->[0,1]*0.4=[0,0.4]
    assert E._blend_score_sets([[0.5, 0.5], [0.1, 0.9]], [0.6, 0.4]) == \
        [0.3, 0.7]


def test_blend_output_stays_in_unit_range():
    # 0~1 标尺是下游 RERANK_TAU 补录逻辑的硬约束
    out = E._blend_score_sets([[0.9, 0.1, 0.5], [0.3, 0.7, 0.5]], [0.6, 0.4])
    assert all(0.0 <= v <= 1.0 for v in out)


# ==================== HTTP 双路分支 ====================
def test_http_dual_path_blended_order(monkeypatch):
    pts = _pts()
    monkeypatch.setattr(C, "RERANK_API_BASE", "http://vllm:8001", raising=False)
    monkeypatch.setattr(C, "RERANK2_MODEL", "BAAI/bge-reranker-v2-m3",
                        raising=False)
    monkeypatch.setattr(C, "RERANK2_API_BASE", "http://vllm:8002",
                        raising=False)
    monkeypatch.setattr(C, "RERANK_W1", 0.6, raising=False)
    calls = []

    def fake_score_once(base, model, query, documents):
        calls.append((base, model))
        if "8001" in base:      # 主路:把 d2 排第一
            return [0.1, 0.2, 0.9, 0.2]
        return [0.9, 0.1, 0.8, 0.1]   # 副路:d0 第一

    monkeypatch.setattr(E, "_score_once", fake_score_once)
    out = E._rerank("q", pts, k=4)
    order = [p.payload["content"] for p, _ in out]
    # 主路归一 [0,0,8/9,1/9]*0.6 = [0,0,.533,.067]
    # 副路归一 [1,0,7/8,0]*0.4  = [.4,0,.35,0]
    assert order[0] == "d2"             # 主路强信号压过副路
    assert len(calls) == 2              # 两路都打了
    scores = [s for _, s in out]
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_http_single_path_failure_degrades_to_one_path(monkeypatch):
    pts = _pts()
    monkeypatch.setattr(C, "RERANK_API_BASE", "http://vllm:8001", raising=False)
    monkeypatch.setattr(C, "RERANK2_MODEL", "BAAI/bge-reranker-v2-m3",
                        raising=False)
    monkeypatch.setattr(C, "RERANK2_API_BASE", "http://vllm:8002",
                        raising=False)

    def fake_score_once(base, model, query, documents):
        if "8002" in base:              # 副路挂
            raise urllib.error.URLError("down")
        return [0.1, 0.9, 0.3, 0.2]     # 主路:d1 第一

    monkeypatch.setattr(E, "_score_once", fake_score_once)
    out = E._rerank("q", pts, k=4)
    assert [p.payload["content"] for p, _ in out][0] == "d1"


def test_http_both_paths_fail_falls_back_to_recall_order(monkeypatch):
    pts = _pts()
    monkeypatch.setattr(C, "RERANK_API_BASE", "http://vllm:8001", raising=False)
    monkeypatch.setattr(C, "RERANK2_MODEL", "BAAI/bge-reranker-v2-m3",
                        raising=False)
    monkeypatch.setattr(C, "RERANK2_API_BASE", "http://vllm:8002",
                        raising=False)
    monkeypatch.setattr(E, "_score_once",
                        lambda *a, **k: (_ for _ in ()).throw(
                            urllib.error.URLError("down")))
    out = E._rerank("q", pts, k=3)
    assert [p.payload["content"] for p, _ in out] == ["d0", "d1", "d2"]


def test_http_no_rerank2_model_keeps_single_path(monkeypatch):
    pts = _pts()
    monkeypatch.setattr(C, "RERANK_API_BASE", "http://vllm:8001", raising=False)
    monkeypatch.setattr(C, "RERANK2_MODEL", "", raising=False)
    called = {"n": 0}

    def fake_http(query, documents):
        called["n"] += 1
        return [0.1, 0.9, 0.3, 0.2]

    monkeypatch.setattr(E, "_rerank_http", fake_http)
    out = E._rerank("q", pts, k=4)
    assert called["n"] == 1
    assert [p.payload["content"] for p, _ in out][0] == "d1"


def test_http_rerank2_breaker_open_main_path_still_works(monkeypatch):
    # 副路熔断 OPEN(快速失败)不影响主路出分
    pts = _pts()
    monkeypatch.setattr(C, "RERANK_API_BASE", "http://vllm:8001", raising=False)
    monkeypatch.setattr(C, "RERANK2_MODEL", "BAAI/bge-reranker-v2-m3",
                        raising=False)
    monkeypatch.setattr(C, "RERANK2_API_BASE", "http://vllm:8002",
                        raising=False)
    def score_once(base, model, q, docs):
        if "8001" in base:
            return [0.1, 0.9, 0.3, 0.2]
        raise CircuitOpenError("rerank2 open")

    monkeypatch.setattr(E, "_score_once", score_once)
    out = E._rerank("q", pts, k=4)
    assert [p.payload["content"] for p, _ in out][0] == "d1"


# ==================== 进程内双路分支 ====================
def test_inproc_dual_path_blended(monkeypatch):
    pts = _pts()
    monkeypatch.setattr(C, "RERANK_API_BASE", "", raising=False)
    monkeypatch.setattr(C, "RERANK_MODEL", "Qwen/Qwen3-Reranker-4B",
                        raising=False)
    monkeypatch.setattr(C, "RERANK2_MODEL", "BAAI/bge-reranker-v2-m3",
                        raising=False)
    monkeypatch.setattr(C, "RERANK_W1", 0.6, raising=False)

    class FakeRe:
        def rerank(self, q, docs):
            return [0.1, 0.2, 0.9, 0.2]     # 主路 d2 第一

    class FakeRe2:
        def rerank(self, q, docs):
            return [0.9, 0.1, 0.8, 0.1]     # 副路 d0 第一

    old_re, old_re2 = E._re, E._re2
    E._re, E._re2 = FakeRe(), FakeRe2()
    try:
        out = E._rerank("q", pts, k=4)
        assert [p.payload["content"] for p, _ in out][0] == "d2"
        scores = [s for _, s in out]
        assert all(0.0 <= s <= 1.0 for s in scores)
    finally:
        E._re, E._re2 = old_re, old_re2


def test_inproc_sub_path_fails_single_path_used(monkeypatch):
    pts = _pts()
    monkeypatch.setattr(C, "RERANK_API_BASE", "", raising=False)
    monkeypatch.setattr(C, "RERANK2_MODEL", "BAAI/bge-reranker-v2-m3",
                        raising=False)

    class FakeRe:
        def rerank(self, q, docs):
            return [0.1, 0.9, 0.3, 0.2]

    class BadRe2:
        def rerank(self, q, docs):
            raise RuntimeError("oom")

    old_re, old_re2 = E._re, E._re2
    E._re, E._re2 = FakeRe(), BadRe2()
    try:
        out = E._rerank("q", pts, k=4)
        assert [p.payload["content"] for p, _ in out][0] == "d1"
    finally:
        E._re, E._re2 = old_re, old_re2


def test_inproc_both_fail_recall_order(monkeypatch):
    pts = _pts()
    monkeypatch.setattr(C, "RERANK_API_BASE", "", raising=False)
    monkeypatch.setattr(C, "RERANK2_MODEL", "BAAI/bge-reranker-v2-m3",
                        raising=False)

    class Bad:
        def rerank(self, q, docs):
            raise RuntimeError("down")

    old_re, old_re2 = E._re, E._re2
    E._re, E._re2 = Bad(), Bad()
    try:
        out = E._rerank("q", pts, k=3)
        assert [p.payload["content"] for p, _ in out] == ["d0", "d1", "d2"]
    finally:
        E._re, E._re2 = old_re, old_re2


def test_inproc_no_rerank2_single_path(monkeypatch):
    pts = _pts()
    monkeypatch.setattr(C, "RERANK_API_BASE", "", raising=False)
    monkeypatch.setattr(C, "RERANK2_MODEL", "", raising=False)
    calls = {"n": 0}

    class FakeRe:
        def rerank(self, q, docs):
            calls["n"] += 1
            return [0.1, 0.9, 0.3, 0.2]

    old_re, old_re2 = E._re, E._re2
    E._re, E._re2 = FakeRe(), None
    try:
        out = E._rerank("q", pts, k=4)
        assert calls["n"] == 1
        assert [p.payload["content"] for p, _ in out][0] == "d1"
    finally:
        E._re, E._re2 = old_re, old_re2
