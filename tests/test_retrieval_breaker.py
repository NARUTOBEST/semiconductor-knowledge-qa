# -*- coding: utf-8 -*-
"""检索服务外呼三状态熔断单测:驱动层(_call_with_backoff)状态机行为 +
translate/rerank 接线降级(不依赖模型/网络)。"""
import threading
import time
import urllib.error

import pytest

import config as C
import mcp_servers.retrieval.engine_api as E
from agent_reasoning.ReAct.support.tool_circuit import (
    CircuitBreaker, CircuitOpenError, CircuitState,
)


def _brk(**kw):
    d = dict(failure_threshold=2, cooldown_seconds=30.0)
    d.update(kw)
    return CircuitBreaker("t", **d)


# ==================== 驱动层:CLOSED 指数退避重试 ====================
def test_closed_retries_with_backoff_then_succeeds(monkeypatch):
    brk = _brk(failure_threshold=5)
    monkeypatch.setattr(C, "RETRIEVAL_BREAKER_RETRY_ATTEMPTS", 3, raising=False)
    sleeps = []
    monkeypatch.setattr(E.time, "sleep", lambda s: sleeps.append(s))
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise urllib.error.URLError("timeout")
        return "ok"

    assert E._call_with_backoff(brk, flaky) == "ok"
    assert calls["n"] == 3
    assert sleeps == [0.5, 1.0]              # 指数退避 0.5*2^n
    assert brk.state == CircuitState.CLOSED  # 成功不计数


def test_closed_backoff_capped(monkeypatch):
    brk = _brk(failure_threshold=5)
    monkeypatch.setattr(C, "RETRIEVAL_BREAKER_RETRY_ATTEMPTS", 4, raising=False)
    monkeypatch.setattr(C, "RETRIEVAL_BREAKER_BACKOFF_BASE_S", 2.0, raising=False)
    monkeypatch.setattr(C, "RETRIEVAL_BREAKER_BACKOFF_MAX_S", 3.0, raising=False)
    seen = []
    monkeypatch.setattr(E.time, "sleep", lambda s: seen.append(s))
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 4:
            raise urllib.error.URLError("x")
        return 1

    E._call_with_backoff(brk, flaky)
    assert seen == [2.0, 3.0, 3.0]           # 2^n 封顶 BACKOFF_MAX


def test_retries_exhausted_records_failure_and_trips_open(monkeypatch):
    brk = _brk(failure_threshold=2)
    monkeypatch.setattr(E.time, "sleep", lambda s: None)

    def down():
        raise urllib.error.URLError("conn refused")

    with pytest.raises(urllib.error.URLError):
        E._call_with_backoff(brk, down)      # 重试耗尽 -> 记失败 1
    assert brk.state == CircuitState.CLOSED
    with pytest.raises(urllib.error.URLError):
        E._call_with_backoff(brk, down)      # 失败 2 -> 达阈值
    assert brk.state == CircuitState.OPEN


# ==================== OPEN:快速失败(不触外呼) ====================
def test_open_fails_fast_without_calling(monkeypatch):
    brk = _brk(failure_threshold=1)
    monkeypatch.setattr(E.time, "sleep", lambda s: None)

    def down():
        raise urllib.error.URLError("x")

    with pytest.raises(urllib.error.URLError):
        E._call_with_backoff(brk, down)
    assert brk.state == CircuitState.OPEN
    called = {"n": 0}

    def fn():
        called["n"] += 1

    with pytest.raises(CircuitOpenError):
        E._call_with_backoff(brk, fn)        # OPEN 快速失败
    assert called["n"] == 0


# ==================== HALF_OPEN:单探测恢复 ====================
def test_half_open_probe_success_recovers(monkeypatch):
    brk = _brk(failure_threshold=1, cooldown_seconds=0.05)
    monkeypatch.setattr(E.time, "sleep", lambda s: None)

    def down():
        raise urllib.error.URLError("x")

    with pytest.raises(urllib.error.URLError):
        E._call_with_backoff(brk, down)
    assert brk.state == CircuitState.OPEN
    brk._opened_at -= 1.0                    # 回拨冷却起点 -> 下一调用转半开
    monkeypatch.setattr(C, "RETRIEVAL_BREAKER_RETRY_ATTEMPTS", 3, raising=False)
    probe_calls = {"n": 0}

    def probe():
        probe_calls["n"] += 1
        raise urllib.error.URLError("still down")

    with pytest.raises(urllib.error.URLError):
        E._call_with_backoff(brk, probe)
    assert probe_calls["n"] == 1             # 半开只打一发(不重试)
    assert brk.state == CircuitState.OPEN    # 探测失败 -> 重新打开重新计时

    brk._opened_at -= 1.0
    ok = {"n": 0}

    def good():
        ok["n"] += 1
        return "up"

    assert E._call_with_backoff(brk, good) == "up"   # 冷却已过,探测成功
    assert ok["n"] == 1
    assert brk.state == CircuitState.CLOSED


def test_half_open_second_call_fails_fast(monkeypatch):
    brk = _brk(failure_threshold=1, cooldown_seconds=0.05)
    monkeypatch.setattr(E.time, "sleep", lambda s: None)

    def down():
        raise urllib.error.URLError("x")

    with pytest.raises(urllib.error.URLError):
        E._call_with_backoff(brk, down)
    brk._opened_at -= 1.0                    # 回拨冷却起点 -> 下一调用转半开
    slow = {"n": 0}
    entered = threading.Event()              # 探测已过 before_call(占住探测槽)
    release = threading.Event()              # 主线程断言完再放探测完成

    def slow_probe():
        slow["n"] += 1
        entered.set()
        release.wait(timeout=2.0)
        return "up"

    # 并发:第一个探测在飞,第二个请求半开期必须快速失败
    t = threading.Thread(target=lambda: E._call_with_backoff(brk, slow_probe))
    t.start()
    assert entered.wait(timeout=2.0)
    with pytest.raises(CircuitOpenError):
        E._call_with_backoff(brk, lambda: "x")
    release.set()
    t.join()
    assert slow["n"] == 1
    assert brk.state == CircuitState.CLOSED


# ==================== fatal(4xx):不重试、CLOSED 下不计数 ====================
def test_fatal_4xx_no_retry_no_count(monkeypatch):
    brk = _brk(failure_threshold=1)
    monkeypatch.setattr(E.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def auth_fail():
        calls["n"] += 1
        raise urllib.error.HTTPError("u", 401, "unauthorized", None, None)

    for _ in range(3):
        with pytest.raises(urllib.error.HTTPError):
            E._call_with_backoff(brk, auth_fail)
    assert calls["n"] == 3                   # 每次只打一发(不重试)
    assert brk.state == CircuitState.CLOSED  # fatal 不计入熔断


def test_429_counts_as_retryable(monkeypatch):
    brk = _brk(failure_threshold=1)
    monkeypatch.setattr(E.time, "sleep", lambda s: None)

    def rate_limited():
        raise urllib.error.HTTPError("u", 429, "too many", None, None)

    with pytest.raises(urllib.error.HTTPError):
        E._call_with_backoff(brk, rate_limited)
    assert brk.state == CircuitState.OPEN    # 限流算服务故障


# ==================== 接线:translate / rerank 降级 ====================
def test_translate_open_fails_fast_silent(monkeypatch):
    brk = _brk(failure_threshold=1)
    monkeypatch.setattr(E.time, "sleep", lambda s: None)

    def down():
        raise urllib.error.URLError("x")

    with pytest.raises(urllib.error.URLError):
        E._call_with_backoff(brk, down)
    old = E._translate_brk
    E._translate_brk = brk
    try:
        E._tr_cache.clear()
        out = E._translate_query("矢量传感器的重复精度是多少?")
        assert out == ""                        # 熔断 OPEN:静默单路
    finally:
        E._translate_brk = old
        E._tr_cache.clear()


def test_translate_success_and_recovery(monkeypatch):
    E._tr_cache.clear()
    old = E._translate_brk
    E._translate_brk = _brk(failure_threshold=5)
    try:
        class FakeResp:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return b'{"choices":[{"message":{"content":"repeatability of vector sensor"}}]}'

        monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: FakeResp())
        out = E._translate_query("矢量传感器的重复精度是多少?")
        assert out == "repeatability of vector sensor"
        assert E._translate_brk.state == CircuitState.CLOSED
    finally:
        E._translate_brk = old
        E._tr_cache.clear()


def test_rerank_open_falls_back_to_recall_order(monkeypatch):
    pts = [type("P", (), {"payload": {"content": f"d{i}"}, "score": 1.0 - i * 0.1})()
           for i in range(4)]
    old = E._rerank_brk
    brk = _brk(failure_threshold=1)
    monkeypatch.setattr(E.time, "sleep", lambda s: None)

    def down():
        raise urllib.error.URLError("x")

    with pytest.raises(urllib.error.URLError):
        E._call_with_backoff(brk, down)
    E._rerank_brk = brk
    try:
        monkeypatch.setattr(C, "RERANK_API_BASE", "http://vllm:8001", raising=False)
        out = E._rerank("q", pts, k=3)
        assert [p.payload["content"] for p, _ in out] == \
            [f"d{i}" for i in range(3)]          # 按召回分序降级
    finally:
        E._rerank_brk = old


def test_rerank_http_down_retries_then_recall_order(monkeypatch):
    pts = [type("P", (), {"payload": {"content": f"d{i}"}, "score": 1.0 - i * 0.1})()
           for i in range(4)]
    old = E._rerank_brk
    E._rerank_brk = _brk(failure_threshold=5)
    try:
        monkeypatch.setattr(C, "RERANK_API_BASE", "http://vllm:8001", raising=False)

        def http_fail(*a, **k):
            raise urllib.error.URLError("vllm down")

        monkeypatch.setattr(E, "_rerank_http", http_fail)
        out = E._rerank("q", pts, k=3)           # 重试耗尽 -> 回退召回序
        assert len(out) == 3
        assert E._rerank_brk.state == CircuitState.CLOSED  # 阈值 5 未达
    finally:
        E._rerank_brk = old


def test_breaker_states_snapshot():
    st = E.breaker_states()
    assert set(st) == {"translate", "rerank", "rerank2"}
    assert st["translate"] in (CircuitState.CLOSED, CircuitState.OPEN,
                               CircuitState.HALF_OPEN)
