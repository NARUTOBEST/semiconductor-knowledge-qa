# -*- coding: utf-8 -*-
"""熔断器单测:状态机 + 韧性中间件(call_with_resilience)集成。

熔断器已从 tools/circuit_breaker.py 迁到 agent_reasoning.ReAct.support.tool_circuit,
由韧性中间件 tool_resilience 驱动(tools.dispatch 现在是纯路由,不再含熔断/重试)。
"""
import time

import httpx
import pytest

import config as C
from agent_reasoning.ReAct.support.tool_circuit import (
    CircuitBreaker, CircuitBreakerRegistry, CircuitOpenError, CircuitState,
)
from agent_reasoning.ReAct.support.tool_resilience import call_with_resilience
from agent_reasoning.ReAct.support.tool_errors import Kind
from tools.base import ErrorType, ToolSpec, TruncatePolicy


# ==================== 状态机 ====================
def test_closed_to_open_after_threshold():
    changes = []
    cb = CircuitBreaker("t", failure_threshold=3, cooldown_seconds=60,
                        on_state_change=lambda n, o, new: changes.append((o, new)))
    for _ in range(3):
        cb.on_failure(ErrorType.RETRYABLE)
    assert cb.state == CircuitState.OPEN
    assert changes[-1] == (CircuitState.CLOSED, CircuitState.OPEN)


def test_fatal_does_not_count():
    cb = CircuitBreaker("t", failure_threshold=2, cooldown_seconds=60)
    for _ in range(5):
        cb.on_failure(ErrorType.FATAL)
    assert cb.state == CircuitState.CLOSED


def test_success_resets_count_when_closed():
    cb = CircuitBreaker("t", failure_threshold=3, cooldown_seconds=60)
    cb.on_failure(ErrorType.TIMEOUT)
    cb.on_failure(ErrorType.TIMEOUT)
    cb.on_success()
    cb.on_failure(ErrorType.TIMEOUT)
    cb.on_failure(ErrorType.TIMEOUT)
    assert cb.state == CircuitState.CLOSED  # 计数被成功复位


def test_before_call_blocked_when_open():
    cb = CircuitBreaker("t", failure_threshold=1, cooldown_seconds=60)
    cb.on_failure(ErrorType.RETRYABLE)
    assert cb.state == CircuitState.OPEN
    with pytest.raises(CircuitOpenError):
        cb.before_call()


def test_open_to_half_open_after_cooldown():
    cb = CircuitBreaker("t", failure_threshold=1, cooldown_seconds=0.1)
    cb.on_failure(ErrorType.RETRYABLE)
    assert cb.state == CircuitState.OPEN
    time.sleep(0.12)
    # 冷却到期后 state 属性自动转 HALF_OPEN
    assert cb.state == CircuitState.HALF_OPEN


def test_half_open_probe_success_closes():
    cb = CircuitBreaker("t", failure_threshold=1, cooldown_seconds=0.1)
    cb.on_failure(ErrorType.RETRYABLE)
    time.sleep(0.12)
    assert cb.state == CircuitState.HALF_OPEN
    cb.before_call()          # 放行试探
    cb.on_success()
    assert cb.state == CircuitState.CLOSED


def test_half_open_probe_failure_reopens():
    cb = CircuitBreaker("t", failure_threshold=1, cooldown_seconds=60)
    cb.on_failure(ErrorType.RETRYABLE)
    # 手工把状态切到 HALF_OPEN 模拟冷却后
    with cb._lock:
        cb._set_state_locked(CircuitState.HALF_OPEN)
    cb.before_call()
    cb.on_failure(ErrorType.RETRYABLE)
    assert cb.state == CircuitState.OPEN


def test_half_open_probe_fatal_failure_reopens_and_releases_slot():
    # 半开探测即使返回 FATAL(如 key 失效 401),也必须释放探测槽并转回 OPEN,
    # 否则会永久卡在 HALF_OPEN + inflight,后续请求全部被拒(不重启无法恢复)。
    cb = CircuitBreaker("t", failure_threshold=1, cooldown_seconds=60)
    with cb._lock:
        cb._set_state_locked(CircuitState.HALF_OPEN)
    cb.before_call()                       # 放行探测(inflight=True)
    cb.on_failure(ErrorType.FATAL)         # 探测返回 fatal
    assert cb.state == CircuitState.OPEN   # 转回 OPEN(而非卡死 HALF_OPEN)
    # OPEN 下冷却未到 -> before_call 应正常抛 CircuitOpenError(而不是因 inflight 永久挡死)
    with pytest.raises(CircuitOpenError):
        cb.before_call()


def test_half_open_blocks_extra_probes():
    cb = CircuitBreaker("t", failure_threshold=1, cooldown_seconds=60)
    with cb._lock:
        cb._set_state_locked(CircuitState.HALF_OPEN)
    cb.before_call()  # 第一个放行
    with pytest.raises(CircuitOpenError):
        cb.before_call()  # 第二个被挡


def test_reset_closes():
    cb = CircuitBreaker("t", failure_threshold=1, cooldown_seconds=60)
    cb.on_failure(ErrorType.RETRYABLE)
    assert cb.state == CircuitState.OPEN
    cb.reset()
    assert cb.state == CircuitState.CLOSED


def test_registry_caches_breakers():
    reg = CircuitBreakerRegistry(failure_threshold=2, cooldown_seconds=10)
    assert reg.get("a") is reg.get("a")
    assert reg.get("a") is not reg.get("b")


def test_registry_reset_all():
    reg = CircuitBreakerRegistry(failure_threshold=1, cooldown_seconds=60)
    b = reg.get("x")
    b.on_failure(ErrorType.RETRYABLE)
    assert b.state == CircuitState.OPEN
    reg.reset_all()
    assert b.state == CircuitState.CLOSED


# ==================== 韧性中间件集成 ====================
@pytest.fixture
def breakable_tool(isolated_registry):
    """注册一个默认抛超时的临时工具(handler 可在测试中替换),并复位全局熔断器。"""
    from agent_reasoning.ReAct.support import tool_circuit

    def boom():
        raise httpx.ReadTimeout("read timed out")

    spec = ToolSpec(
        name="boom_tool",
        description="t",
        category="test",
        parameters={"type": "object", "properties": {}, "required": []},
        handler=boom,
        enabled=True,
        retry_times=0,
        truncate=TruncatePolicy(),
    )
    isolated_registry.register(spec)
    tool_circuit._breakers.reset_all()
    yield isolated_registry.get("boom_tool")
    tool_circuit._breakers.reset_all()


def _invoke_dispatch(name, args):
    """中间件的 invoke 回调:走真实 dispatch(从 registry 取 handler 调一次)。"""
    from tools.dispatch import dispatch
    return dispatch(name, args)


def test_middleware_opens_breaker_after_failures(monkeypatch, breakable_tool):
    from agent_reasoning.ReAct.support import tool_circuit
    # 熔断阈值设为 2:前两次调用 timeout(中间件上交),第三次熔断短路
    tool_circuit._breakers.get("boom_tool").failure_threshold = 2
    results, kinds = [], []
    for _ in range(3):
        res, err = call_with_resilience(
            "boom_tool", {}, breakable_tool, invoke=_invoke_dispatch)
        results.append(res)
        kinds.append(err.kind if err else None)
    assert kinds[0] == Kind.TIMEOUT
    assert kinds[1] == Kind.TIMEOUT
    assert kinds[2] == Kind.CIRCUIT_OPEN   # 熔断打开,handler 不再被调用
    assert results == [None, None, None]


def test_middleware_success_closes_breaker(monkeypatch, breakable_tool):
    from agent_reasoning.ReAct.support import tool_circuit
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] <= 1:
            raise httpx.ReadTimeout("down")
        return {"ok": True}

    breakable_tool.handler = flaky
    breakable_tool.retry_times = 3  # 给足重试,让中间件在同一次调用内恢复

    res, err = call_with_resilience(
        "boom_tool", {}, breakable_tool, invoke=_invoke_dispatch)
    assert err is None and res == {"ok": True}
    # 成功后熔断器保持 closed
    assert tool_circuit._breakers.get("boom_tool").state == CircuitState.CLOSED


def test_middleware_retries_then_recovers(breakable_tool):
    # handler 前 2 次超时、第 3 次成功;retry_times=2 -> 中间件退避重试后恢复
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise httpx.ReadTimeout("down")
        return {"ok": True}

    breakable_tool.handler = flaky
    breakable_tool.retry_times = 2
    res, err = call_with_resilience(
        "boom_tool", {}, breakable_tool, invoke=_invoke_dispatch)
    assert err is None and res == {"ok": True}
    assert calls["n"] == 3


def test_middleware_auth_not_retried(breakable_tool):
    calls = {"n": 0}

    def auth_fail():
        calls["n"] += 1
        resp = httpx.Response(403, request=httpx.Request("POST", "http://x"))
        raise httpx.HTTPStatusError("forbidden", request=resp.request, response=resp)

    breakable_tool.handler = auth_fail
    breakable_tool.retry_times = 3
    res, err = call_with_resilience(
        "boom_tool", {}, breakable_tool, invoke=_invoke_dispatch)
    assert err is not None and err.kind == Kind.AUTH
    assert calls["n"] == 1  # 鉴权失败不重试


def test_breaker_disabled_gate(monkeypatch, breakable_tool):
    monkeypatch.setattr(C, "CIRCUIT_BREAKER_ENABLED", False)
    # 熔断关闭:连续失败也始终返回 timeout(不出现 circuit_open)
    for _ in range(6):
        res, err = call_with_resilience(
            "boom_tool", {}, breakable_tool, invoke=_invoke_dispatch)
        assert err is not None and err.kind == Kind.TIMEOUT


def test_on_event_emits_circuit_open(monkeypatch, breakable_tool):
    """熔断状态变化通过 on_event 回调发出 circuit 事件。"""
    from agent_reasoning.ReAct.support import tool_circuit
    tool_circuit._breakers.get("boom_tool").failure_threshold = 1
    events = []
    call_with_resilience("boom_tool", {}, breakable_tool,
                         on_event=lambda ev: events.append(ev),
                         invoke=_invoke_dispatch)
    circuit_events = [e for e in events if e.get("type") == "circuit"]
    assert any(e["state"] == "open" and e["name"] == "boom_tool"
               for e in circuit_events)


def test_on_event_emits_closed_on_recovery(monkeypatch, breakable_tool):
    """半开试探成功 -> closed,通过 on_event 上报。"""
    from agent_reasoning.ReAct.support import tool_circuit
    brk = tool_circuit._breakers.get("boom_tool")
    brk.failure_threshold = 1
    brk.cooldown_seconds = 0.05

    # 第一次失败 -> open
    call_with_resilience("boom_tool", {}, breakable_tool, invoke=_invoke_dispatch)
    time.sleep(0.07)  # 等待冷却 -> 半开
    breakable_tool.handler = lambda: {"ok": True}
    events = []
    call_with_resilience("boom_tool", {}, breakable_tool,
                         on_event=lambda ev: events.append(ev),
                         invoke=_invoke_dispatch)
    states = [e["state"] for e in events if e.get("type") == "circuit"]
    assert "closed" in states
