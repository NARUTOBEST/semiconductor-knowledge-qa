# -*- coding: utf-8 -*-
"""LLM 调用三状态熔断单测(tool_circuit 试探上限 + llm_create_with_retry 接线)。

规格(2026-10-10 用户拍板):CLOSED 下指数退避重试+超时限定;连续 3 次失败→
OPEN 冷却 10s→HALF_OPEN 试探;连续 2 次试探失败→放弃期(exhausted),调用方
提示用户"该环节 LLM 服务不可用,请稍后作答,本次作答无效";放弃期过后重走循环。
"""
import pytest

import config as C
from agent_reasoning.ReAct.support import tool_circuit
from agent_reasoning.ReAct.support.llm import (
    LLMServiceUnavailable, llm_create_with_retry,
)
from agent_reasoning.ReAct.support.tool_circuit import (
    CircuitBreaker, CircuitOpenError, CircuitState, _llm_breakers,
    get_llm_breaker,
)


class _Clock:
    """可控时钟:monkeypatch tool_circuit._now,测试里手动推进。"""

    def __init__(self):
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


@pytest.fixture(autouse=True)
def clock(monkeypatch):
    c = _Clock()
    monkeypatch.setattr(tool_circuit, "_now", c)
    yield c


@pytest.fixture(autouse=True)
def _clean_llm_breakers(monkeypatch):
    monkeypatch.setattr(C, "CIRCUIT_BREAKER_ENABLED", True)
    # 屏蔽备用模型 failover:否则主模型失败会被备用模型救回,熔断计不到失败
    monkeypatch.setattr(C, "EFFECTIVE_FALLBACK_MODEL", "")
    monkeypatch.setattr(C, "OPENAI_FALLBACK_MODEL", "")
    _llm_breakers.reset_all()
    yield
    _llm_breakers.reset_all()


# ==================== 试探上限状态机 ====================
def test_two_failed_probes_enter_exhausted(clock):
    cb = CircuitBreaker("llm-t1", failure_threshold=1, cooldown_seconds=10,
                        half_open_probe_limit=2, exhausted_cooldown_seconds=60)
    cb.on_failure("timeout")
    assert cb.state == CircuitState.OPEN
    clock.advance(10)
    cb.before_call()
    cb.on_failure("timeout")          # 第 1 次试探失败 -> OPEN 重新计时
    clock.advance(10)
    cb.before_call()
    cb.on_failure("timeout")          # 第 2 次试探失败 -> 放弃期
    with pytest.raises(CircuitOpenError) as ei:
        cb.before_call()
    assert ei.value.exhausted is True


def test_exhausted_blocks_until_expired_then_recovers(clock):
    cb = CircuitBreaker("llm-t2", failure_threshold=1, cooldown_seconds=10,
                        half_open_probe_limit=2, exhausted_cooldown_seconds=60)
    cb.on_failure("timeout")          # CLOSED -> OPEN
    for _ in range(2):                # 两次半开试探均失败 -> 放弃期
        clock.advance(10)
        cb.before_call()
        cb.on_failure("timeout")
    with pytest.raises(CircuitOpenError) as ei:
        cb.before_call()
    assert ei.value.exhausted is True
    clock.advance(10)
    with pytest.raises(CircuitOpenError) as ei:   # 放弃期内即使过了冷却也不放试探
        cb.before_call()
    assert ei.value.exhausted is True
    clock.advance(51)                 # 过放弃期:清零计数,重走 OPEN 冷却
    cb.before_call()                  # 不抛,放试探
    cb.on_success()
    assert cb.state == CircuitState.CLOSED


def test_probe_success_resets(clock):
    cb = CircuitBreaker("llm-t3", failure_threshold=1, cooldown_seconds=10,
                        half_open_probe_limit=2, exhausted_cooldown_seconds=60)
    cb.on_failure("timeout")
    clock.advance(10)
    cb.before_call()
    cb.on_failure("timeout")   # 1 次试探失败
    clock.advance(10)
    cb.before_call()
    cb.on_success()            # 第 2 次试探成功 -> CLOSED + 试探计数清零
    assert cb.state == CircuitState.CLOSED
    cb.on_failure("timeout")
    assert cb.state == CircuitState.OPEN   # 从零计数重新开闸
    clock.advance(10)
    cb.before_call()
    cb.on_failure("timeout")   # 试探失败仅 1 次,不进放弃期
    with pytest.raises(CircuitOpenError) as ei:
        cb.before_call()
    assert ei.value.exhausted is False


def test_legacy_breaker_no_probe_limit(clock):
    cb = CircuitBreaker("tool-t", failure_threshold=1, cooldown_seconds=10)
    for _ in range(5):         # 无上限(工具熔断默认):无限循环也不进放弃期
        clock.advance(10)
        cb.before_call()
        cb.on_failure("timeout")
    with pytest.raises(CircuitOpenError) as ei:
        cb.before_call()
    assert ei.value.exhausted is False


def test_registry_params():
    brk = get_llm_breaker("reg-check")
    assert brk.failure_threshold == C.LLM_BREAKER_FAILURE_THRESHOLD
    assert brk.cooldown_seconds == C.LLM_BREAKER_COOLDOWN_SECONDS
    assert brk.half_open_probe_limit == 2


# ==================== llm_create_with_retry 接线 ====================
class _FlakyClient:
    """前 fail_n 次抛超时,之后成功。"""

    def __init__(self, fail_n):
        self.fail_n = fail_n
        self.calls = 0

    class _Completions:
        def __init__(self, outer):
            self._outer = outer

        def create(self, **kwargs):
            self._outer.calls += 1
            if self._outer.calls <= self._outer.fail_n:
                raise TimeoutError("upstream hang")

            class _R:
                choices = []
            return _R()

    @property
    def chat(self):
        outer = self

        class _Chat:
            completions = _FlakyClient._Completions(outer)
        return _Chat()


def test_llm_create_opens_breaker_then_raises_unavailable():
    client = _FlakyClient(fail_n=99)
    site = "t-unavail"
    for _ in range(C.LLM_BREAKER_FAILURE_THRESHOLD):
        resp, err = llm_create_with_retry(
            client, trace_id="x", site=site, retries=1,
            model="m", messages=[], stream=False)
        assert resp is None and err is not None
    # 阈值达到已 OPEN;冷却 10s 未过,直接拒绝并带用户提示
    with pytest.raises(LLMServiceUnavailable) as ei:
        llm_create_with_retry(
            client, trace_id="x", site=site, retries=1,
            model="m", messages=[], stream=False)
    assert "LLM 服务不可用" in str(ei.value)
    assert ei.value.site == site


def test_llm_create_success_records_and_recovers():
    client = _FlakyClient(fail_n=1)
    site = "t-recover"
    _, err = llm_create_with_retry(
        client, trace_id="x", site=site, retries=1,
        model="m", messages=[], stream=False)
    assert err is not None          # 第一次失败(计 1 次,未达阈值 3)
    _, err = llm_create_with_retry(
        client, trace_id="x", site=site, retries=1,
        model="m", messages=[], stream=False)
    assert err is None              # 成功 -> 计数清零
    assert tool_circuit.get_llm_breaker(site).state == CircuitState.CLOSED
    # 再失败 2 次不应触发(计数已复位,阈值 3)
    for _ in range(2):
        client2 = _FlakyClient(fail_n=99)
        _, err = llm_create_with_retry(
            client2, trace_id="x", site=site, retries=1,
            model="m", messages=[], stream=False)
        assert err is not None
    assert tool_circuit.get_llm_breaker(site).state == CircuitState.CLOSED


def test_llm_create_default_site_not_trace_id():
    """site 缺省记在 "llm" 而非 trace_id(否则每请求新熔断器永远到不了阈值)。"""
    client = _FlakyClient(fail_n=99)
    for _ in range(C.LLM_BREAKER_FAILURE_THRESHOLD):
        llm_create_with_retry(
            client, trace_id="dyn-abc", retries=1,
            model="m", messages=[], stream=False)
    with pytest.raises(LLMServiceUnavailable) as ei:
        llm_create_with_retry(
            client, trace_id="other-xyz", retries=1,
            model="m", messages=[], stream=False)
    assert ei.value.site == "llm"


def test_disabled_feature_skips_breaker(monkeypatch):
    monkeypatch.setattr(C, "CIRCUIT_BREAKER_ENABLED", False)
    client = _FlakyClient(fail_n=99)
    for _ in range(C.LLM_BREAKER_FAILURE_THRESHOLD + 2):
        _, err = llm_create_with_retry(
            client, trace_id="x", site="t-off", retries=1,
            model="m", messages=[], stream=False)
        assert err is not None      # 始终返回 error,不抛熔断异常
