# -*- coding: utf-8 -*-
"""按工具维度的熔断器(从 tools/circuit_breaker.py 迁入 ReAct 支撑层)。

韧性「机械重试」机制的一部分:由韧性中间件(tool_resilience.call_with_resilience)
在每次工具调用前后驱动,tools 层本身不再感知熔断。

状态机:
  CLOSED  --连续 failure_threshold 次失败-->  OPEN
  OPEN    --冷却 cooldown_seconds 后-->      HALF_OPEN(放一个试探请求)
  HALF_OPEN --试探成功--> CLOSED;试探失败--> OPEN(重新计时)

试探上限(half_open_probe_limit,LLM 熔断用):HALF_OPEN 连续试探失败 N 次(默认
0=不限制,按上面的循环无限重试)后进入【放弃期】——exhausted_cooldown_seconds 内
before_call 一律抛 CircuitOpenError(exhausted=True),调用方据此向用户明确报
"该环节服务不可用"并终止本次作答,不再无限 open→halfopen 循环;放弃期过后
清零试探计数,从 OPEN 冷却重新走一轮完整循环(服务恢复后可自愈)。

FATAL 类错误(参数错误/4xx)不计入失败计数,只对 TIMEOUT/RETRYABLE 计数。
线程安全;状态变化通过可选回调通知(用于发 SSE status 事件)。
"""
from __future__ import annotations

import threading
from typing import Callable, Optional

import config as C


class CircuitState:
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(Exception):
    """熔断打开时由 before_call 抛出,中间件捕获转为 circuit_open 错误。

    exhausted=True 表示已连续 2 次半开试探失败进入放弃期:上游应向用户
    明确提示"服务不可用,请稍后作答"并认定本次作答无效,而非静默降级。
    """

    def __init__(self, name: str, retry_after: float = 0.0,
                 exhausted: bool = False):
        super().__init__(f"工具 {name} 熔断中,暂不可用")
        self.name = name
        self.retry_after = retry_after
        self.exhausted = exhausted


class CircuitBreaker:
    def __init__(self, name: str,
                 failure_threshold: int = 5,
                 cooldown_seconds: float = 30.0,
                 half_open_probe_limit: int = 0,
                 exhausted_cooldown_seconds: float = 60.0,
                 on_state_change: Optional[Callable[[str, str, str], None]] = None):
        self.name = name
        self.failure_threshold = max(1, int(failure_threshold))
        self.cooldown_seconds = float(cooldown_seconds)
        # 0=不限制(无限 open→halfopen 循环);>0=连续 N 次试探失败进入放弃期
        self.half_open_probe_limit = max(0, int(half_open_probe_limit))
        self.exhausted_cooldown_seconds = float(exhausted_cooldown_seconds)
        self._on_state_change = on_state_change
        self._lock = threading.Lock()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._opened_at = 0.0
        self._half_open_inflight = False
        self._probe_failures = 0
        self._exhausted_until = 0.0

    @property
    def state(self) -> str:
        with self._lock:
            self._maybe_half_open_locked()
            return self._state

    def _maybe_half_open_locked(self) -> None:
        """OPEN 冷却到期 -> HALF_OPEN(调用方持锁);放弃期到期则重置试探计数。"""
        now = _now()
        if self._exhausted_until and now >= self._exhausted_until:
            # 放弃期结束:给服务一个重新自愈的机会,从 OPEN 冷却重走完整循环
            self._exhausted_until = 0.0
            self._probe_failures = 0
        if (self._state == CircuitState.OPEN
                and not self._exhausted_until
                and now - self._opened_at >= self.cooldown_seconds):
            self._set_state_locked(CircuitState.HALF_OPEN)

    def _set_state_locked(self, new_state: str) -> None:
        old = self._state
        self._state = new_state
        if new_state == CircuitState.OPEN:
            self._opened_at = _now()
            self._half_open_inflight = False
        elif new_state == CircuitState.CLOSED:
            self._failure_count = 0
            self._half_open_inflight = False
            self._probe_failures = 0
            self._exhausted_until = 0.0
        if old != new_state and self._on_state_change is not None:
            try:
                self._on_state_change(self.name, old, new_state)
            except Exception:
                pass  # 回调失败不影响熔断

    def before_call(self) -> None:
        """调用工具前检查;OPEN 抛 CircuitOpenError,HALF_OPEN 放行一个试探。"""
        with self._lock:
            self._maybe_half_open_locked()
            if self._state == CircuitState.OPEN:
                retry_after = max(
                    0.0,
                    self.cooldown_seconds - (_now() - self._opened_at))
                raise CircuitOpenError(self.name, retry_after=retry_after,
                                       exhausted=bool(self._exhausted_until))
            if self._state == CircuitState.HALF_OPEN:
                # 半开期间只放一个试探请求,其余直接当打开
                if self._half_open_inflight:
                    raise CircuitOpenError(self.name)
                self._half_open_inflight = True

    def on_success(self) -> None:
        with self._lock:
            self._set_state_locked(CircuitState.CLOSED)

    def on_failure(self, error_type: str) -> None:
        """记录一次失败。

        FATAL 类错误(参数错误/4xx)在 CLOSED 下不计数(不代表服务故障);
        TIMEOUT / RETRYABLE / UNKNOWN 计入。
        但 HALF_OPEN 探测期例外:探测只要没成功(无论 fatal 与否)都判定服务未恢复,
        必须释放探测槽并转回 OPEN 重新计时——否则会永久卡在
        HALF_OPEN + _half_open_inflight=True,之后所有请求都被拒。
        """
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                # 试探失败(含 fatal) -> 释放探测槽 + 重新打开
                self._set_state_locked(CircuitState.OPEN)
                # 试探上限:连续 N 次试探都没成功 -> 放弃期,期间 before_call
                # 抛 exhausted=True,上游提示用户"服务不可用"并终止本次作答。
                if self.half_open_probe_limit > 0:
                    self._probe_failures += 1
                    if self._probe_failures >= self.half_open_probe_limit:
                        self._exhausted_until = (
                            _now() + self.exhausted_cooldown_seconds)
                        logger.critical(
                            "breaker %s: %d consecutive half-open probes "
                            "failed, exhausted for %.0fs",
                            self.name, self._probe_failures,
                            self.exhausted_cooldown_seconds)
                return
            if error_type == "fatal":
                return
            if self._state == CircuitState.OPEN:
                return
            self._failure_count += 1
            if self._failure_count >= self.failure_threshold:
                self._set_state_locked(CircuitState.OPEN)

    def reset(self) -> None:
        """测试/运维用:强制复位到 CLOSED。"""
        with self._lock:
            self._set_state_locked(CircuitState.CLOSED)


def _now() -> float:
    """隔离 time.time,便于测试 monkeypatch。"""
    import time
    return time.time()


class CircuitBreakerRegistry:
    """按工具名持有 CircuitBreaker 实例(进程内单例)。"""

    def __init__(self, failure_threshold: int = 5, cooldown_seconds: float = 30.0,
                 half_open_probe_limit: int = 0,
                 exhausted_cooldown_seconds: float = 60.0,
                 on_state_change: Optional[Callable] = None):
        self._threshold = failure_threshold
        self._cooldown = cooldown_seconds
        self._probe_limit = half_open_probe_limit
        self._exhausted_cooldown = exhausted_cooldown_seconds
        self._on_state_change = on_state_change
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = threading.Lock()

    def get(self, name: str) -> CircuitBreaker:
        with self._lock:
            b = self._breakers.get(name)
            if b is None:
                b = CircuitBreaker(
                    name,
                    failure_threshold=self._threshold,
                    cooldown_seconds=self._cooldown,
                    half_open_probe_limit=self._probe_limit,
                    exhausted_cooldown_seconds=self._exhausted_cooldown,
                    on_state_change=self._on_state_change)
                self._breakers[name] = b
            return b

    def reset_all(self) -> None:
        with self._lock:
            for b in self._breakers.values():
                b.reset()

    def snapshot(self) -> dict[str, str]:
        with self._lock:
            return {name: b.state for name, b in self._breakers.items()}


# 进程内单例:参数来自 config,可经环境变量覆盖。韧性中间件与健康度自适应共用。
#
# 单进程假设(有意接受,勿"修复"):熔断计数不外置 Redis——熔断是韧性逻辑,
# 依赖可能正在故障的组件等于给保护机制加故障点,且每次工具调用加一次 Redis
# 往返不值得。多 worker 下各 worker 独立计数,后果仅是保守性变差(实际故障
# 约 N 倍次数才熔断,不产生错误行为);若真开多 worker,把
# CIRCUIT_BREAKER_FAILURE_THRESHOLD 除以 worker 数即可(零代码,env 覆盖)。
import logging

logger = logging.getLogger("agent.tool_circuit")

_breakers = CircuitBreakerRegistry(
    failure_threshold=C.CIRCUIT_BREAKER_FAILURE_THRESHOLD,
    cooldown_seconds=C.CIRCUIT_BREAKER_COOLDOWN_SECONDS,
    on_state_change=lambda name, old, new: logger.warning(
        "tool %s circuit breaker: %s -> %s", name, old, new),
)


def circuit_snapshot() -> dict[str, str]:
    """各工具当前熔断状态 {tool_name: 'closed'|'open'|'half_open'}。

    供 agent 节点感知工具健康度:open 的工具从本轮 schema 摘除并改走替代策略。
    熔断器关闭(CIRCUIT_BREAKER_ENABLED=False)时返回空 dict。
    """
    if not C.CIRCUIT_BREAKER_ENABLED:
        return {}
    return _breakers.snapshot()


def get_breaker(name: str) -> Optional[CircuitBreaker]:
    """取某工具的熔断器;熔断特性关闭时返回 None。"""
    if not C.CIRCUIT_BREAKER_ENABLED:
        return None
    return _breakers.get(name)


# ---- LLM 调用专用熔断注册表(与工具熔断分池,参数独立)----
# 规格(2026-10-10 用户拍板,所有 LLM 调用环节统一):CLOSED 下指数退避重试+
# 超时限定(由 llm_create_with_retry 承担);连续 3 次失败(每次内部已重试/
# failover 过)→ OPEN 冷却 10s → HALF_OPEN 试探;连续 2 次试探失败 → 放弃期,
# 上游提示用户"该环节 LLM 服务不可用,请稍后作答,本次作答无效";放弃期 60s
# 后重置计数,从 OPEN 冷却重走循环(服务恢复自愈)。按环节(site)独立计数,
# 环节名见各 llm_create_with_retry 调用点的 site= 参数。
_llm_breakers = CircuitBreakerRegistry(
    failure_threshold=int(getattr(C, "LLM_BREAKER_FAILURE_THRESHOLD", 3)),
    cooldown_seconds=float(getattr(C, "LLM_BREAKER_COOLDOWN_SECONDS", 10.0)),
    half_open_probe_limit=2,
    exhausted_cooldown_seconds=float(
        getattr(C, "LLM_BREAKER_EXHAUSTED_COOLDOWN_SECONDS", 60.0)),
    on_state_change=lambda name, old, new: logger.warning(
        "llm site %s circuit breaker: %s -> %s", name, old, new),
)


def get_llm_breaker(site: str) -> Optional[CircuitBreaker]:
    """取某 LLM 环节的熔断器;熔断特性关闭时返回 None(调用方跳过熔断)。"""
    if not C.CIRCUIT_BREAKER_ENABLED:
        return None
    return _llm_breakers.get(site)
