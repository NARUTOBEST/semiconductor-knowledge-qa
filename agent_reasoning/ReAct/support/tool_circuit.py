# -*- coding: utf-8 -*-
"""按工具维度的熔断器(从 tools/circuit_breaker.py 迁入 ReAct 支撑层)。

韧性「机械重试」机制的一部分:由韧性中间件(tool_resilience.call_with_resilience)
在每次工具调用前后驱动,tools 层本身不再感知熔断。

状态机:
  CLOSED  --连续 failure_threshold 次失败-->  OPEN
  OPEN    --冷却 cooldown_seconds 后-->      HALF_OPEN(放一个试探请求)
  HALF_OPEN --试探成功--> CLOSED;试探失败--> OPEN(重新计时)

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
    """熔断打开时由 before_call 抛出,中间件捕获转为 circuit_open 错误。"""

    def __init__(self, name: str, retry_after: float = 0.0):
        super().__init__(f"工具 {name} 熔断中,暂不可用")
        self.name = name
        self.retry_after = retry_after


class CircuitBreaker:
    def __init__(self, name: str,
                 failure_threshold: int = 5,
                 cooldown_seconds: float = 30.0,
                 on_state_change: Optional[Callable[[str, str, str], None]] = None):
        self.name = name
        self.failure_threshold = max(1, int(failure_threshold))
        self.cooldown_seconds = float(cooldown_seconds)
        self._on_state_change = on_state_change
        self._lock = threading.Lock()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._opened_at = 0.0
        self._half_open_inflight = False

    @property
    def state(self) -> str:
        with self._lock:
            self._maybe_half_open_locked()
            return self._state

    def _maybe_half_open_locked(self) -> None:
        """OPEN 冷却到期 -> HALF_OPEN(调用方持锁)。"""
        if (self._state == CircuitState.OPEN
                and _now() - self._opened_at >= self.cooldown_seconds):
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
                raise CircuitOpenError(self.name, retry_after=retry_after)
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
                 on_state_change: Optional[Callable] = None):
        self._threshold = failure_threshold
        self._cooldown = cooldown_seconds
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
