# -*- coding: utf-8 -*-
"""工具调用韧性中间件(「机械重试」层)。

包裹【单次】工具调用,透明处理与推理无关的机械性失败,对图编排不可见:
  - 熔断前置(OPEN 直接短路,不调用);
  - 限流(429)/ 网络超时 / 传输抖动 / 5xx -> 指数退避重试(以 spec.retry_times 为界,
    且不超过端到端硬预算剩余时间);
  - handler 意外崩溃(非 HTTP 异常)-> 最多重试 1 次;
  - 401/403 鉴权失败 -> 不重试;
  - 成功复位熔断器;重试耗尽/持久失败 -> 记录熔断失败并上交结构化错误。

本中间件【只做机制、不做决策】:返回裸结果或 ToolCallError,
「是否摘工具 / 是否降级常识 / 是否换词再检索」由 reflect 决策节点研判。

熔断状态机见 tool_circuit(跨请求进程内单例);错误分类见 tool_errors。
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable, Optional

from tools.base import ToolSpec

from . import tool_circuit
from .tool_circuit import CircuitOpenError
from .tool_errors import (
    Stage, Kind, ToolCallError, classify_exception, BREAKER_FAILURE_KINDS,
)

logger = logging.getLogger("agent.tool_resilience")

# 退避基数(秒):wait = BASE * (attempt + 1)
_RETRY_BACKOFF_BASE = 0.5
# handler 内部崩溃(非 HTTP)最多重试次数,无论 spec.retry_times 多大
_CRASH_MAX_RETRY = 1


def _remaining(deadline: Optional[float]) -> Optional[float]:
    if deadline is None:
        return None
    return deadline - time.time()


def call_with_resilience(
    name: str,
    args: dict,
    spec: Optional[ToolSpec],
    *,
    deadline: Optional[float] = None,
    on_event: Optional[Callable[[dict], None]] = None,
    invoke: Optional[Callable[..., Any]] = None,
) -> tuple[Any, Optional[ToolCallError]]:
    """对单次工具调用施加熔断/限流/重试/超时兜底。

    :param name: 工具名
    :param args: 已通过 runtime 校验的参数(dict)
    :param spec: ToolSpec(可为 None:未知工具,但正常已被 validate_generation 拦截)
    :param deadline: 端到端硬截止墙钟时间戳;None 表示不设硬预算
    :param on_event: 可选熔断事件回调(供 execute 节点补发 SSE circuit 事件)
    :param invoke: 实际单次调用(默认 tools.dispatch;测试可注入桩)
    :returns: (result, None) 成功;或 (None, ToolCallError) 失败。
    """
    if invoke is None:
        from tools.dispatch import dispatch as invoke  # 延迟导入,便于 monkeypatch

    retry_times = int(getattr(spec, "retry_times", 0) or 0)
    attempts = max(1, retry_times + 1)

    # ---- 熔断前置 ----
    breaker = tool_circuit.get_breaker(name)
    if breaker is not None:
        prev_state = breaker.state
        try:
            breaker.before_call()
        except CircuitOpenError:
            logger.warning("tool %s 熔断打开,中间件短路", name)
            return None, ToolCallError(
                stage=Stage.EXECUTION, kind=Kind.CIRCUIT_OPEN,
                message="检索服务熔断中,已暂时停止调用", tool=name, retryable=False)
        _maybe_emit(breaker, prev_state, on_event, name)

    last_kind = Kind.UPSTREAM
    last_exc: Optional[BaseException] = None
    last_detail: Optional[str] = None
    crash_attempts = 0

    for attempt in range(attempts):
        # 硬预算:剩余时间不足以再试则提前结束
        rem = _remaining(deadline)
        if rem is not None and rem <= 0:
            return None, ToolCallError(
                stage=Stage.EXECUTION, kind=Kind.BUDGET_TIMEOUT,
                message="端到端时限已到", tool=name, retryable=False)

        try:
            result = invoke(name, args)
        except Exception as e:  # noqa: BLE001 - 中间件必须兜住一切 handler 异常
            last_exc = e
            kind, retryable = classify_exception(e)

            # 崩溃类单独限额(最多 _CRASH_MAX_RETRY 次)
            if kind == Kind.CRASH:
                if crash_attempts >= _CRASH_MAX_RETRY:
                    retryable = False
                crash_attempts += 1

            logger.info("tool %s 第 %d 次尝试失败: %s: %s",
                        name, attempt + 1, kind,
                        f"{type(e).__name__}: {e}"[:200])

            if not retryable or kind == Kind.AUTH:
                last_kind = kind
                break  # 鉴权/不可重试:立即终止
            if attempt >= attempts - 1:
                last_kind = kind
                break  # 重试用尽
            # 退避(不越过硬预算)
            wait = _RETRY_BACKOFF_BASE * (attempt + 1)
            rem = _remaining(deadline)
            if rem is not None:
                wait = min(wait, max(0.0, rem - 0.05))
                if wait <= 0:
                    last_kind = Kind.BUDGET_TIMEOUT
                    break
            time.sleep(wait)
            continue

        # handler 未抛异常:检查业务失败 dict({"error": ...})。
        # 未知/禁用工具已被 validate_generation 拦截;能到这里的业务错误视为下游服务失败
        # (如检索微服务返回错误),不重试(handler 已表达确定失败),记熔断后上交为机械错误。
        if isinstance(result, dict) and "error" in result:
            last_kind = _map_business_error(result.get("error_type"))
            last_detail = str(result.get("error"))[:200]
            logger.warning("tool %s 返回业务错误: %s", name, last_detail)
            break

        # 成功
        if breaker is not None:
            prev_state = breaker.state
            breaker.on_success()
            _maybe_emit(breaker, prev_state, on_event, name)
        return result, None

    # ---- 重试用尽 / 持久失败:记熔断 ----
    if breaker is not None:
        prev_state = breaker.state
        # auth 等非服务抖动类按 fatal 记账(CLOSED 不计数);HALF_OPEN 探测期仍会重新 OPEN
        breaker.on_failure(
            last_kind if last_kind in BREAKER_FAILURE_KINDS else "fatal")
        _maybe_emit(breaker, prev_state, on_event, name)

    message = _failure_message(last_kind, last_exc, last_detail)
    return None, ToolCallError(
        stage=Stage.EXECUTION, kind=last_kind, message=message,
        tool=name, retryable=last_kind in (Kind.TIMEOUT, Kind.RATE_LIMITED,
                                           Kind.UPSTREAM, Kind.CRASH),
        retries=min(attempt, attempts - 1))


def _map_business_error(error_type: Optional[str]) -> str:
    """handler/dispatch 返回的 {"error_type":...} 映射到机械 kind。"""
    mapping = {
        "timeout": Kind.TIMEOUT,
        "retryable": Kind.UPSTREAM,
        "circuit_open": Kind.CIRCUIT_OPEN,
        "auth": Kind.AUTH,
    }
    return mapping.get(error_type or "", Kind.UPSTREAM)


def _failure_message(kind: str, exc: Optional[BaseException],
                     detail: Optional[str] = None) -> str:
    if exc is not None:
        return f"{type(exc).__name__}: {str(exc)[:200]}"
    if detail:
        return detail
    return {"timeout": "响应超时", "rate_limited": "触发限流",
            "upstream": "下游服务错误", "crash": "工具内部异常",
            "auth": "鉴权失败", "circuit_open": "服务熔断中",
            "budget_timeout": "超出响应时限"}.get(kind, "工具调用失败")


def _maybe_emit(breaker, prev_state: str,
                on_event: Optional[Callable[[dict], None]], name: str) -> None:
    """熔断状态变化时通过 on_event 发出 circuit 事件(execute 节点转 SSE)。"""
    if on_event is None or breaker is None:
        return
    new_state = breaker.state
    if new_state != prev_state:
        try:
            on_event({"type": "circuit", "name": name, "state": new_state})
        except Exception:
            pass
