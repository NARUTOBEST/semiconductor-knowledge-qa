# -*- coding: utf-8 -*-
"""记忆后台任务共享的 LLM 调用助手:主模型失败自动切备用模型。

独立于 chat.react.support.llm(避免 memories <-> chat 循环导入),但策略一致:
  - 主模型指数退避重试 LLM_RETRIES 次
  - 仍失败则切 OPENAI_FALLBACK_MODEL 再试一轮
  - 全部失败返回 (None, error)
后台任务(摘要/升迁)均为旁路,失败应降级返回 None/空,不抛到主流程。
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional, Tuple

import httpx

import config as C

logger = logging.getLogger("agent")

LLM_RETRIES = 3
# 单次 LLM 请求读超时(秒),与 chat.react.support.llm 保持一致
LLM_TIMEOUT = 20
# 分段超时:connect 判"服务死活"(连不上 1s 判死),read(=LLM_TIMEOUT/调用方覆盖)
# 判"单次调用干不干得完";write/pool 为极端场景保底,正常毫秒级。
LLM_CONNECT_TIMEOUT = float(os.getenv("LLM_CONNECT_TIMEOUT", "1.0"))
LLM_WRITE_TIMEOUT = float(os.getenv("LLM_WRITE_TIMEOUT", "2.0"))
LLM_POOL_TIMEOUT = float(os.getenv("LLM_POOL_TIMEOUT", "1.0"))

_client = None


def _apply_thinking_policy(kwargs: dict) -> None:
    """按模型族注入思考开关(与 chat 侧 ReAct/support/llm.py 同策略;此处独立
    实现,保持 memories 不反向依赖 agent_reasoning):
    - doubao 系:支持 thinking.type=disabled;升迁门/摘要类调用带思考 12-14s,
      3s 级门超时必挂,关闭后 2-3s;
    - GLM 系:纯思考模型,请求体带 thinking 参数直接 400 InvalidParameter。
    LLM_ENABLE_THINKING=1 整体豁免。"""
    if getattr(C, "LLM_ENABLE_THINKING", False):
        return
    low = str(kwargs.get("model") or "").lower()
    if "doubao" in low:
        extra = dict(kwargs.get("extra_body") or {})
        extra.setdefault("thinking", {"type": "disabled"})
        kwargs["extra_body"] = extra
    elif "glm" in low:
        kwargs.pop("thinking", None)
        extra = dict(kwargs.get("extra_body") or {})
        extra.pop("thinking", None)
        if extra:
            kwargs["extra_body"] = extra
        else:
            kwargs.pop("extra_body", None)


def _record_internal_usage(resp) -> None:
    """把记忆链内部 LLM 的 token 用量记到 internal 分账(Req12),best-effort 不影响旁路。"""
    try:
        u = getattr(resp, "usage", None)
        if not u:
            return
        from support.metrics import metrics  # 同进程,server/ 在 sys.path
        metrics.record_internal_tokens(
            getattr(u, "prompt_tokens", 0) or 0,
            getattr(u, "completion_tokens", 0) or 0)
    except Exception:  # noqa: BLE001
        pass


def _seg_timeout(read: float, budget: Optional[float] = None) -> httpx.Timeout:
    """把单次请求超时拆成分段(read=干活上限,connect=建连上限)。

    budget(剩余整链预算)不为 None 时 connect/read 都压进预算内,
    保证分段配置不把总耗时拖过 deadline。
    """
    rem = budget if budget is not None else float("inf")
    return httpx.Timeout(
        connect=min(LLM_CONNECT_TIMEOUT, rem),
        read=min(max(read, 0.05), rem),
        write=LLM_WRITE_TIMEOUT,
        pool=LLM_POOL_TIMEOUT,
    )


def get_client():
    """全局单例 OpenAI 客户端(文本模型,云 API)。"""
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI(
            base_url=getattr(C, "EFFECTIVE_LLM_BASE_URL", C.OPENAI_BASE_URL),
            api_key=getattr(C, "EFFECTIVE_LLM_API_KEY", C.OPENAI_API_KEY),
            timeout=_seg_timeout(LLM_TIMEOUT),
        )
    return _client


def chat_completion_with_fallback(
    *, trace_id: str = "", retries: int = LLM_RETRIES,
    deadline: Optional[float] = None, **kwargs
) -> Tuple[Optional[Any], Optional[Exception]]:
    """带主/备切换的 chat.completions.create。

    model 缺省取 OPENAI_TEXT_MODEL;备用模型取 OPENAI_FALLBACK_MODEL。
    若主模型本身就是备用模型(或未配置备用),不重复切换。
    deadline:可选 monotonic 时间戳(整链时间预算)。每次尝试前检查剩余预算,
    耗尽立即返回失败(调用方走各自降级路径);单次请求超时同时压到剩余预算内,
    保证"挂死的服务"最坏也只吃掉预算本身而非 各超时×重试次数 的串行和。
    返回 (response, error):成功时 error=None。
    """
    if retries < 1:
        retries = 1
    # 默认主模型用网关感知的 MODEL_MAIN(网关模式=逻辑名 main,云端=deepseek)
    model = kwargs.pop("model", "") or getattr(C, "MODEL_MAIN", "") or C.OPENAI_TEXT_MODEL
    fallback = (getattr(C, "EFFECTIVE_FALLBACK_MODEL", "")
                or getattr(C, "OPENAI_FALLBACK_MODEL", "") or "")
    models = [model]
    if fallback and fallback != model:
        models.append(fallback)

    def _remaining() -> Optional[float]:
        return None if deadline is None else deadline - time.monotonic()

    client = get_client()
    last_err: Optional[Exception] = None
    # 数字型单次超时配置先取出(分段化是逐次重建的,不能回写 kwargs 后再读它)
    to_cfg = kwargs.get("timeout")
    numeric_timeout = to_cfg if isinstance(to_cfg, (int, float)) else None
    for mi, m in enumerate(models):
        kwargs["model"] = m
        _apply_thinking_policy(kwargs)
        for i in range(retries):
            rem = _remaining()
            if rem is not None and rem <= 0:
                logger.info("memories llm budget exhausted [%s] model=%s",
                            trace_id, m)
                return None, (last_err or TimeoutError("memory chain budget exhausted"))
            # 单次请求超时不越过剩余预算(硬帽 = 预算,而非 配置超时×重试 的串行和);
            # 数字超时逐次重建为分段(connect/read),剩余预算每次都重新压缩
            if numeric_timeout is not None:
                kwargs["timeout"] = _seg_timeout(
                    read=numeric_timeout, budget=rem)
            try:
                resp = client.chat.completions.create(**kwargs)
                _record_internal_usage(resp)
                return resp, None
            except Exception as e:  # noqa: BLE001
                last_err = e
                if i == retries - 1:
                    if mi < len(models) - 1:
                        logger.info(
                            "memories llm failover [%s]: %s -> %s: %s",
                            trace_id, m, models[mi + 1], str(e)[:120])
                        break  # 换下一个模型
                    logger.info(
                        "memories llm fail [%s] model=%s: %s: %s",
                        trace_id, m, type(e).__name__, str(e)[:120])
                    return None, e
                wait = 1.5 ** i
                rem = _remaining()
                if rem is not None:
                    if rem <= 0:
                        return None, e
                    wait = min(wait, rem)
                time.sleep(wait)
    return None, last_err
