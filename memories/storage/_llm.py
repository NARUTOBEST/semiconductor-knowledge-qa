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
import time
from typing import Any, Optional, Tuple

import config as C

logger = logging.getLogger("agent")

LLM_RETRIES = 3
# 单次 LLM 请求读超时(秒),与 chat.react.support.llm 保持一致
LLM_TIMEOUT = 20

_client = None


def get_client():
    """全局单例 OpenAI 客户端(文本模型,云 API)。"""
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI(
            base_url=C.OPENAI_BASE_URL,
            api_key=C.OPENAI_API_KEY,
            timeout=LLM_TIMEOUT,
        )
    return _client


def chat_completion_with_fallback(
    *, trace_id: str = "", retries: int = LLM_RETRIES, **kwargs
) -> Tuple[Optional[Any], Optional[Exception]]:
    """带主/备切换的 chat.completions.create。

    model 缺省取 OPENAI_TEXT_MODEL;备用模型取 OPENAI_FALLBACK_MODEL。
    若主模型本身就是备用模型(或未配置备用),不重复切换。
    返回 (response, error):成功时 error=None。
    """
    if retries < 1:
        retries = 1
    model = kwargs.pop("model", "") or C.OPENAI_TEXT_MODEL
    fallback = getattr(C, "OPENAI_FALLBACK_MODEL", "") or ""
    models = [model]
    if fallback and fallback != model:
        models.append(fallback)

    client = get_client()
    last_err: Optional[Exception] = None
    for mi, m in enumerate(models):
        kwargs["model"] = m
        for i in range(retries):
            try:
                return client.chat.completions.create(**kwargs), None
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
                time.sleep(wait)
    return None, last_err
