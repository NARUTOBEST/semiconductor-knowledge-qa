# -*- coding: utf-8 -*-
"""LLM 客户端单例 + 带指数退避的流式调用(从原 chat.service 抽出)。"""
import json
import time
import logging

import config as C

logger = logging.getLogger("agent")

LLM_RETRIES = 3
# 非流式 LLM 请求的读超时(秒):必须在此时间内返回完整响应体。
LLM_TIMEOUT = 20
# 流式 LLM 请求的读超时(秒):这是 token 间"空闲超时",持续吐字不会被切断,
# 仅当连续 10s 无任何新数据才判定卡死断开。
STREAM_TIMEOUT = 10

_llm_client = None


def get_client():
    """全局单例 OpenAI 客户端。"""
    global _llm_client
    if _llm_client is None:
        _llm_client = __import__("openai").OpenAI(
            api_key=C.OPENAI_API_KEY,
            base_url=C.OPENAI_BASE_URL,
            timeout=LLM_TIMEOUT,
        )
    return _llm_client


def llm_create_with_retry(client, trace_id="", retries=LLM_RETRIES, **kwargs):
    """带指数退避重试的 LLM 调用,主模型耗尽后自动切备用模型(P1⑥)。

    返回 (stream, error):成功时 error=None;失败(含备用模型)返回 (None, error)。
    仅对创建请求抛出的同步异常重试/切换;流式中途断线由调用方兜底。
    """
    # retries<1 时循环体一次都不执行,函数会隐式返回 None,
    # 调用方 `stream, err = ...` 解包直接 TypeError -- 钳到至少 1 次
    if retries < 1:
        retries = 1
    model = kwargs.pop("model", "") or ""
    fallback = getattr(C, "OPENAI_FALLBACK_MODEL", "") or ""
    models = [model] + ([fallback] if (fallback and model and fallback != model) else [])
    last_err = None
    for mi, m in enumerate(models):
        kwargs["model"] = m
        for i in range(retries):
            try:
                return client.chat.completions.create(**kwargs), None
            except Exception as e:
                last_err = e
                if i == retries - 1:
                    if mi < len(models) - 1:
                        logger.info(json.dumps({
                            "trace_id": trace_id, "event": "llm_failover",
                            "from_model": m, "to_model": models[mi + 1],
                            "error": str(e)[:120],
                        }, ensure_ascii=False))
                        break  # 换下一个模型再试
                    logger.info(json.dumps({
                        "trace_id": trace_id, "event": "llm_fail",
                        "model": m, "attempt": i + 1, "error": str(e)[:120],
                    }, ensure_ascii=False))
                    return None, e
                wait = 1.5 ** i
                logger.info(json.dumps({
                    "trace_id": trace_id, "event": "llm_retry",
                    "model": m, "attempt": i + 1, "wait_s": round(wait, 1),
                    "error": str(e)[:120],
                }, ensure_ascii=False))
                time.sleep(wait)
    return None, last_err
