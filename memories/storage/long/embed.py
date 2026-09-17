# -*- coding: utf-8 -*-
"""长期记忆文本嵌入:复用检索微服务(8002)的 BGE-m3 /embed_text。

主服务进程不加载 torch/模型,向量由检索微服务持有(与 RAG/embed_http、eval/judge
同一条 HTTP 路径)。长期记忆只用 dense(1024 维)做 pgvector 余弦召回,不需要 sparse。

旁路:微服务不可用 / 未装 httpx 时返回 None,调用方据此跳过向量部分(偏好仍可结构化
存储,只是无语义召回),不抛异常。
"""
from __future__ import annotations

import logging
import os
import sys
import time

_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
for _p in (os.path.join(_PROJECT_ROOT, "config"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import config as C  # noqa: E402

logger = logging.getLogger("agent")

# 个性化嵌入是【旁路】:绝不能拖慢聊天主流程。给短超时(个性化不值得等),
# 且失败后进入冷却(熔断),冷却期内直接返回 None、不再发请求,避免每轮都去敲一个
# 挂掉的检索微服务(后台抽取线程同样受益)。冷却到期后自动试探恢复。
_TIMEOUT = float(os.getenv("LONG_MEM_EMBED_TIMEOUT", "4"))
_FAIL_COOLDOWN = float(os.getenv("LONG_MEM_EMBED_COOLDOWN", "30"))
_fail_until = 0.0  # monotonic 时间戳;此前视为服务不可用
# 单进程假设(无害,不改):多 worker 下各 worker 各自冷却,最多多几次对挂掉
# 服务的探测(每 worker 30s 一次),无需外置。


def _base_url() -> str:
    return getattr(C, "RETRIEVAL_SERVICE_URL", "http://127.0.0.1:8002").rstrip("/")


def _headers():
    """服务间鉴权:RETRIEVAL_INTERNAL_TOKEN 非空时携带 X-Internal-Token。"""
    tok = getattr(C, "RETRIEVAL_INTERNAL_TOKEN", "")
    return {"X-Internal-Token": tok} if tok else None


def embed_texts(texts):
    """把若干文本编码为 dense 向量。

    :returns: list[list[float]](与 texts 等长,每条 1024 维);失败返回 None。
    """
    global _fail_until
    texts = [t for t in (texts or []) if t]
    if not texts:
        return []
    # 熔断冷却期内:不发请求,直接降级(避免每轮阻塞)
    if time.monotonic() < _fail_until:
        return None
    try:
        import httpx
        resp = httpx.post(
            _base_url() + "/embed_text",
            json={"texts": list(texts)},
            headers=_headers(),
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        dense = resp.json().get("dense")
        if not dense or len(dense) != len(texts):
            raise RuntimeError("embed_text 返回 dense 为空/长度不符")
        return [list(map(float, v)) for v in dense]
    except Exception as e:  # noqa: BLE001
        _fail_until = time.monotonic() + _FAIL_COOLDOWN
        logger.info("long-memory embed skipped (retrieval service?): %s: %s "
                    "(冷却 %.0fs)", type(e).__name__, str(e)[:120], _FAIL_COOLDOWN)
        return None


def embed_one(text: str):
    """便捷:单条文本 -> 向量 list[float];失败返回 None。"""
    vecs = embed_texts([text])
    if not vecs:
        return None
    return vecs[0]
