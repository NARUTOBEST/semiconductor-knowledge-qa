# -*- coding: utf-8 -*-
"""嵌入/重排的 HTTP 客户端 —— 主服务(8001)专用,接口与 RAG/embed.py 同构。

模型(BGE-m3 / CLIP / BGE-reranker)统一由检索微服务(8002)持有,主服务进程
**不再加载 torch / 模型**,也不打开 Qdrant 本地库(避免数 GB 内存重复占用与
文件锁冲突)。本模块用 httpx 调微服务的 /embed_text、/rerank,返回与本地
embed.py 相同形状的对象(np.ndarray / list[float]),调用方无需感知差异。

延迟导入 numpy/httpx,纯写库或测试环境不引入重依赖。
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))          # RAG/
_PROJECT = os.path.dirname(_HERE)                           # 项目根
_CONFIG = os.path.join(_PROJECT, "config")
for _p in (_HERE, _PROJECT, _CONFIG):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config as C  # noqa: E402

_TIMEOUT = 60  # 嵌入/重排为 CPU 推理,批量时给足超时


def _base_url():
    return getattr(C, "RETRIEVAL_SERVICE_URL", "http://127.0.0.1:8002").rstrip("/")


def _headers():
    """服务间鉴权:RETRIEVAL_INTERNAL_TOKEN 非空时携带 X-Internal-Token。"""
    tok = getattr(C, "RETRIEVAL_INTERNAL_TOKEN", "")
    return {"X-Internal-Token": tok} if tok else None


class _RemoteTextEncoder:
    """与 embed.TextEncoder 同构:encode(texts) -> (dense np.ndarray (n,1024), sparse list)。"""

    def encode(self, texts, batch_size=12):
        import httpx
        import numpy as np
        resp = httpx.post(
            _base_url() + "/embed_text",
            json={"texts": list(texts)},
            headers=_headers(),
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        dense = np.asarray(data["dense"], dtype=np.float32)
        # sparse: JSON 对象键为字符串,转回 {int: float}
        sparse = [{int(k): float(v) for k, v in (d or {}).items()}
                  for d in data.get("sparse", [])]
        return dense, sparse


class _RemoteReranker:
    """与 embed.Reranker 同构:rerank(query, documents) -> list[float]。"""

    def rerank(self, query, documents, batch_size=8):
        import httpx
        resp = httpx.post(
            _base_url() + "/rerank",
            json={"query": query, "documents": list(documents)},
            headers=_headers(),
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        return [float(s) for s in resp.json()["scores"]]


# ---- 单例(与 embed.py 的 get_* 命名一致,便于直接替换)----
_te = None
_re = None


def get_text_encoder():
    global _te
    if _te is None:
        _te = _RemoteTextEncoder()
    return _te


def get_reranker():
    global _re
    if _re is None:
        _re = _RemoteReranker()
    return _re


def wait_until_ready(timeout: float = 120.0) -> bool:
    """轮询微服务 /health 直到就绪或超时(供启动预热)。"""
    import time
    import httpx
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = httpx.get(_base_url() + "/health", timeout=5)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False
