# -*- coding: utf-8 -*-
"""健康检查业务逻辑:探测关键依赖是否可用。

安全:健康端点无需认证(供存活/就绪探针调用),故**不得**在响应里回带异常
文本(可能泄露内部地址/凭据/路径)。依赖不可用时只回 "error",细节记日志。

注意:本进程(主服务 8001)**不得**直接打开 Qdrant 本地库 —— qdrant 本地文件
模式用排他文件锁(portalocker EXCLUSIVE),检索微服务(8002)已持有该锁,
第二进程打开会直接抛 LockException。向量库/嵌入模型的可用性一律通过
HTTP 探测检索微服务获得。
"""
import logging

import httpx

import config as C

logger = logging.getLogger("health")


def check_qdrant():
    """检查向量库是否可用:经检索微服务(8002)的 /health 探测。

    微服务持有 Qdrant 连接与嵌入模型,它健康即代表检索链路可用。
    不在本进程直接开 QdrantClient(会与微服务抢本地库文件锁)。
    """
    try:
        base = getattr(C, "RETRIEVAL_SERVICE_URL", "http://127.0.0.1:8002")
        resp = httpx.get(base.rstrip("/") + "/health", timeout=5)
        if resp.status_code == 200:
            return "ok"
        logger.warning("health check retrieval service returned %s", resp.status_code)
        return "error"
    except Exception as e:
        logger.warning("health check retrieval service failed: %s: %s", type(e).__name__, e)
        return "error"


def check_llm():
    """检查 LLM 客户端是否可初始化(不发实际推理请求)。"""
    try:
        from chat.service import get_client
        get_client()
        return "ok"
    except Exception as e:
        logger.warning("health check llm failed: %s: %s", type(e).__name__, e)
        return "error"
