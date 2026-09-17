# -*- coding: utf-8 -*-
"""检索就绪探测。

模型(BGE-m3 / CLIP / Reranker)与 Qdrant 全部在检索微服务(8002)进程内,主服务
**不再加载任何模型**。启动时后台轮询微服务 /health,等它把模型预热完成,避免
首个用户请求撞上模型冷启动(约 30s)。

历史:本模块曾在主服务进程内 import embed 预热 BGE-m3/Reranker,造成两个进程
各加载一套数 GB 模型。现统一收进微服务。
"""
import os
import sys
import threading
import time

# 路径:为了 import RAG 的 embed_http/config
_HERE = os.path.dirname(os.path.abspath(__file__))            # server/support/
_PROJECT = os.path.dirname(os.path.dirname(_HERE))            # project root
_RAG = os.path.join(_PROJECT, "RAG")
_CONFIG = os.path.join(_PROJECT, "config")
for _p in (_PROJECT, _RAG, _CONFIG):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_ready = False
_lock = threading.Lock()


def ensure_retriever(timeout: float = 180.0) -> bool:
    """等待检索微服务就绪(后台预热模型完成)。就绪返回 True,超时返回 False。

    多次调用幂等:已就绪立即返回;并发调用由锁保证只探测一次。
    """
    global _ready
    if _ready:
        return True
    with _lock:
        if _ready:
            return True
        try:
            import embed_http
            print("[retriever] 等待检索微服务(8002)预热模型...", flush=True)
            ok = embed_http.wait_until_ready(timeout=timeout)
            if ok:
                _ready = True
                print("[retriever] 检索微服务就绪", flush=True)
            else:
                print(f"[retriever] 等待微服务就绪超时({timeout:.0f}s),"
                      f"首个请求可能较慢或失败", flush=True)
            return ok
        except Exception as e:
            print(f"[retriever] 探测微服务失败: {e}", flush=True)
            return False
