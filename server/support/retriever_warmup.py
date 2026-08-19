# -*- coding: utf-8 -*-
"""检索模型预热(BGE-m3 + Reranker)。

启动时后台预热,避免用户首次检索时等待。
"""
import os
import sys
import threading

# 路径:为了 import RAG 的 embed/config
_HERE = os.path.dirname(os.path.abspath(__file__))            # server/support/
_PROJECT = os.path.dirname(os.path.dirname(_HERE))            # project root
_RAG = os.path.join(_PROJECT, "RAG")
_CONFIG = os.path.join(_PROJECT, "config")
for _p in (_PROJECT, _RAG, _CONFIG):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import embed

# ==================== 检索模型懒加载(线程安全)====================
_load_lock = threading.Lock()
_retriever_ready = False

def ensure_retriever():
    """预热文本检索模型(BGE-m3)+ 重排模型(BGE-reranker-v2-m3)。"""
    global _retriever_ready
    if _retriever_ready:
        return True
    with _load_lock:
        if _retriever_ready:
            return True
        try:
            print("[retriever] 首次加载 BGE-m3,约 30s...", flush=True)
            embed.get_text_encoder()
            print("[retriever] BGE-m3 就绪", flush=True)

            print("[retriever] 加载 Reranker(BGE-reranker-v2-m3),约 10s...", flush=True)
            embed.get_reranker()
            print("[retriever] Reranker 就绪", flush=True)

            _retriever_ready = True
            return True
        except Exception as e:
            print(f"[retriever] 加载失败: {e}", flush=True)
            return False
