# -*- coding: utf-8 -*-
"""检索微服务:独立进程,BGE-m3 + CLIP + Reranker + Qdrant。

Agent 通过 HTTP 调用本服务执行检索,超时=关 HTTP 连接,无僵尸线程。
启动时后台加载模型,首次请求可能稍慢(等模型加载完)。
"""
import os
import sys
import threading
import logging

# ---- 路径设置(与 _common.py 一致)----
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
_RAG = os.path.join(_PROJECT, "RAG")
_CONFIG = os.path.join(_PROJECT, "config")
for _p in (_PROJECT, _RAG, _CONFIG):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("retrieval")

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

import config as C
import embed
import query as Q
from tools._common import _pid, _text_dict, _image_dict, _collection_hint

app = FastAPI(title="检索微服务")

# ---- 模型(懒加载,首次请求触发)----
_te = None  # BGE-m3
_ie = None  # CLIP
_re = None  # Reranker
_model_lock = threading.Lock()  # 懒加载互斥:并发首请求不重复加载模型


# ---- 请求模型 ----
class SearchTextRequest(BaseModel):
    query: str
    k: int = 3
    score_ratio: float = 0.6

class SearchImageRequest(BaseModel):
    query: str
    k: int = 3
    include_portraits: bool = False
    score_ratio: float = 0.6

class GetChunkRequest(BaseModel):
    chunk_id: str


# ---- 辅助函数(从 search_tools.py 迁移)----
def _rerank(query, points, k=3, text_fn=None):
    if len(points) <= 1:
        return [(p, p.score) for p in points]
    try:
        global _re
        if _re is None:
            # 加锁:启动初期并发首请求会各自触发加载(数 GB 内存/数分钟)
            with _model_lock:
                if _re is None:
                    _re = embed.get_reranker()
        documents = (
            [text_fn(p) for p in points] if text_fn
            else [(p.payload or {}).get("content", "") for p in points]
        )
        scores = _re.rerank(query, documents)
        ranked = sorted(zip(points, scores), key=lambda x: x[1], reverse=True)
        return [(p, s) for p, s in ranked[:k]]
    except Exception as e:
        logger.warning(f"重排失败,回退原始顺序: {e}")
        return [(p, p.score) for p in points[:k]]


def _image_text(p):
    pl = p.payload or {}
    parts = [pl.get("caption", ""), pl.get("description", "")]
    text = "\n".join(s for s in parts if s).strip()
    return text or "[无描述]"


# ---- API 端点 ----
@app.post("/search_text")
def search_text(req: SearchTextRequest):
    """文本库混合检索(dense+sparse RRF)+ Cross-Encoder 重排。"""
    recall_k = getattr(C, "RERANK_RECALL_K", 20)
    points = Q.query_text(req.query, k=recall_k)
    if not points:
        return []
    top_score = points[0].score
    filtered = [p for p in points if p.score >= top_score * req.score_ratio]
    ranked = _rerank(req.query, filtered, k=req.k)
    return [_text_dict(p.payload or {}, score=s) for p, s in ranked]


@app.post("/search_image")
def search_image(req: SearchImageRequest):
    """图像库检索(CLIP+BGE-m3 双路 RRF)+ Cross-Encoder 重排。"""
    recall_k = getattr(C, "RERANK_RECALL_K", 20)
    points = Q.query_image(req.query, k=recall_k, include_portraits=req.include_portraits)
    if not points:
        return []
    top_score = points[0].score
    filtered = [p for p in points if p.score >= top_score * req.score_ratio]
    ranked = _rerank(req.query, filtered, k=req.k, text_fn=_image_text)
    return [_image_dict(p.payload or {}, score=s) for p, s in ranked]


@app.post("/get_chunk")
def get_chunk(req: GetChunkRequest):
    """按 chunk_id 取完整块/图。"""
    client = Q.get_client()
    pid = _pid(req.chunk_id)
    hint = _collection_hint(req.chunk_id)
    if hint is not None:
        other = C.IMAGE_COLLECTION if hint == C.TEXT_COLLECTION else C.TEXT_COLLECTION
        order = [hint, other]
    else:
        order = [C.TEXT_COLLECTION, C.IMAGE_COLLECTION]
    for coll in order:
        try:
            hits = client.retrieve(coll, ids=[pid], with_payload=True, with_vectors=False)
        except Exception:
            hits = []
        if hits:
            pl = hits[0].payload or {}
            return _text_dict(pl) if coll == C.TEXT_COLLECTION else _image_dict(pl)
    return None


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    # 后台预热模型
    def _warmup():
        try:
            logger.info("后台预热 BGE-m3 + Reranker...")
            embed.get_text_encoder()
            embed.get_reranker()
            logger.info("预热完成")
        except Exception as e:
            logger.warning(f"预热失败: {e}")

    threading.Thread(target=_warmup, daemon=True).start()

    print("检索微服务  http://127.0.0.1:8002", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=8002, log_level="info")
