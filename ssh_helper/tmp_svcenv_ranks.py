# -*- coding: utf-8 -*-
"""服务环境编码器下:金块在 主路/翻译路 的 dense 排名 + query_text 池归属。"""
import sys
sys.path.insert(0, "/root/autodl-tmp/app/config")
sys.path.insert(0, "/root/autodl-tmp/app")
sys.path.insert(0, "/root/autodl-tmp/app/RAG")

import numpy as np
import config as C
from qdrant_client import QdrantClient, models
from mcp_servers.retrieval import query as Q
from mcp_servers.retrieval.engine_api import _recall_queries

qc = QdrantClient(url=C.QDRANT_URL, timeout=90)
GOLD = "ba5cd3d9-4ffa-53e7-8614-fa79e462d1b1"
ZH = "一片刻线密度 600 grooves/mm 的光栅，500 nm 的光垂直入射（α=0°），取一级衍射（m=1），衍射角是多少度？"

qs = _recall_queries(ZH)
print("召回查询组:", [q[:50] for q in qs])

def dense_rank(q):
    Q._te = Q._te or __import__("embed").get_text_encoder()
    dense, _ = Q._te.encode([q])
    hits = qc.query_points(collection_name=C.TEXT_COLLECTION,
                           query=dense[0].tolist(), using="dense", limit=200,
                           with_payload=False).points
    r = next((i + 1 for i, h in enumerate(hits) if str(h.id) == GOLD), None)
    return r, hits[0].score, dense[0].tolist()[:2]

for q in qs:
    r, top1, head = dense_rank(q)
    inq = Q.query_text(q, k=96)
    ids = [(p.payload or {}).get("chunk_id") for p in inq]
    print(f"[{q[:36]}...] dense200排名={r or '>200'} top1={top1:.4f} vec头={head} "
          f"query_text池: {'第'+str(ids.index(GOLD)+1) if GOLD in ids else '不在'}")
