# -*- coding: utf-8 -*-
"""qdrant dense prefetch 确定性探针:金块排名是否跨调用抖动 + HNSW/量化配置。"""
import sys
sys.path.insert(0, "/root/autodl-tmp/app/config")
sys.path.insert(0, "/root/autodl-tmp/app")
sys.path.insert(0, "/root/autodl-tmp/app/RAG")

import json, urllib.request
import numpy as np
import config as C
from qdrant_client import QdrantClient
from RAG.embed import get_text_encoder

qc = QdrantClient(url="http://127.0.0.1:6333", timeout=90)
enc = get_text_encoder()
GOLD = "ba5cd3d9-4ffa-53e7-8614-fa79e462d1b1"
GOLD_PID = GOLD  # 已是 uuid5 点 id

info = json.loads(urllib.request.urlopen("http://127.0.0.1:6333/collections/ald_text").read())
cfg = info["result"]["config"]
print("hnsw_config:", json.dumps(cfg.get("hnsw_config", {})))
print("quantization:", json.dumps(cfg.get("quantization_config", {}))[:300])

EN = "For a grating with a groove density of 600 grooves/mm, when 500 nm light is incident perpendicularly (alpha=0), what is the diffraction angle for the first-order diffraction (m=1)?"
_r = enc.encode([EN])
qv = np.asarray(_r[0][0] if isinstance(_r, tuple) else _r[0], dtype=float).ravel()

for i in range(5):
    hits = qc.query_points(collection_name=C.TEXT_COLLECTION, query=qv.tolist(),
                           using="dense", limit=80, with_payload=False).points
    rank = next((j + 1 for j, h in enumerate(hits) if str(h.id) == GOLD_PID), None)
    top1 = hits[0].score
    print(f"dense80#{i+1}: 金块排名 {rank or '不在'}  top1={top1:.4f}  命中数={len(hits)}")
