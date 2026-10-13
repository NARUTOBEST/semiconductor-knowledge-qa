# -*- coding: utf-8 -*-
"""设备/指纹/同进程对照:query_text vs 手发,打印编码器真实 device。"""
import sys
sys.path.insert(0, "/root/autodl-tmp/app/config")
sys.path.insert(0, "/root/autodl-tmp/app")
sys.path.insert(0, "/root/autodl-tmp/app/RAG")

import numpy as np
import config as C
from mcp_servers.retrieval import query as Q

print("EMBED_DEVICE=", getattr(C, "EMBED_DEVICE", None))
Q._te = Q._te or __import__("embed").get_text_encoder()
try:
    print("encoder device=", next(Q._te.model.model.parameters()).device)
except Exception as e:
    print("device introspect fail:", e)

EN = "For a grating with a groove density of 600 grooves/mm, when 500 nm light is incident perpendicularly (α=0°), what is the diffraction angle for the first-order diffraction (m=1)?"
GOLD = "ba5cd3d9-4ffa-53e7-8614-fa79e462d1b1"
dense, sparse = Q._te.encode([EN])
v = np.asarray(dense[0], dtype=float).ravel()
print("指纹=", [round(x, 6) for x in v[:2]])

pts = Q.query_text(EN, k=96)
ids = [(p.payload or {}).get("chunk_id") for p in pts]
print("query_text: 金块", ids.index(GOLD) + 1 if GOLD in ids else "不在")

from qdrant_client import models
pf = [models.Prefetch(query=dense[0].tolist(), using="dense", limit=80),
      models.Prefetch(query=models.SparseVector(indices=[int(x) for x in sparse[0].keys()],
                                                values=[float(x) for x in sparse[0].values()]),
                      using="sparse", limit=80)]
r = Q.get_client().query_points(collection_name=C.TEXT_COLLECTION, prefetch=pf,
                                query=models.FusionQuery(fusion=models.Fusion.RRF),
                                limit=96, with_payload=True)
ids2 = [(p.payload or {}).get("chunk_id") for p in r.points]
print("手发同参: 金块", ids2.index(GOLD) + 1 if GOLD in ids2 else "不在")
