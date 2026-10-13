# -*- coding: utf-8 -*-
"""对照:直接编码+手发 prefetch vs query_text 实发参数(向量指纹)。"""
import sys
sys.path.insert(0, "/root/autodl-tmp/app/config")
sys.path.insert(0, "/root/autodl-tmp/app")
sys.path.insert(0, "/root/autodl-tmp/app/RAG")

import numpy as np
import config as C
from qdrant_client import models
from mcp_servers.retrieval import query as Q

EN = "For a grating with a groove density of 600 grooves/mm, when 500 nm light is incident perpendicularly (alpha=0), what is the diffraction angle for the first-order diffraction (m=1)?"
GOLD = "ba5cd3d9-4ffa-53e7-8614-fa79e462d1b1"

def fused(prefetch, limit=96):
    r = Q.get_client().query_points(
        collection_name=C.TEXT_COLLECTION, prefetch=prefetch,
        query=models.FusionQuery(fusion=models.Fusion.RRF), limit=limit, with_payload=False)
    ids = [(p.payload or {}).get("chunk_id") for p in r.points]
    return ids

# 1) 直接编码 + 手发
Q._te = Q._te or __import__("embed").get_text_encoder()
dense, sparse = Q._te.encode([EN])
sp = sparse[0]
pf1 = [
    models.Prefetch(query=dense[0].tolist(), using="dense", limit=80),
    models.Prefetch(query=models.SparseVector(indices=[int(x) for x in sp.keys()],
                                              values=[float(x) for x in sp.values()]),
                    using="sparse", limit=80),
]
ids1 = fused(pf1)
print("① 直接编码+手发: 金块", ids1.index(GOLD) + 1 if GOLD in ids1 else "不在",
      " dense[0][:2]=", [round(x, 6) for x in dense[0].tolist()[:2]],
      " sparse_keys=", len(sp))

# 2) query_text + 捕获实参
cap = {}
orig = Q.get_client().query_points
def spy(cn, **kw):
    cap["kw"] = kw
    return orig(cn, **kw)
Q.get_client().query_points = spy
ids2 = Q.query_text(EN, k=96)
Q.get_client().query_points = orig
ids2 = [(p.payload or {}).get("chunk_id") for p in ids2]
print("② query_text: 金块", ids2.index(GOLD) + 1 if GOLD in ids2 else "不在")

kw = cap["kw"]
pf2 = kw["prefetch"]
pd = [p for p in pf2 if p.using == "dense"][0]
ps = [p for p in pf2 if p.using == "sparse"][0]
v1 = np.asarray(dense[0], dtype=float).ravel()
v2 = np.asarray(pd.query, dtype=float).ravel()
cos = float(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))
sk1 = set(int(x) for x in sp.keys())
sk2 = set(int(x) for x in ps.query.indices)
print("③ 实参指纹: dense cos=", round(cos, 6),
      " sparse keys相等=", sk1 == sk2, " ②limit=", kw.get("limit"),
      " sparse差异项=", len(sk1 ^ sk2))

# 3) 用捕获实参原样重发
ids3 = fused(pf2)
print("④ 捕获实参重发: 金块", ids3.index(GOLD) + 1 if GOLD in ids3 else "不在")
