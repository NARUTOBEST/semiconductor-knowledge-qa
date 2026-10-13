# -*- coding: utf-8 -*-
"""ret.env 下:spy 捕获 query_text 实参与手发 D 参数逐字段 diff。"""
import sys
sys.path.insert(0, "/root/autodl-tmp/app/config")
sys.path.insert(0, "/root/autodl-tmp/app")
sys.path.insert(0, "/root/autodl-tmp/app/RAG")

import numpy as np
import config as C
from qdrant_client import models
from mcp_servers.retrieval import query as Q

EN = "For a grating with a groove density of 600 grooves/mm, when 500 nm light is incident perpendicularly (α=0°), what is the diffraction angle for the first-order diffraction (m=1)?"
GOLD = "ba5cd3d9-4ffa-53e7-8614-fa79e462d1b1"

# spy 捕获
cap = {}
orig = Q.get_client().query_points
def spy(*a, **kw):
    cap.setdefault("calls", []).append(kw)
    return orig(*a, **kw)
Q.get_client().query_points = spy
pts = Q.query_text(EN, k=96)
Q.get_client().query_points = orig
ids = [(p.payload or {}).get("chunk_id") for p in pts]
print("query_text: 金块", ids.index(GOLD) + 1 if GOLD in ids else "不在", " 调用次数:", len(cap["calls"]))

kw = cap["calls"][-1]
pf = kw["prefetch"]
pd = [p for p in pf if p.using == "dense"][0]
ps = [p for p in pf if p.using == "sparse"][0]
print("query_text实参: limit=", kw.get("limit"), " with_payload=", kw.get("with_payload"),
      " query=", kw.get("query"), " dense_pf.limit=", pd.limit, " sparse_pf.limit=", ps.limit,
      " sparse维度=", len(ps.query.indices))

# 手发 D(重新编码)
dense, sparse = Q._te.encode([EN])
sp = sparse[0]
v1 = np.asarray(dense[0], dtype=float).ravel()
v2 = np.asarray(pd.query, dtype=float).ravel()
print("向量对比: cos=", round(float(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)), 8),
      " 完全相等=", bool(np.allclose(v1, v2, atol=0)))
sk1 = [int(x) for x in sp.keys()]
print("sparse对比: 相等=", sk1 == list(ps.query.indices), " len=", len(sk1), "vs", len(ps.query.indices))

# 手发(同参)结果
r = orig(C.TEXT_COLLECTION, prefetch=pf, query=kw["query"], limit=kw["limit"], with_payload=False)
ids2 = [(p.payload or {}).get("chunk_id") for p in r.points]
print("手发同参: 金块", ids2.index(GOLD) + 1 if GOLD in ids2 else "不在")
