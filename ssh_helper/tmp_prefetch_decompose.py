# -*- coding: utf-8 -*-
"""两阶段查询拆解:dense-only prefetch / 加大 limit / hnsw_ef,定位金块在哪一步丢失。"""
import sys
sys.path.insert(0, "/root/autodl-tmp/app/config")
sys.path.insert(0, "/root/autodl-tmp/app")
sys.path.insert(0, "/root/autodl-tmp/app/RAG")

import config as C
from qdrant_client import QdrantClient, models
from mcp_servers.retrieval import query as Q

qc = QdrantClient(url=C.QDRANT_URL, timeout=90)
Q._te = Q._te or __import__("embed").get_text_encoder()
GOLD = "ba5cd3d9-4ffa-53e7-8614-fa79e462d1b1"
EN = "For a grating with a groove density of 600 grooves/mm, when 500 nm light is incident perpendicularly (α=0°), what is the diffraction angle for the first-order diffraction (m=1)?"

dense, sparse = Q._te.encode([EN])
sp = sparse[0]
vec = dense[0].tolist()
sv = models.SparseVector(indices=[int(x) for x in sp.keys()], values=[float(x) for x in sp.values()])

def run(label, **kw):
    r = qc.query_points(collection_name=C.TEXT_COLLECTION, with_payload=False, **kw)
    ids = [str(p.id) for p in r.points]
    print(f"{label}: 返回{len(ids)} 金块位置:", ids.index(GOLD) + 1 if GOLD in ids else "不在")

# A. 裸 dense 直查(参照,已知第14)
run("A 裸dense直查200", query=vec, using="dense", limit=200)
# B. 两阶段:dense-only prefetch 80 + RRF
run("B 两阶段dense80+RRF96", prefetch=[models.Prefetch(query=vec, using="dense", limit=80)],
    query=models.FusionQuery(fusion=models.Fusion.RRF), limit=96)
# C. 两阶段:dense-only prefetch 200 + RRF
run("C 两阶段dense200+RRF250", prefetch=[models.Prefetch(query=vec, using="dense", limit=200)],
    query=models.FusionQuery(fusion=models.Fusion.RRF), limit=250)
# D. 两阶段:dense+sparse(复现生产)
run("D 两阶段dense80+sparse80+RRF96", prefetch=[models.Prefetch(query=vec, using="dense", limit=80),
    models.Prefetch(query=sv, using="sparse", limit=80)],
    query=models.FusionQuery(fusion=models.Fusion.RRF), limit=96)
# E. 同 D 但 prefetch 200
run("E 两阶段dense200+sparse200+RRF250", prefetch=[models.Prefetch(query=vec, using="dense", limit=200),
    models.Prefetch(query=sv, using="sparse", limit=200)],
    query=models.FusionQuery(fusion=models.Fusion.RRF), limit=250)
# F. prefetch 带 hnsw_ef=512
run("F dense80 hnsw_ef512+RRF96", prefetch=[models.Prefetch(query=vec, using="dense", limit=80, hnsw_ef=512)],
    query=models.FusionQuery(fusion=models.Fusion.RRF), limit=96)
