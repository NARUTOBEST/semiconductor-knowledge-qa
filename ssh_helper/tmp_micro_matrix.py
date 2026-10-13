# -*- coding: utf-8 -*-
"""微测:payload 开关/单路/大 prefetch/两阶段 ef,定位两阶段融合丢点条件。"""
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
vec = dense[0].tolist()
sv = models.SparseVector(indices=[int(x) for x in sparse[0].keys()],
                         values=[float(x) for x in sparse[0].values()])
print("sparse维度:", len(sv.indices), " dense头:", [round(x, 6) for x in vec[:2]])

def run(label, **kw):
    r = qc.query_points(collection_name=C.TEXT_COLLECTION, **kw)
    ids = [str(p.id) for p in r.points]
    pos = ids.index(GOLD) + 1 if GOLD in ids else None
    print(f"{label}: 返回{len(ids)} 金块", pos or "不在")

run("1 dense直查96(payload=F)", query=vec, using="dense", limit=96, with_payload=False)
run("2 dense-only pf80+RRF96(F)", prefetch=[models.Prefetch(query=vec, using="dense", limit=80)],
    query=models.FusionQuery(fusion=models.Fusion.RRF), limit=96, with_payload=False)
run("3 sparse-only pf80+RRF96(F)", prefetch=[models.Prefetch(query=sv, using="sparse", limit=80)],
    query=models.FusionQuery(fusion=models.Fusion.RRF), limit=96, with_payload=False)
run("4 双prefetch+RRF96(F)", prefetch=[models.Prefetch(query=vec, using="dense", limit=80),
    models.Prefetch(query=sv, using="sparse", limit=80)],
    query=models.FusionQuery(fusion=models.Fusion.RRF), limit=96, with_payload=False)
run("5 双prefetch+RRF96(T)", prefetch=[models.Prefetch(query=vec, using="dense", limit=80),
    models.Prefetch(query=sv, using="sparse", limit=80)],
    query=models.FusionQuery(fusion=models.Fusion.RRF), limit=96, with_payload=True)
run("6 双prefetch200+RRF250(F)", prefetch=[models.Prefetch(query=vec, using="dense", limit=200),
    models.Prefetch(query=sv, using="sparse", limit=200)],
    query=models.FusionQuery(fusion=models.Fusion.RRF), limit=250, with_payload=False)
run("7 双prefetch80+RRF192(F)", prefetch=[models.Prefetch(query=vec, using="dense", limit=80),
    models.Prefetch(query=sv, using="sparse", limit=80)],
    query=models.FusionQuery(fusion=models.Fusion.RRF), limit=192, with_payload=False)
