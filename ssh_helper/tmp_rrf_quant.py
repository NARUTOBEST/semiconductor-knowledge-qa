# -*- coding: utf-8 -*-
"""RRF 稀释量化:dense/sparse 各自排名、重叠度、模拟融合排名 vs query_text 实际。"""
import sys
sys.path.insert(0, "/root/autodl-tmp/app/config")
sys.path.insert(0, "/root/autodl-tmp/app")
sys.path.insert(0, "/root/autodl-tmp/app/RAG")

import config as C
from qdrant_client import QdrantClient, models
from mcp_servers.retrieval import query as Q

qc = QdrantClient(url=C.QDRANT_URL, timeout=90)
Q._te = Q._te or __import__("embed").get_text_encoder()

CASES = [
    ("dev_198", "ba5cd3d9-4ffa-53e7-8614-fa79e462d1b1",
     "For a grating with a groove density of 600 grooves/mm, when 500 nm light is incident perpendicularly (α=0°), what is the diffraction angle for the first-order diffraction (m=1)?"),
    ("dev_043", "8d432e70-4992-5873-b3ab-e1d4693edf81",
     "What is the nominal value of the wafer placement repeatability of this robot?"),
]

def ranks(q, gold):
    dense, sparse = Q._te.encode([q])
    sp = sparse[0]
    d = qc.query_points(collection_name=C.TEXT_COLLECTION, query=dense[0].tolist(),
                        using="dense", limit=200, with_payload=False).points
    s = qc.query_points(collection_name=C.TEXT_COLLECTION,
                        query=models.SparseVector(indices=[int(x) for x in sp.keys()],
                                                  values=[float(x) for x in sp.values()]),
                        using="sparse", limit=200, with_payload=False).points
    dr = {str(p.id): i + 1 for i, p in enumerate(d)}
    sr = {str(p.id): i + 1 for i, p in enumerate(s)}
    d80 = set(list(dr)[:80]); s80 = set(list(sr)[:80])
    ov = len(d80 & s80)
    # 模拟 qdrant RRF: 1/(60+rank),只对 top80 计
    rrf = {}
    for i, p in enumerate(d[:80]): rrf[str(p.id)] = rrf.get(str(p.id), 0) + 1 / (60 + i + 1)
    for i, p in enumerate(s[:80]): rrf[str(p.id)] = rrf.get(str(p.id), 0) + 1 / (60 + i + 1)
    order = sorted(rrf.items(), key=lambda x: -x[1])
    sim = next((i + 1 for i, (pid, _) in enumerate(order) if pid == gold), None)
    # query_text 实际
    pts = Q.query_text(q, k=96)
    ids = [(p.payload or {}).get("chunk_id") for p in pts]
    act = ids.index(gold) + 1 if gold in ids else None
    return dict(dense=dr.get(gold), sparse=sr.get(gold), overlap80=ov,
                cand=len(set(list(dr)[:80]) | s80), sim_fused=sim, actual=act)

for qid, gold, en in CASES:
    r = ranks(en, gold)
    print(f"{qid}: dense200={r['dense'] or '>200'} sparse200={r['sparse'] or '>200'} "
          f"top80重叠={r['overlap80']} 候选={r['cand']} 模拟融合={r['sim_fused'] or '>96'} "
          f"query_text实际={r['actual'] or '不在'}")
