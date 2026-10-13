# -*- coding: utf-8 -*-
"""dev_043/dev_198 dense 召回关探针:金块在纯 dense top-96 中的排名 + 相似度。"""
import sys, json, re
sys.path.insert(0, "/root/autodl-tmp/app/config")
sys.path.insert(0, "/root/autodl-tmp/app")

import config as C
from qdrant_client import QdrantClient

from RAG.embed import get_text_encoder

CASES = [
    ("dev_043", "这台机器人的晶圆放置重复精度（placement repeatability）标称多少？",
     r"Placement Repeatability", "双臂取晶圆机器人magnatran"),
    ("dev_198", "一片刻线密度 600 grooves/mm 的光栅，500 nm 的光垂直入射（α=0°），取一级衍射（m=1），衍射角是多少度？",
     r"Grating Equation", "MKS 仪器手册光子技术"),
]

qc = QdrantClient(url="http://127.0.0.1:6333", timeout=90)
enc = get_text_encoder()

def gold_ids(stem_pat, fact_pat):
    ids, offset = [], None
    while True:
        pts, offset = qc.scroll(
            collection_name=C.TEXT_COLLECTION, limit=128, offset=offset,
            with_payload=True, with_vectors=False,
            scroll_filter={"must": [{"key": "source_stem", "match": {"text": stem_pat}}]})
        for p in pts:
            pl = p.payload or {}
            if re.search(fact_pat, pl.get("content") or "", re.I | re.S):
                ids.append(pl.get("chunk_id") or str(p.id))
        if offset is None or not pts:
            break
    return ids

import uuid as _u
def _pid(chunk_id):
    return str(_u.uuid5(_u.NAMESPACE_DNS, chunk_id))

for qid, q, fact_pat, stem_pat in CASES:
    golds = [_pid(g) for g in gold_ids(stem_pat, fact_pat)]
    _r = enc.encode([q])
    qv = _r[0][0] if isinstance(_r, tuple) else _r[0]
    import numpy as np
    qv = np.asarray(qv, dtype=float).ravel().tolist()
    hits = qc.query_points(collection_name=C.TEXT_COLLECTION, query=qv,
                           using="dense", limit=96, with_payload=False).points
    print(f"===== {qid} 金块{len(golds)}个 dense-top96:", flush=True)
    rank_map = {h.id: i + 1 for i, h in enumerate(hits)}
    for gid in golds:
        # 金块自身向量相似度
        gp = qc.retrieve(collection_name=C.TEXT_COLLECTION, ids=[gid], with_vectors=True, with_payload=True)
        if not gp:
            print(f"  金块 {gid} 不在库!"); continue
        gv = gp[0].vector.get("dense") if isinstance(gp[0].vector, dict) else gp[0].vector
        dot = sum(a * b for a, b in zip(qv, gv))
        nq = sum(a * a for a in qv) ** 0.5
        ng = sum(a * a for a in gv) ** 0.5
        sim = dot / (nq * ng + 1e-9)
        r = rank_map.get(gp[0].id)
        c = (gp[0].payload or {}).get("content", "")[:100]
        print(f"  dense排名: {r if r else '>96'}  cos={sim:.4f}  chunk={gid}", flush=True)
        print(f"    content: {c}".replace(chr(10), " "), flush=True)
    # top-5 对照
    for i, h in enumerate(hits[:5]):
        pl = qc.retrieve(collection_name=C.TEXT_COLLECTION, ids=[h.id], with_payload=True)
        stem = (pl[0].payload or {}).get("source_stem", "?")[:30] if pl else "?"
        print(f"  dense#{i+1} sim={h.score:.4f} [{stem}]", flush=True)
