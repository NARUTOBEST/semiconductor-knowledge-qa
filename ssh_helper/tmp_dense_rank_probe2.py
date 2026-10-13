# -*- coding: utf-8 -*-
"""双语对照:同一金块,中文原查 vs 英文改写 vs 关键词短语,各自 dense 排名。"""
import sys, re
sys.path.insert(0, "/root/autodl-tmp/app/config")
sys.path.insert(0, "/root/autodl-tmp/app")

import numpy as np
import config as C
from qdrant_client import QdrantClient
from RAG.embed import get_text_encoder

CASES = [
    ("dev_043", "8d432e70-4992-5873-b3ab-e1d4693edf81", [
        ("中文原题", "这台机器人的晶圆放置重复精度（placement repeatability）标称多少？"),
        ("英文改写", "What is the placement repeatability specification of the robot? repeatability at appropriate speeds"),
        ("关键词", "placement repeatability 0.1mm TIR Magnatran"),
    ]),
    ("dev_198", "ba5cd3d9-4ffa-53e7-8614-fa79e462d1b1", [
        ("中文原题", "一片刻线密度 600 grooves/mm 的光栅，500 nm 的光垂直入射（α=0°），取一级衍射（m=1），衍射角是多少度？"),
        ("英文改写", "What is the diffraction angle from the grating equation m lambda = d (sin alpha + sin beta)?"),
        ("关键词", "grating equation diffraction angle grooves monochromatic light incident"),
    ]),
]

qc = QdrantClient(url="http://127.0.0.1:6333", timeout=90)
enc = get_text_encoder()
ALL = 399685

for qid, gid, variants in CASES:
    gp = qc.retrieve(collection_name=C.TEXT_COLLECTION, ids=[gid], with_vectors=True, with_payload=True)
    gv = np.asarray(gp[0].vector["dense"], dtype=float)
    ng = np.linalg.norm(gv)
    print(f"===== {qid} 金块", flush=True)
    print("  payload:", repr((gp[0].payload or {}).get("content", ""))[:220], flush=True)
    for label, q in variants:
        _r = enc.encode([q])
        qv = np.asarray(_r[0][0] if isinstance(_r, tuple) else _r[0], dtype=float).ravel()
        sim = float(qv @ gv / (np.linalg.norm(qv) * ng + 1e-9))
        hits = qc.query_points(collection_name=C.TEXT_COLLECTION, query=qv.tolist(),
                               using="dense", limit=96, with_payload=False).points
        rank = next((i + 1 for i, h in enumerate(hits) if h.id == gp[0].id), None)
        print(f"  [{label}] cos={sim:.4f} dense排名={rank or '>96'}  top1={hits[0].score:.4f}", flush=True)
