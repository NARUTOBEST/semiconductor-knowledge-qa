# -*- coding: utf-8 -*-
"""服务环境编码器一致性:同文本多次编码,向量指纹与金块排名是否稳定。"""
import sys
sys.path.insert(0, "/root/autodl-tmp/app/config")
sys.path.insert(0, "/root/autodl-tmp/app")
sys.path.insert(0, "/root/autodl-tmp/app/RAG")

import numpy as np
import config as C
import embed
from qdrant_client import QdrantClient

qc = QdrantClient(url=C.QDRANT_URL, timeout=90)
enc = embed.get_text_encoder()
GOLD = "ba5cd3d9-4ffa-53e7-8614-fa79e462d1b1"
EN = "For a grating with a groove density of 600 grooves/mm, when 500 nm light is incident perpendicularly (α=0°), what is the diffraction angle for the first-order diffraction (m=1)?"
ZH = "一片刻线密度 600 grooves/mm 的光栅，500 nm 的光垂直入射（α=0°），取一级衍射（m=1），衍射角是多少度？"

ref = None
for i in range(5):
    dense, _ = enc.encode([EN])
    v = np.asarray(dense[0], dtype=float).ravel()
    h = [round(x, 6) for x in v[:2]]
    same = "同" if ref is None or np.allclose(v, ref, atol=1e-4) else "异"
    if ref is None:
        ref = v
    hits = qc.query_points(collection_name=C.TEXT_COLLECTION, query=v.tolist(),
                           using="dense", limit=96, with_payload=False).points
    r = next((j + 1 for j, p in enumerate(hits) if str(p.id) == GOLD), None)
    print(f"encode#{i+1}: 指纹={h} 与首编{same} 金块dense96排名={r or '不在'}")

# 批量编码 vs 单条编码
dense2, _ = enc.encode([ZH, EN])
v2 = np.asarray(dense2[1], dtype=float).ravel()
print("批量[ZH,EN]中EN指纹=", [round(x, 6) for x in v2[:2]],
      " 与单条allclose=", bool(np.allclose(v2, ref, atol=1e-4)))
