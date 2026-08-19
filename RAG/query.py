# -*- coding: utf-8 -*-
"""检索验证:文本库混合检索(dense+sparse RRF)+ 图像库(CLIP 跨模态)。"""
import os, sys
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from qdrant_client import QdrantClient, models

import config as C
import embed

_client = None
_te = None
_ie = None


def get_client():
    global _client
    if _client is None:
        if getattr(C, "QDRANT_URL", ""):
            _client = QdrantClient(url=C.QDRANT_URL)
        else:
            _client = QdrantClient(path=C.QDRANT_PATH)
    return _client


def query_text(q, k=5):
    """文本库混合检索:dense + sparse,RRF 融合。"""
    global _te
    _te = _te or embed.get_text_encoder()
    dense, sparse = _te.encode([q])
    sp = sparse[0]
    idx = [int(x) for x in sp.keys()]
    val = [float(x) for x in sp.values()]
    res = get_client().query_points(
        collection_name=C.TEXT_COLLECTION,
        prefetch=[
            models.Prefetch(query=dense[0].tolist(), using="dense", limit=20),
            models.Prefetch(query=models.SparseVector(indices=idx, values=val),
                            using="sparse", limit=20),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=k, with_payload=True,
    )
    return res.points


def query_image(q, k=5, include_portraits=False):
    """图像库:CLIP(文本->图) + BGE-m3(文本->描述) 双路 RRF 融合。
    dense=CLIP 图向量;desc_dense=BGE-m3 编的 description+caption 文本向量。
    默认过滤掉 content_type=portrait 的人像(tag_portraits.py 打的标签);
    设 include_portraits=True 可放回人像。"""
    global _ie, _te
    _ie = _ie or embed.get_image_encoder()
    _te = _te or embed.get_text_encoder()
    clip_vec = _ie.encode_text([q])[0]
    bge_dense, _ = _te.encode([q])
    qfilter = None if include_portraits else models.Filter(
        must_not=[models.FieldCondition(key="content_type",
                                        match=models.MatchValue(value="portrait"))]
    )
    res = get_client().query_points(
        collection_name=C.IMAGE_COLLECTION,
        prefetch=[
            models.Prefetch(query=clip_vec.tolist(), using="dense", limit=20, filter=qfilter),
            models.Prefetch(query=bge_dense[0].tolist(), using="desc_dense", limit=20, filter=qfilter),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=k, with_payload=True,
    )
    return res.points


def _show_text(points):
    for i, p in enumerate(points, 1):
        pl = p.payload
        print(f"\n[{i}] score={p.score:.4f}  {pl.get('source_stem','')} p{pl.get('page_start','?')}-{pl.get('page_end','?')}")
        print(f"    heading: {pl.get('heading_path','')}")
        c = (pl.get('content') or '').replace('\n', ' ')
        print(f"    content: {c[:200]}{'...' if len(c)>200 else ''}")
        if pl.get('image_paths'):
            print(f"    images: {pl['image_paths']}")


def _show_image(points):
    for i, p in enumerate(points, 1):
        pl = p.payload
        print(f"\n[{i}] score={p.score:.4f}  {pl.get('source_stem','')} p{pl.get('page_num','?')} ({pl.get('item_type','')})")
        print(f"    image: {os.path.basename(pl.get('image_path',''))}")
        if pl.get('caption'):
            print(f"    caption: {pl['caption'][:100]}")
        if pl.get('description'):
            print(f"    desc: {pl['description'][:150]}")


if __name__ == "__main__":
    # 样例查询(中英 + 术语)
    queries = [
        "Savannah 200 ALD system",          # 设备型号
        "TMA 前驱体",                        # 化学术语
        "wafer chuck 温度控制",              # 工艺
        "ALD 原子层沉积原理",                # 概念
    ]
    mode = sys.argv[1] if len(sys.argv) > 1 else "text"
    for q in queries:
        print(f"\n{'='*70}\n查询: {q}  [{mode}库]")
        try:
            pts = query_text(q, k=5) if mode == "text" else query_image(q, k=5)
            if mode == "text":
                _show_text(pts)
            else:
                _show_image(pts)
        except Exception as e:
            print(f"  查询失败: {e}")
