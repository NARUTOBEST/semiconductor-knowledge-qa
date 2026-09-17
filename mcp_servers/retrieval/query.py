# -*- coding: utf-8 -*-
"""Qdrant 向量库查询封装:文本库混合检索(dense+sparse RRF)+ 图像库(CLIP 跨模态)。

位置:mcp_servers/retrieval/ —— 检索工具的底层引擎(向量编码 + Qdrant 召回),
被检索服务(service.py / engine_api.py)、健康检查(health/service.py)、
检索质检脚本(RAG/RAG_tools/quality_check.py)以裸 ``import query`` 调用。
本模块自身做 sys.path 兜底,故可直接 ``python query.py [text|image]`` 运行自测。
"""
import os, re, sys
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# 裸 import config(在 config/)、import embed(在 RAG/):把两者加入 sys.path。
# 生产链由服务入口、测试由 conftest 注入;此处兜底,保证直接脚本运行也可解析。
_HERE = os.path.dirname(os.path.abspath(__file__))                 # mcp_servers/retrieval
_PROJECT = os.path.dirname(os.path.dirname(_HERE))                 # project root
for _p in (os.path.join(_PROJECT, "config"), os.path.join(_PROJECT, "RAG")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

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


# dense/sparse 各召回多少条再进 RRF 融合。调大(20->40)是为了让"稀疏相关但
# 词面命中"的块(如报警代码表行)也能进融合池:实测 4 位报警码块纯 sparse 仅排
# 第 46 名,旧 prefetch=20 时它根本进不了池,精确报警码问题因此检索不到答案。
# dense/sparse 各召回多少条再进 RRF 融合。40->80:加大融合池让重排器有机会
# 看到 RRF 排 20 名以后的 gold 块(R@10≈0.97 说明它们多半在池内,只是被截掉);
# 下游由 RERANK_RECALL_K=32 + 重排 τ 动态截断控制精度,召回加宽不伤 P。
_RECALL_POOL = 80
# 精确码兜底召回时扫描的 sparse 池大小。
_EXACT_SCAN = 200
# 数字串两侧边界:既不能是另一个数字/小数点(避免 24120 含 2412 误命中),
# 也不能是千分位逗号(避免 2,412 这种数字误命中报警码)。
_CODE_BOUND = re.compile(r"(?<![\d.,])\d{3,5}(?![\d.,])")


def extract_codes(q):
    """提取查询中的报警/故障代码(3~5 位独立数字)。过滤常见年份(19xx/20xx)。"""
    codes = []
    for m in _CODE_BOUND.finditer(q or ""):
        d = m.group(0)
        if len(d) == 4 and d[:2] in ("19", "20"):
            continue  # 年份
        codes.append(d)
    return codes


def exact_code_recall(q, limit=6):
    """精确报警/故障代码兜底召回:词面(带数字边界)含查询代码的块。

    报警码表行在 dense/sparse 向量检索里排名靠后(实测 4 位码块纯 sparse 仅第 46
    名,进不了 RRF 池)。这里用 broad sparse 池 + 正则子串兜底,把含该代码的块
    显式捞回,交给服务层并入候选;真实 reranker 启用后仍按相关性重排。
    无代码或无命中返回 []。"""
    codes = extract_codes(q)
    if not codes:
        return []
    global _te
    _te = _te or embed.get_text_encoder()
    _, sparse = _te.encode([q])
    sp = sparse[0]
    res = get_client().query_points(
        collection_name=C.TEXT_COLLECTION,
        query=models.SparseVector(indices=[int(x) for x in sp.keys()],
                                  values=[float(x) for x in sp.values()]),
        using="sparse", limit=_EXACT_SCAN, with_payload=True,
    )
    pat = re.compile(r"(?<![\d.,])(?:" + "|".join(codes) + r")(?![\d.,])")
    hits, seen = [], set()
    for p in res.points:
        c = (p.payload or {}).get("content", "") or ""
        if pat.search(c) and p.id not in seen:
            seen.add(p.id)
            hits.append(p)
            if len(hits) >= limit:
                break
    return hits


def sparse_rescue(q, limit=2):
    """sparse 词面保底召回:cross-encoder 对表格/码表类块常打低分被 τ 截掉
    (run20 探针:垫脚数量表 dense 第10、重排 <0.3,而泛相关段落 0.78+),
    但这类块恰是查询词面命中率最高的。返回纯 sparse top-N 点,由服务层
    按 chunk_id 去重后追加到重排结果尾部(评分压在 τ 之下,只排尾不抢位)。"""
    global _te
    _te = _te or embed.get_text_encoder()
    _, sparse = _te.encode([q])
    sp = sparse[0]
    res = get_client().query_points(
        collection_name=C.TEXT_COLLECTION,
        query=models.SparseVector(indices=[int(x) for x in sp.keys()],
                                  values=[float(x) for x in sp.values()]),
        using="sparse", limit=limit, with_payload=True,
    )
    return res.points


def query_text(q, k=5):
    """文本库混合检索:dense + sparse,RRF 融合(召回池 _RECALL_POOL)。"""
    global _te
    _te = _te or embed.get_text_encoder()
    dense, sparse = _te.encode([q])
    sp = sparse[0]
    idx = [int(x) for x in sp.keys()]
    val = [float(x) for x in sp.values()]
    res = get_client().query_points(
        collection_name=C.TEXT_COLLECTION,
        prefetch=[
            models.Prefetch(query=dense[0].tolist(), using="dense",
                            limit=_RECALL_POOL),
            models.Prefetch(query=models.SparseVector(indices=idx, values=val),
                            using="sparse", limit=_RECALL_POOL),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=max(k, _RECALL_POOL), with_payload=True,
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
