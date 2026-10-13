# -*- coding: utf-8 -*-
"""7 题硬失败全链路探针(GPU 模式跑):走真实检索管线(:8002 search_text,含 scope/翻译双路/
级联重排),看事实块落在哪一关——top-4 内 / 池内被排后 / 池外。
用法: GPU 上 python recall7_pipeline_diag.py(先 source ret.env)"""
import sys, json, re, urllib.request
sys.path.insert(0, "/root/autodl-tmp/app/config")
sys.path.insert(0, "/root/autodl-tmp/app/RAG")
sys.path.insert(0, "/root/autodl-tmp/app")

import config as C
from qdrant_client import QdrantClient

TOKEN = "rt-7f3a9c2e5b1d486fa0e4d3c2b1a09f87"
URL = "http://127.0.0.1:8002/search_text"

# (qid, 查询文本, 事实块判定 regex, 期望来源 stem 前缀)
CASES = [
    ("dev_043", "这台机器人的晶圆放置重复精度（placement repeatability）标称多少？",
     r"Placement Repeatability", "双臂取晶圆机器人magnatran"),
    ("dev_094", "做出来的载带剥离力(Peel Force)不合格,手册说受哪几个因素影响?应该怎么调?",
     r"3 factors.*affect.*peel|peel force", "WAFER  HANDLER Mi30"),
    ("dev_098", "Overspeed 这个报警的判定条件是什么?如果我们伺服电机最高转速是 3000rpm,转到多少就会触发?该怎么处理?",
     r"1\.1 times the maximum speed|Overspeed", "WAFER  HANDLER Mi30"),
    ("dev_114", "砂轮磨损量是怎么通过 GEM 上报给主机的?两个 VID 分别是多少?轮子没做参考(referenced)时上报什么值?",
     r"Variable ID number 139", "晶圆减薄strasbaugh Model 7AF"),
    ("dev_128", "手册里命令参数的单位是怎么约定的？Theta、R、Z 分别用什么单位？",
     r"decimal millidegrees", "双臂取晶圆机器人magnatran"),
    ("dev_033", "手册里对磨削冷却液(grinding coolant)有什么要求?进机的 D.I. 水流量和压力范围又是多少?",
     r"Grinding Coolant: D\.I\. water; 6 GPM", "晶圆减薄strasbaugh Model 7AF"),
    ("dev_198", "一片刻线密度 600 grooves/mm 的光栅，500 nm 的光垂直入射（α=0°），取一级衍射（m=1），衍射角是多少度？",
     r"Grating Equation", "MKS 仪器手册光子技术"),
]

def search(payload):
    req = urllib.request.Request(URL, data=json.dumps(payload).encode(),
                                 headers={"x-internal-token": TOKEN,
                                          "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())

def norm(s):
    return re.sub(r"\s+", "", (s or "").lower())

qc = QdrantClient(url="http://127.0.0.1:6333", timeout=90)

def fact_chunk_ids(stem_pat, fact_pat):
    """在期望文档里找含事实文本的 chunk_id(判定金块)。"""
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
    return set(ids)

for qid, q, fact_pat, stem_pat in CASES:
    gold = fact_chunk_ids(stem_pat, fact_pat)
    d = search({"query": q, "k": 4, "with_pool": True, "with_ranked": True})
    items = d.get("items") or []
    ranked = d.get("ranked") or []
    pool = d.get("pool") or []
    top_ids = [it.get("chunk_id") for it in items]
    rank_ids = [r.get("chunk_id") for r in ranked]
    pool_ids = [p.get("chunk_id") for p in pool]  # pool 是 dict 列表,须先取字段再比对
    hit_top = [g for g in gold if g in top_ids]
    hit_rank = [g for g in gold if g in rank_ids]
    hit_pool = [g for g in gold if g in pool_ids]
    pos_rank = [rank_ids.index(g) + 1 for g in gold if g in rank_ids]
    print(f"===== {qid} 金块{len(gold)}个", flush=True)
    print(f"  top-4命中: {len(hit_top)}  精排序命中: {len(hit_rank)} {pos_rank}  池命中: {len(hit_pool)}", flush=True)
    # 池内未进精排的,查 m3 粗筛名单(with_ranked 里没有的话只能看池总数)
    if hit_pool and not hit_rank:
        print("  !! 金块在召回池但被 m3 粗筛(K=KEEP)筛掉", flush=True)
    if not hit_pool:
        print("  !! 金块根本不在召回池(召回关失败:翻译/sparse/scope)", flush=True)
    for i, it in enumerate(items[:4]):
        print(f"  #{i+1} [{(it.get('source_stem') or '?')[:28]}] {(it.get('content') or '')[:80]}".replace(chr(10), " "), flush=True)
