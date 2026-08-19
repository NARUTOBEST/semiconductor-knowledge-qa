# -*- coding: utf-8 -*-
"""自拟问题检测检索质量:报 Qdrant 入库状态 + 跑一组多样化问题(文本库/图像库 top3)。"""
import os, sys
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
_HERE = os.path.dirname(os.path.abspath(__file__))
_RAG = os.path.dirname(_HERE)
_PROJECT = os.path.dirname(_RAG)
_CONFIG = os.path.join(_PROJECT, "config")
for _p in (_CONFIG, _RAG):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import config as C
import query as Q

client = Q.get_client()

print("=== Qdrant 入库状态 ===")
for name in (C.TEXT_COLLECTION, C.IMAGE_COLLECTION):
    try:
        cnt = client.count(name, exact=True).count
        print(f"  {name}: {cnt} 个点")
    except Exception as e:
        print(f"  {name}: 读取失败 {e}")
try:   # 图像库有 LLM 描述的比例
    with_desc = total = 0; off = None
    while True:
        res, off = client.scroll(C.IMAGE_COLLECTION, limit=256, offset=off,
                                 with_payload=True, with_vectors=False)
        for p in res:
            total += 1
            if (p.payload.get("description") or "").strip():
                with_desc += 1
        if not off:
            break
    print(f"  图像块有 LLM 描述: {with_desc}/{total}")
except Exception as e:
    print(f"  描述统计失败: {e}")

TEXT_QS = [
    "FIJI 200 系统的急停按钮在哪里",          # 设备细节
    "为什么 TMA 接触空气会燃烧",              # 化学机理
    "如何装载 recipe 并运行 ALD 工艺",        # 操作流程
    "Oxford OpAL ALD 设备在哪个房间",         # 事实位置
    "ALD 自限制 self-limiting 反应机理",      # 概念+跨语言
    "更换 O-ring 密封圈的维护周期",           # 维护
]
IMAGE_QS = [
    "ALD 工艺原理示意图",                     # 示意图
    "设备外观实物照片",                       # 设备照(原 CLIP 误中肖像)
    "前驱体饱和曲线图",                       # 曲线图(原 CLIP 漏中)
    "FIJI 急停按钮 EMO",                      # 测描述匹配(EMO 在图描述里)
    "Oxford OpAL 房间位置",                   # 测图注匹配(房间在 caption 里)
]

print("\n=== 文本库检索(自拟问题, top 3) ===")
for q in TEXT_QS:
    print(f"\nQ: {q}")
    try:
        for i, p in enumerate(Q.query_text(q, k=3), 1):
            pl = p.payload
            c = (pl.get("content") or "").replace("\n", " ")
            print(f"  [{i}] {p.score:.4f}  {pl.get('source_stem','')[:32]} p{pl.get('page_start','?')}")
            print(f"      标题: {pl.get('heading_path','')[:64]}")
            print(f"      内容: {c[:96]}")
    except Exception as e:
        print(f"  失败: {e}")

print("\n=== 图像库检索(自拟问题, top 3) ===")
for q in IMAGE_QS:
    print(f"\nQ: {q}")
    try:
        for i, p in enumerate(Q.query_image(q, k=3), 1):
            pl = p.payload
            d = (pl.get("description") or "").replace("\n", " ")
            print(f"  [{i}] {p.score:.4f}  {pl.get('source_stem','')[:32]} p{pl.get('page_num','?')} ({pl.get('item_type','')})")
            if pl.get("caption"):
                print(f"      caption: {pl['caption'][:64]}")
            print(f"      desc: {d[:88]}")
    except Exception as e:
        print(f"  失败: {e}")
