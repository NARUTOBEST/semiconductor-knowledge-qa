# -*- coding: utf-8 -*-
"""dev_198 query_text 丢失复现:精确字节对照(原译文含 α/° vs ASCII 版)。"""
import sys
sys.path.insert(0, "/root/autodl-tmp/app/config")
sys.path.insert(0, "/root/autodl-tmp/app")
sys.path.insert(0, "/root/autodl-tmp/app/RAG")

import config as C
from qdrant_client import models
from mcp_servers.retrieval import query as Q
from mcp_servers.retrieval.engine_api import _translate_query

EN_ASCII = "For a grating with a groove density of 600 grooves/mm, when 500 nm light is incident perpendicularly (alpha=0), what is the diffraction angle for the first-order diffraction (m=1)?"
GOLD = "ba5cd3d9-4ffa-53e7-8614-fa79e462d1b1"
ZH = "一片刻线密度 600 grooves/mm 的光栅，500 nm 的光垂直入射（α=0°），取一级衍射（m=1），衍射角是多少度？"

en_real = _translate_query(ZH)
print("服务真实译文 repr:", repr(en_real))

for label, q in [("真实译文", en_real), ("ASCII版", EN_ASCII)]:
    print(f"--- [{label}] len={len(q)}")
    print("    repr:", repr(q))
    ids = Q.query_text(q, k=96)
    ids = [(p.payload or {}).get("chunk_id") for p in ids]
    print("    query_text96 金块位置:", ids.index(GOLD) + 1 if GOLD in ids else "不在")

# 再看服务内 search_text 全链路(带翻译双路)
from mcp_servers.retrieval import engine_api as E
d = E.search_text(ZH, k=4, with_pool=True, with_ranked=True)
pool = [p.get("chunk_id") for p in d.get("pool") or []]
print("search_text 全链路: pool", len(pool), "金块位置:", pool.index(GOLD) + 1 if GOLD in pool else "不在")
