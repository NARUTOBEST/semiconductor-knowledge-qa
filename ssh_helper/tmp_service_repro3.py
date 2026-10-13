# -*- coding: utf-8 -*-
"""服务同款环境(ret.env)下 search_text 全链路 3 连跑:金块池归属是否稳定。"""
import sys
sys.path.insert(0, "/root/autodl-tmp/app/config")
sys.path.insert(0, "/root/autodl-tmp/app")
sys.path.insert(0, "/root/autodl-tmp/app/RAG")

import config as C
from mcp_servers.retrieval import engine_api as E

print("env: RERANK_RECALL_K=", C.RERANK_RECALL_K, " EMBED_DEVICE=", getattr(C, "EMBED_DEVICE", "?"),
      " DEVICE_FILTER=", getattr(C, "DEVICE_FILTER", "?"), " RETRIEVAL_QUERY_TRANSLATE=", C.RETRIEVAL_QUERY_TRANSLATE)

ZH = "一片刻线密度 600 grooves/mm 的光栅，500 nm 的光垂直入射（α=0°），取一级衍射（m=1），衍射角是多少度？"
GOLD = "ba5cd3d9-4ffa-53e7-8614-fa79e462d1b1"

for i in range(3):
    d = E.search_text(ZH, k=4, with_pool=True, with_ranked=True)
    pool = [p.get("chunk_id") for p in d.get("pool") or []]
    ranked = [r.get("chunk_id") for r in d.get("ranked") or []]
    print(f"run#{i+1}: pool={len(pool)} 金块{'池内第'+str(pool.index(GOLD)+1) if GOLD in pool else '不在池'}"
          f"  ranked={len(ranked)} 金块{'精排第'+str(ranked.index(GOLD)+1) if GOLD in ranked else '不在精排'}")
