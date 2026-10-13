# -*- coding: utf-8 -*-
"""进程内 Qwen3 重排对照:同召回池打分,对比金块排名(vLLM 打分质量验证)。"""
import sys
sys.path.insert(0, "/root/autodl-tmp/app/config")
sys.path.insert(0, "/root/autodl-tmp/app/RAG")
sys.path.insert(0, "/root/autodl-tmp/app")

q = "6DS-SP 主轴真空开关的整定读数是多少?开关是什么品牌?左主轴安装在哪里?"
from mcp_servers.retrieval import query as Q
pts = Q.query_text(q, k=96)
docs = [str((p.payload or {}).get("content", ""))[:800] for p in pts]

import embed
re_ = embed.get_reranker()
scores = re_.rerank(q, docs)
ranked = sorted(range(len(docs)), key=lambda i: -scores[i])
print("== inproc Qwen3 top8 ==")
for i in ranked[:8]:
    print(" ", round(scores[i], 3), "idx", i, docs[i][:55].replace("\n", " "))
for gi in (2, 4, 45):
    print("GOLD", gi, "inproc rank", ranked.index(gi) + 1, "score", round(scores[gi], 3))
