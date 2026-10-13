# -*- coding: utf-8 -*-
"""轮次19 术语表替换效果验证:跨语言零重叠(dev_036)与中文查询 vs 英文规格表
(dev_043)badcase,对比开关前后的召回池金块命中(本地 CPU embed + VM qdrant
隧道,无需 GPU 机;口径与线上 search_text 一致:原句路 + 翻译路 RRF 融合,
池宽 RERANK_RECALL_K=96)。
用法:项目根 python ssh_helper/verify_glossary_badcase.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "config"))
sys.path.insert(0, os.path.join(ROOT, "RAG"))
# 线上检索环境变量先于 import config 注入(qdrant 隧道/翻译 key)
for _line in open(os.path.join(ROOT, "ssh_helper", "ret.env"), encoding="utf-8"):
    _line = _line.strip()
    if _line and not _line.startswith("#") and "=" in _line:
        _k, _, _v = _line.partition("=")
        os.environ.setdefault(_k.strip(), _v.strip())

import config as C
from mcp_servers.retrieval import engine_api as E
from mcp_servers.retrieval import query as Q

# VM qdrant 跨隧道全文扫描慢(399k 点,~30-150s):客户端加大超时
from qdrant_client import QdrantClient
Q.get_client = lambda: QdrantClient(url=C.QDRANT_URL, timeout=150)

RECALL_K = int(os.getenv("RERANK_RECALL_K", "96"))
BASE = json.load(open(os.path.join(ROOT, "ssh_helper", "badcase14_baseline.json"),
                      encoding="utf-8"))
CASES = ["dev_036", "dev_043", "dev_124"]   # 124 反事实:预期无变化(不伤害)


def fused_pool(q):
    """复现 search_text 召回段:qs = _recall_queries(q);主路 + 辅路 RRF。"""
    qs = E._recall_queries(q)
    fused = Q.query_text(qs[0], k=RECALL_K)
    for extra in qs[1:]:
        fused = E._rrf_merge(fused, Q.query_text(extra, k=RECALL_K),
                             w_b=float(getattr(C, "TRANSLATE_RRF_WEIGHT", 0.7)))
    return qs, fused


def main():
    for qid in CASES:
        rec = BASE[qid]
        q = rec["q"]
        gold = {s["chunk_id"] for s in rec.get("sources", [])}

        C.RETRIEVAL_GLOSSARY_REPLACE = False
        qs0, pool0 = fused_pool(q)
        C.RETRIEVAL_GLOSSARY_REPLACE = True
        qs1, pool1 = fused_pool(q)
        C.RETRIEVAL_GLOSSARY_REPLACE = False

        def report(tag, qs, pool):
            ids = [p.payload.get("chunk_id") for p in pool]
            ranks = [(i + 1, cid) for i, cid in enumerate(ids) if cid in gold]
            best = min((r for r, _ in ranks), default=None)
            print(f"  {tag}: 路数={len(qs)} 池={len(ids)} "
                  f"金块入池={len(ranks)}/{len(gold)} 最佳名次={best}")
            print(f"    queries={[(x or '')[:70] for x in qs]}")
            if ranks:
                print(f"    金块名次={ranks[:3]}")
            return best

        print(f"\n== {qid}: {q}")
        print(f"   金块: {sorted(gold)[0][:40]}... 等 {len(gold)} 条")
        b0 = report("关(基线)", qs0, pool0)
        b1 = report("开(替换)", qs1, pool1)
        delta = "" if b0 == b1 else f"  <-- {b0} -> {b1}"
        print(f"  结论: {'改善' if (b1 and (not b0 or b1 < b0)) else ('持平' if b0 == b1 else '回退')}{delta}")


if __name__ == "__main__":
    main()
