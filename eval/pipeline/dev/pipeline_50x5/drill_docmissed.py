# -*- coding: utf-8 -*-
"""钻取轮次8 doc_missed 低分题:金文档是没进候选池(一路盲区)还是
进了池但没被重排送进 top-4(截断)。

用法: python drill_docmissed.py
"""
import glob
import json
import os
from collections import Counter

import collect_ranked
from collect_ranked import search

HERE = os.path.dirname(os.path.abspath(__file__))
collect_ranked.TOKEN = collect_ranked._token()

gold = json.load(open(os.path.join(HERE, "runs", "gold_cache.json"),
                      encoding="utf-8"))
r8 = {}
for f in glob.glob(os.path.join(HERE, "runs", "dev200_round8_final_judge",
                                "*.jsonl")):
    for l in open(f, encoding="utf-8"):
        r = json.loads(l)
        if r.get("judge_done") and r.get("judge_context_recall") is not None:
            r8[r["id"]] = r

cat = Counter()
detail = []
for k, v in r8.items():
    if v["judge_context_recall"] >= 0.5:
        continue
    stem = v.get("source") or ""
    g = set(gold.get(stem, {}).get("gold") or [])
    if not g or v.get("source") in [s.get("source_stem") for s in
                                    (v.get("sources") or [])]:
        continue  # 只看 doc_missed 桶
    d = search(v["q"])
    ranked = d.get("ranked") or []
    pool_ids = [c.get("chunk_id") for c in ranked]
    pool_stems = {c.get("source_stem") for c in ranked}
    top4 = [s.get("chunk_id") for s in (v.get("sources") or [])]
    if stem not in pool_stems:
        cat["金文档不在池(第一路盲区)"] += 1
        detail.append((k, "no_doc_in_pool", len(pool_ids)))
    elif any(c in g for c in pool_ids):
        pos = next(i + 1 for i, c in enumerate(pool_ids) if c in g)
        cat[f"金文档在池金块在池(池内位次 {1 if pos<=24 else 25 if pos<=48 else 49}+档)"] += 1
        detail.append((k, f"gold_in_pool_pos{pos}", len(top4)))
    else:
        cat["金文档在池但金块不在池"] += 1
        detail.append((k, "doc_in_pool_gold_not", len(pool_ids)))

print(len(detail), "doc_missed 钻取:")
for c, n in cat.most_common():
    print(" ", c, n)
print()
for d in detail:
    print(" ", d)
