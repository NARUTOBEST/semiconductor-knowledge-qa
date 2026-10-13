# -*- coding: utf-8 -*-
"""轮次8 部署验证:7 题池外题重采,看扩表(533词表)+双路重排后金块是否进池。

用法: python verify_pool7.py
gold 取 scan_tau 的 gold_cache(按 source_stem 键);pool/ranked 走线上
127.0.0.1:8002(当前配置 = DUAL_RERANK=1 + 词表 533)。
"""
import glob
import json
import os

import collect_ranked
from collect_ranked import search  # 复用 token 解析与请求

collect_ranked.TOKEN = collect_ranked._token()

HERE = os.path.dirname(os.path.abspath(__file__))
IDS = {"dev_196", "dev_030", "dev_036", "dev_075", "dev_084", "dev_169",
       "dev_171"}


def main():
    gold_cache = json.load(open(os.path.join(HERE, "runs", "gold_cache.json"),
                                encoding="utf-8"))
    recs = []
    for f in sorted(glob.glob(os.path.join(
            HERE, "runs", "dev200_baseline", "qa_sets", "user*.json"))):
        for r in json.load(open(f, encoding="utf-8")):
            if r.get("id") in IDS:
                recs.append(r)
    print(f"loaded {len(recs)} recs")
    for r in recs:
        stem = r.get("source") or ""
        gold = set((gold_cache.get(stem) or {}).get("gold") or [])
        d = search(r["q"])
        ranked = [c.get("chunk_id") for c in (d.get("ranked") or [])]
        pool_ids = ranked  # ranked 即重排前池全量
        first = next((i + 1 for i, c in enumerate(ranked) if c in gold), None)
        print(f"{r['id']}: gold={len(gold)} pool={len(pool_ids)} "
              f"first_gold_pos={first} "
              f"gold_in_pool={any(c in gold for c in pool_ids)}")


if __name__ == "__main__":
    main()
