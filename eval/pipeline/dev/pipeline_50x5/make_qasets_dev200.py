# -*- coding: utf-8 -*-
"""dev_200.json -> 评测运行器用的 qa_sets(单 user 或 N 均分),字段映射:
query_id->id, query->q, source_id->source, gold_standard->facts, expected_rounds->tier。
用法: python make_qasets_dev200.py --users 5 --out <dir>
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
QA = os.path.join(HERE, "..", "..", "..", "qa_sets", "v2_20261001", "dev_200.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(HERE, "runs", "dev200_baseline", "qa_sets"))
    a = ap.parse_args()
    items = json.load(open(QA, encoding="utf-8"))
    conv = [{"id": q["query_id"], "q": q["query"], "source": q["source_id"],
             "facts": q["gold_standard"], "tier": q["expected_rounds"],
             "difficulty": q["difficulty"], "task_type": q["task_type"]}
            for q in items]
    os.makedirs(a.out, exist_ok=True)
    n = a.users
    for i in range(1, n + 1):                      # 均分保序
        part = conv[(i - 1) * len(conv) // n: i * len(conv) // n]
        json.dump(part, open(os.path.join(a.out, f"user{i}.json"), "w",
                             encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"user{i}: {len(part)} 题")


if __name__ == "__main__":
    main()
