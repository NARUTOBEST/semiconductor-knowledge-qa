# -*- coding: utf-8 -*-
"""heldout 泛化集成集:本 track 8 篇草稿 -> heldout_200.json(200 题)。

规格同 protocol_v2.md §三:每篇 25 题,easy 8/medium 12/hard 5,
类型 信息查询 15/数据计算 5/工具编排 3/异常处理 2,池级容差 2%。
自检:字段完整(缺即退出)、分布、去重 <=1%、同源相似对=0、排序重编号 heldout_NNN。
题集输出:eval/qa_sets/v2_20261001/
"""
import json
import os
import re
import sys
from collections import Counter
from itertools import combinations

HERE = os.path.dirname(os.path.abspath(__file__))
MAT = os.path.join(HERE, "draft_material")
OUT = os.path.join(HERE, "..", "..", "qa_sets", "v2_20261001")

DIFF_TARGET = {"easy": 0.30, "medium": 0.50, "hard": 0.20}
TYPE_TARGET = {"信息查询": 0.60, "数据计算": 0.20, "工具编排": 0.12, "异常处理": 0.08}
DIFF_ORDER = {"easy": 0, "medium": 1, "hard": 2}
REQUIRED = ("query", "task_type", "difficulty", "source_id", "required_tools",
            "expected_rounds", "gold_standard", "success_criteria", "evidence")

HELDOUT_TAGS = ["06_SECS500", "10_EUV_Lithography", "11_Chemistry_Lithography",
                "12_先进封装材料", "13_Optical_Litho_Mack", "15_印制电路上册",
                "16_超洁净处理XIII", "18_微纳加工_顾长志"]


def norm(t):
    return re.sub(r"[\s\W_]+", "", t.lower())


def load_draft(tag):
    fp = os.path.join(MAT, f"draft_{tag}.json")
    arr = json.load(open(fp, encoding="utf-8"))
    for q in arr:
        missing = [k for k in REQUIRED if not q.get(k)]
        if missing:
            sys.exit(f"[质量门] {os.path.basename(fp)} {q.get('query_id')} 缺 {missing}")
    return arr


def dist_check(items, report, tol=0.02):
    n = len(items)
    dc = Counter(i["difficulty"] for i in items)
    tc = Counter(i["task_type"] for i in items)
    dmax = max(round(abs(dc.get(k, 0) / n - v), 6) for k, v in DIFF_TARGET.items())
    tmax = max(round(abs(tc.get(k, 0) / n - v), 6) for k, v in TYPE_TARGET.items())
    ok = dmax <= tol and tmax <= tol
    report["difficulty"] = {k: round(dc.get(k, 0) / n, 4) for k in DIFF_TARGET}
    report["task_type"] = {k: round(tc.get(k, 0) / n, 4) for k in TYPE_TARGET}
    report["distribution_check"] = "pass" if ok else f"fail({dmax:.3f}/{tmax:.3f})"
    return ok


def main():
    items = [q for t in HELDOUT_TAGS for q in load_draft(t)]
    report = {"total": len(items)}
    ok = dist_check(items, report)

    seen, dup = {}, []
    for i in items:
        k = norm(i["query"])
        if k in seen:
            dup.append((seen[k], i["query_id"]))
        seen[k] = i["query_id"]
    report["duplicate_check"] = "pass" if len(dup) / len(items) <= 0.01 else f"fail {dup}"

    sim = []
    for a, b in combinations(items, 2):
        if a["source_id"] != b["source_id"]:
            continue
        na, nb = norm(a["query"]), norm(b["query"])
        if len(na) > 8 and len(nb) > 8 and (na in nb or nb in na):
            sim.append((a["query_id"], b["query_id"]))
    report["same_source_similar_pairs"] = sim

    items.sort(key=lambda x: DIFF_ORDER[x["difficulty"]])
    for n, i in enumerate(items, 1):
        i["query_id"] = f"heldout_{n:03d}"

    os.makedirs(OUT, exist_ok=True)
    json.dump(report, open(os.path.join(OUT, "validation_report_heldout.json"), "w",
                           encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(items, open(os.path.join(OUT, "heldout_200.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
