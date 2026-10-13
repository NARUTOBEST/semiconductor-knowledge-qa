# -*- coding: utf-8 -*-
"""400 题验收 v2:evidence 对素材原文的 n-gram 窗口覆盖率。

标准:evidence 按页标/省略号/中文括号批注切分成片段;每个片段算 12 字符滑窗命中率
(表格片段按 | 分格后取均值)。题目通过 = 所有片段覆盖率 >=0.6(agent 引文允许省略
与穿插,但不得大面积无中生有)。输出 per-question 最低覆盖率,人工复核最差项。
"""
import html
import json, os, re, unicodedata
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
MAT = os.path.join(HERE, "draft_material")
PAGE = re.compile(r"--\[p.*?\]--")
KEEP = re.compile(r"[^0-9A-Za-z一-鿿]")
WIN, STEP, PASS_COV = 12, 4, 0.60

TAGS = sorted(os.path.splitext(f)[0][6:] for f in os.listdir(MAT)
            if f.startswith("draft_") and f.endswith(".json"))  # 本 track 全部草稿


def loose(t):
    t = html.unescape(t)
    return KEEP.sub("", unicodedata.normalize("NFKC", t)).lower()


def fragments(ev):
    segs = []
    for p in PAGE.split(ev):
        for frag in re.split(r"……|\.{3,}|…", p):
            frag = re.sub(r"（[^）]*）", "", frag).strip()  # 去 agent 中文批注
            if len(loose(frag)) >= 10:
                segs.append(frag)
    return segs


def coverage(frag, mat):
    """滑窗命中率;表格片段取各格均值。"""
    cells = [c for c in re.split(r"\s*\|\s*", frag) if len(loose(c)) >= 6]
    units = cells if len(cells) >= 3 else [frag]
    ratios = []
    for u in units:
        L = loose(u)
        if len(L) <= WIN:
            ratios.append(1.0 if L in mat else 0.0)
            continue
        wins = [L[i:i + WIN] for i in range(0, len(L) - WIN + 1, STEP)]
        ratios.append(sum(w in mat for w in wins) / len(wins))
    return sum(ratios) / len(ratios)


def main():
    report, rows = {}, []
    for tag in TAGS:
        draft = json.load(open(os.path.join(MAT, f"draft_{tag}.json"), encoding="utf-8"))
        mat = loose(open(os.path.join(MAT, f"{tag}.txt"), encoding="utf-8").read())
        stats = Counter()
        for q in draft:
            frs = fragments(q["evidence"])
            if not frs:
                stats["no_frag"] += 1
                rows.append((tag, q["query_id"], 0.0, "evidence无可校验片段"))
                continue
            covs = [coverage(f, mat) for f in frs]
            mc = min(covs)
            if mc >= PASS_COV:
                stats["pass"] += 1
            else:
                worst = frs[covs.index(mc)]
                rows.append((tag, q["query_id"], round(mc, 2), worst[:80]))
                stats["fail"] += 1
        report[tag] = {"pass": stats["pass"], "fail": stats["fail"],
                       "no_frag": stats["no_frag"]}
    total = sum(v["pass"] + v["fail"] + v["no_frag"] for v in report.values())
    ok = sum(v["pass"] for v in report.values())
    report["_SUMMARY"] = {"total": total, "pass": ok, "fail": total - ok,
                          "pass_rate": round(ok / total, 4)}
    rows.sort(key=lambda r: r[2])
    with open(os.path.join(HERE, "_acceptance_failures.txt"), "w", encoding="utf-8") as f:
        for tag, qid, cov, msg in rows:
            f.write(f"{cov:.2f}\t{tag}\t{qid}\t{msg}\n")
    json.dump(report, open(os.path.join(HERE, "acceptance_report_400.json"), "w",
                           encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(report["_SUMMARY"], ensure_ascii=False))
    for tag in TAGS:
        v = report[tag]
        print(f'{tag}: pass {v["pass"]} fail {v["fail"]}' + ("  <--" if v["fail"] else ""))


if __name__ == "__main__":
    main()
