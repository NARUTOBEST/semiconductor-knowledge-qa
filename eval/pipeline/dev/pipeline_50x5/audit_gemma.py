# -*- coding: utf-8 -*-
"""轮次6(gemma)之后剩余低 recall 题再审计。

与 chunk_audit 的区别:gemma 已解决"池内排序"桶,本次用生产最终上下文(items)
联合锚点覆盖率判定剩余损失到底在哪一环:
  doc_missed       金块文档不在候选池(首轮缺失)
  not_in_pool      文档在池但锚点金块 chunk 不在(第一路向量盲区)
  in_pool_not_top4 金块在池但没进 top-4(重排残留)
  in_items         金块已进最终上下文但 judge 仍低分(判分严/要点拆分/生成侧)
  anchor_weak      锚点不可判且 items 联合锚点覆盖也低(代理盲区,待人工)
覆盖度量:union_anchor_items = items 全部 content 命中的锚点占比(最接近 ragas 口径)。

用法: python audit_gemma.py --out runs/audit_gemma.json
"""
import argparse
import glob
import json
import os
import re
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
_ANCHOR_RE = re.compile(r"[A-Za-z]{3,}|\d+(?:\.\d+)?")
_ANCHOR_STOP = {"the", "and", "for", "with", "from", "that", "this", "are",
                "was", "were", "has", "have", "not", "inch", "inches"}


def anchors_of(facts):
    return [a for a in _ANCHOR_RE.findall(str(facts or ""))
            if a.lower() not in _ANCHOR_STOP]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranked", default=os.path.join(
        HERE, "runs", "dev200_gemma_ranked.jsonl"))
    ap.add_argument("--max-recall", type=float, default=0.5)
    ap.add_argument("--out", default=os.path.join(HERE, "runs", "audit_gemma.json"))
    args = ap.parse_args()

    qa = {}
    for f in sorted(glob.glob(os.path.join(
            HERE, "runs", "dev200_baseline", "qa_sets", "user*.json"))):
        for q in json.load(open(f, encoding="utf-8")):
            qa[q["id"]] = q
    cache = json.load(open(os.path.join(HERE, "runs", "gold_cache.json"),
                           encoding="utf-8"))
    jrec = {}
    for f in glob.glob(os.path.join(HERE, "runs", "dev200_gemma", "eval*.jsonl")):
        for line in open(f, encoding="utf-8"):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("judge_done") and not r.get("judge_refusal") \
                    and r.get("judge_context_recall") is not None:
                jrec[r["id"]] = r

    low = []
    for line in open(args.ranked, encoding="utf-8"):
        r = json.loads(line)
        if r.get("error") or r["id"] not in jrec:
            continue
        if float(jrec[r["id"]]["judge_context_recall"]) >= args.max_recall:
            continue
        low.append(r)
    print(f"低 recall 题(<{args.max_recall},剔 refusal): {len(low)}")

    per_q = []
    for r in low:
        q = qa.get(r["id"], {})
        anchors = anchors_of(q.get("facts"))
        gold = set(cache.get(r["source"], {}).get("gold", []))
        ranked = sorted(r["ranked"], key=lambda x: -float(x["score"]))
        items = r.get("items") or []
        items_ids = {x["chunk_id"] for x in items}
        stems = {x["source_stem"].rsplit("_p", 1)[0] for x in ranked}

        pos = next((i + 1 for i, x in enumerate(ranked)
                    if x["chunk_id"] in gold), None)
        union_hit = 0
        if anchors:
            blob = "".join(str(x.get("content") or "") for x in items).lower()
            union_hit = sum(1 for a in anchors if a.lower() in blob)

        if not anchors or not gold:
            if anchors and union_hit / len(anchors) >= 0.5:
                verdict = "in_items"       # 锚点没选中金块但内容其实覆盖了
            else:
                verdict = "anchor_weak"
        elif r["source"] not in stems:
            verdict = "doc_missed"
        elif pos is None:
            verdict = "not_in_pool"
        elif pos > len(items):
            verdict = "in_pool_not_top4"
        else:
            verdict = "in_items"
        per_q.append({
            "id": r["id"],
            "ragas_recall": jrec[r["id"]]["judge_context_recall"],
            "n_anchors": len(anchors),
            "union_anchor_items": round(union_hit / len(anchors), 2) if anchors else None,
            "gold_pos_in_ranked": pos,
            "n_items": len(items),
            "verdict": verdict,
        })

    vc = Counter(q["verdict"] for q in per_q)
    n = max(1, len(per_q))
    summary = {"n": len(per_q), "verdicts": dict(vc),
               "verdict_rate": {k: round(v / n, 3) for k, v in vc.items()},
               "mean_union_anchor": round(sum(q["union_anchor_items"] or 0
                                              for q in per_q) / n, 3),
               "per_question": sorted(per_q, key=lambda x: x["verdict"])}
    json.dump(summary, open(args.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_question"},
                     ensure_ascii=False, indent=1))
    print("== saved ->", args.out)


if __name__ == "__main__":
    main()
