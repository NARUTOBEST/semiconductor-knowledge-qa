# -*- coding: utf-8 -*-
"""chunk 粒度审计 v2:低召回题里,金块内容到底卡在哪一环。

对每道 judge_context_recall < 阈值 的题:
  1. 拉金块文档(题目 source 字段 -> payload source_stem <标题>_p<起>-<止>)全部 chunk;
  2. 把 facts(裁判参考要点,人工提炼的改写句)做 8 字滑窗 shingle,
     对每个 chunk 算 shingle 命中率,取最优块 best_ratio 与邻块联合 union_ratio;
  3. 判定该题卡点:
     - doc_missed   金块文档压根没被检回(sources 里无该 stem 前缀)→ 检索/域收窄问题
     - content_lost best_ratio < 0.25 → 金内容不在库(入库丢/事实跨多文档)
     - truncated    union_ratio - best_ratio ≥ 0.12 → 内容跨块被切断 → 分块边界问题
     - ranking_fail best_ratio ≥ 0.6 但金块不在 top-K → 粘连/重排问题
     - partial      其余(内容在但匹配一般)

输出 JSON(汇总+每题明细)。用法:
  python chunk_audit.py --results-dir runs/dev200_afterscope/judge_coding --out audit.json
"""
import argparse
import glob
import json
import os
import re
import urllib.request
from collections import Counter

QDRANT = "http://127.0.0.1:6333/collections/ald_text/points/scroll"
HERE = os.path.dirname(os.path.abspath(__file__))

_ws = re.compile(r"\s+")


def norm(s):
    return _ws.sub("", str(s or ""))


def get_chunks(stem, limit=3000):
    """拉金块文档全部 chunk。payload source_stem 有两种格式并存:
    裸标题 或 <标题>_p<起>-<止>(500 页一段),match_any 一次查完。"""
    cands = [stem] + [f"{stem}_p{a+1}-{a+500}" for a in range(0, 20000, 500)]
    body = json.dumps({"filter": {"must": [
        {"key": "source_stem", "match": {"any": cands}}]},
        "limit": limit, "with_payload": True}).encode()
    req = urllib.request.Request(QDRANT, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        pts = json.load(r)["result"]["points"]
    out = []
    for p in pts:
        pl = p["payload"]
        out.append({"chunk_id": pl.get("chunk_id"),
                    "content": norm(pl.get("content")),
                    "raw_len": int(pl.get("char_count") or 0),
                    "idx": int(pl.get("chunk_index") or 0),
                    "table": bool(pl.get("has_table"))})
    out.sort(key=lambda c: c["idx"])
    return out


def shingles_of(text, k=8):
    t = norm(text)
    return {t[i:i + k] for i in range(0, max(1, len(t) - k + 1))} or {t}


_ANCHOR_RE = re.compile(r"[A-Za-z]{3,}|\d+(?:\.\d+)?")
_ANCHOR_STOP = {"the", "and", "for", "with", "from", "that", "this", "are",
                "was", "were", "has", "have", "not", "inch", "inches"}


def analyze_question(rec):
    """金块定位用跨语言不变锚点:数字 + 英文术语(文档为英文手册,facts 为中文
    提炼句,字面匹配跨语言全灭;数字/型号/术语在翻译里保留)。"""
    chunks = get_chunks(rec["source"])
    anchors = [a for a in _ANCHOR_RE.findall(str(rec["facts"]))
               if a.lower() not in _ANCHOR_STOP]
    scored = []
    for c in chunks:
        cl = c["content"].lower()
        hit = sum(1 for a in anchors if a.lower() in cl) / max(1, len(anchors))
        scored.append((hit, c))
    scored.sort(key=lambda x: -x[0])
    best_ratio, best_c = (scored[0][0], scored[0][1]) if scored else (0.0, None)
    ret_stems = {str(s.get("source_stem") or "").rsplit("_p", 1)[0]
                 for s in (rec.get("sources") or [])}
    ret_ids = {s.get("chunk_id") for s in (rec.get("sources") or [])}
    doc_missed = rec["source"] not in ret_stems
    gold_chunk_in_topk = best_c is not None and best_c["chunk_id"] in ret_ids

    if doc_missed:
        verdict = "doc_missed"            # 金块文档没被检回
    elif not anchors or best_ratio < 0.5:
        verdict = "anchor_missing"        # 锚点找不到(内容不在库/锚点太弱)
    elif gold_chunk_in_topk:
        verdict = "in_topk"               # 内容块进了 top-K(判分严或块内粘连)
    else:
        verdict = "ranking_fail"          # 内容在库、文档也被检回,但金块没进 top-K
    return {
        "id": rec["id"], "recall": rec.get("judge_context_recall"),
        "n_anchors": len(anchors), "n_chunks": len(chunks),
        "best_ratio": round(best_ratio, 3),
        "gold_chunk_in_topk": gold_chunk_in_topk,
        "best_is_table": bool(best_c and best_c["table"]),
        "best_chunk_len": best_c["raw_len"] if best_c else 0,
        "verdict": verdict,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--max-recall", type=float, default=0.5)
    ap.add_argument("--out", default=os.path.join(HERE, "chunk_audit.json"))
    args = ap.parse_args()

    recs = []
    for f in sorted(glob.glob(os.path.join(args.results_dir, "eval*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if not r.get("judge_done") or r.get("judge_refusal"):
                continue
            if float(r.get("judge_context_recall") or 0) < args.max_recall \
                    and r.get("facts") and r.get("source"):
                recs.append(r)
    print(f"低召回题(recall<{args.max_recall}): {len(recs)} 道")

    per_q = [analyze_question(r) for r in recs]
    vc = Counter(q["verdict"] for q in per_q)
    n = max(1, len(per_q))
    summary = {
        "results_dir": args.results_dir, "n_questions": len(per_q),
        "verdicts": dict(vc),
        "verdict_rate": {k: round(v / n, 3) for k, v in vc.items()},
        "per_question": per_q,
    }
    json.dump(summary, open(args.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_question"},
                     ensure_ascii=False, indent=1))
    print("== saved ->", args.out)


if __name__ == "__main__":
    main()
