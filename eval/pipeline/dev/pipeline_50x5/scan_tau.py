# -*- coding: utf-8 -*-
"""离线扫 RERANK_TAU / RERANK_CLIFF_RATIO / RERANK_FIXED_K(raw logits 口径)。

gold 定义(与 chunk_audit 一致):金块文档内锚点命中率 ≥0.5 的 chunk。
模拟生产选择链:ranked(全量 gemma 分,降序)→ score≥τ 且 score≥cliff×s1 → 截 fixed_k。
指标:R@sel(选中集含任一金块)/ P_proxy(选中集中金块占比)/ F1 / sel 均值 / doc_hit。

用法: python scan_tau.py --ranked runs/dev200_gemma_ranked.jsonl
"""
import argparse
import glob
import itertools
import json
import os
import re
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
QDRANT = "http://127.0.0.1:6333/collections/ald_text/points/scroll"

_ws = re.compile(r"\s+")
_ANCHOR_RE = re.compile(r"[A-Za-z]{3,}|\d+(?:\.\d+)?")
_ANCHOR_STOP = {"the", "and", "for", "with", "from", "that", "this", "are",
                "was", "were", "has", "have", "not", "inch", "inches"}


def norm(s):
    return _ws.sub("", str(s or ""))


def get_chunks(stem, limit=3000):
    cands = [stem] + [f"{stem}_p{a+1}-{a+500}" for a in range(0, 20000, 500)]
    body = json.dumps({"filter": {"must": [
        {"key": "source_stem", "match": {"any": cands}}]},
        "limit": limit, "with_payload": True}).encode()
    req = urllib.request.Request(QDRANT, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        pts = json.load(r)["result"]["points"]
    out = [{"chunk_id": p["payload"].get("chunk_id"),
            "content": norm(p["payload"].get("content"))} for p in pts]
    return out


def gold_ids(rec, chunks):
    anchors = [a for a in _ANCHOR_RE.findall(str(rec.get("facts") or ""))
               if a.lower() not in _ANCHOR_STOP]
    if not anchors:
        return set(), 0
    gold = set()
    for c in chunks:
        cl = c["content"].lower()
        hit = sum(1 for a in anchors if a.lower() in cl) / len(anchors)
        if hit >= 0.5:
            gold.add(c["chunk_id"])
    return gold, len(anchors)


def simulate(ranked, tau, cliff, fixed_k):
    """生产口径近似:score≥τ 且 score≥cliff×s1,再截 fixed_k。ranked 已按分降序。"""
    if not ranked:
        return []
    s1 = float(ranked[0]["score"])
    sel = []
    for r in ranked:
        s = float(r["score"])
        if s < tau:
            break
        if cliff and s < cliff * s1:
            break
        sel.append(r["chunk_id"])
        if len(sel) >= fixed_k:
            break
    return sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranked", default=os.path.join(
        HERE, "runs", "dev200_gemma_ranked.jsonl"))
    ap.add_argument("--out", default=os.path.join(HERE, "runs", "scan_tau.json"))
    args = ap.parse_args()

    qa = {}
    for f in sorted(glob.glob(os.path.join(
            HERE, "runs", "dev200_baseline", "qa_sets", "user*.json"))):
        for q in json.load(open(f, encoding="utf-8")):
            qa[q["id"]] = q

    # gold 缓存(避免重复 scroll 全库)
    cache_path = os.path.join(HERE, "runs", "gold_cache.json")
    cache = {}
    if os.path.exists(cache_path):
        cache = json.load(open(cache_path, encoding="utf-8"))

    recs = []
    for line in open(args.ranked, encoding="utf-8"):
        r = json.loads(line)
        if r.get("error") or not r.get("ranked"):
            continue
        recs.append(r)
    print(f"有 ranked 的题: {len(recs)}")

    data = []
    for i, r in enumerate(recs):
        stem = r["source"]
        if stem not in cache:
            chunks = get_chunks(stem)
            g, na = gold_ids({**qa.get(r["id"], {}), "id": r["id"]}, chunks)
            cache[stem] = {"gold": sorted(g for g in g if g),
                           "id_by_content": [c["chunk_id"] for c in chunks]}
        g = set(cache[stem]["gold"])
        ranked = sorted(r["ranked"], key=lambda x: -float(x["score"]))
        data.append({"id": r["id"], "ranked": ranked, "gold": g,
                     "doc": stem})
        if (i + 1) % 50 == 0:
            print(f"  gold {i+1}/{len(recs)}", flush=True)
    json.dump(cache, open(cache_path, "w", encoding="utf-8"),
              ensure_ascii=False)

    n_anchor_q = sum(1 for d in data if d["gold"])
    print(f"锚点可判金块的题: {n_anchor_q}/{len(data)}")

    grid = list(itertools.product(
        [None, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6],   # τ(None=不过滤)
        [0, 0.4, 0.5, 0.6, 0.7],                    # cliff
        [3, 4, 5, 6]))                              # fixed_k
    rows = []
    for tau, cliff, fk in grid:
        rs, ps, sizes = [], [], []
        for d in data:
            sel = simulate(d["ranked"], tau if tau is not None else -1e9,
                           cliff, fk)
            sel_g = sum(1 for c in sel if c in d["gold"])
            rs.append(1.0 if sel_g else 0.0)
            ps.append(sel_g / len(sel) if sel else 0.0)
            sizes.append(len(sel))
        mr, mp = sum(rs) / len(rs), sum(ps) / len(ps)
        f1 = 2 * mr * mp / (mr + mp) if mr + mp else 0.0
        rows.append({"tau": tau, "cliff": cliff, "fixed_k": fk,
                     "recall": round(mr, 4), "precision": round(mp, 4),
                     "f1": round(f1, 4),
                     "sel_mean": round(sum(sizes) / len(sizes), 2)})
    rows.sort(key=lambda x: -x["f1"])
    json.dump(rows, open(args.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    print("\n== TOP 15 by F1 ==")
    for x in rows[:15]:
        print(x)
    base = [x for x in rows if x["tau"] is None and x["cliff"] == 0
            and x["fixed_k"] == 4][0]
    print("\nbaseline(FIXED_K=4, 无τ/cliff):", base)
    print("== saved ->", args.out)


if __name__ == "__main__":
    main()
