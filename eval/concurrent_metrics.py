# -*- coding: utf-8 -*-
"""并发评测指标计算:Recall@K / Precision@K / F1@K / MRR / NDCG@K。

输入:results_concurrent/eval*.jsonl(run_concurrent.py 产出)+ qa_sets/userN.json
输出:results_concurrent/metrics_report.json + 控制台表格。

判定口径(确定性,不调 LLM):
- 相关性(文档级):来源块 source_stem 与题库 gold source 宽松匹配(归一化后
  双向包含 / LCS≥6 / 字符覆盖≥0.8,与项目 sources.py 同款逻辑)。
- Recall@K  = top-K 中相关块数 / 检索池中相关块总数(池内有界召回)。
- Precision@K = top-K 中相关块数 / K;F1@K = 二者调和平均。
- MRR = 首个相关块排名倒数的均值(全池)。
- NDCG@K = 二元相关增益的标准 NDCG。
- 事实级 Recall@K(可选,Qdrant 取全文):事实点 = facts 内层组,组内任一
  关键词出现在某 gold 文档块的全文即该块为该事实的证据;证据块进入 top-K
  则该事实覆盖。Recall_fact@K = 覆盖事实数 / 总事实数。
- tier 路由准确率:实际 tier == 题库期望 tier。
"""
import argparse
import json
import math
import os
import re
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
KS = (3, 5, 10)
QDRANT = "http://127.0.0.1:6333"


def _norm(s):
    return re.sub(r"[^0-9a-z一-鿿]", "", str(s or "").lower())


def _lcs_len(a, b):
    best = 0
    prev = [0] * (len(b) + 1)
    for ca in a:
        cur = [0] * (len(b) + 1)
        for j, cb in enumerate(b, 1):
            if ca == cb:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best = cur[j]
        prev = cur
    return best


def _char_cover(short, long):
    if not short:
        return 0.0
    chars = set(short)
    return sum(1 for ch in chars if ch in long) / len(chars)


def doc_match(stem, gold):
    """来源书名与 gold source 宽松匹配(同 sources.py 口径)。"""
    sn, gn = _norm(stem), _norm(gold)
    if not sn or not gn:
        return False
    if sn in gn or gn in sn:
        return True
    l = _lcs_len(sn, gn)
    if l >= 6 and l >= 0.5 * len(gn):
        return True
    if l >= 4 and _char_cover(gn, sn) >= 0.8:
        return True
    return False


def qdrant_fetch_contents(chunk_ids):
    """按 chunk_id(uuid5→点id)到 ald_text/ald_image 取全文,返回 {chunk_id: content}。"""
    import uuid as _u
    out = {}
    ids = [str(_u.uuid5(_u.NAMESPACE_DNS, c)) for c in chunk_ids]
    id2cid = dict(zip(ids, chunk_ids))
    for coll in ("ald_text", "ald_image"):
        if not ids:
            break
        body = json.dumps({"ids": ids,
                           "with_payload": True, "with_vector": False}).encode()
        req = urllib.request.Request(
            f"{QDRANT}/collections/{coll}/points/retrieve", data=body,
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                d = json.load(r)
        except Exception:
            continue
        for p in d.get("result", []):
            cid = id2cid.get(p.get("id"))
            if cid is not None:
                out[cid] = (p.get("payload") or {}).get("content", "") or ""
    return out


def fact_covered(fact_groups, text):
    """一个事实点(facts 内层组)是否被 text 覆盖:组内任一关键词命中。"""
    t = _norm(text)
    return any(any(_norm(kw) and _norm(kw) in t for kw in (group if isinstance(
        group, (list, tuple)) else [group])) for group in
        (fact_groups if isinstance(fact_groups, (list, tuple)) else [fact_groups]))


def eval_record(rec, gold, with_qdrant):
    """单题指标。gold: 题库条目(dict 含 source/facts)。"""
    pool = [s for s in (rec.get("sources") or []) if isinstance(s, dict)]
    gold_src = gold.get("source") or ""
    rel_flags = [doc_match(s.get("source_stem", ""), gold_src) for s in pool]
    n_rel_pool = sum(rel_flags)
    first_rel = next((i + 1 for i, f in enumerate(rel_flags) if f), None)
    mrr = 1.0 / first_rel if first_rel else 0.0

    # 事实级证据全文
    contents = {}
    if with_qdrant and pool:
        cids = [s.get("chunk_id") for s in pool if s.get("chunk_id")]
        contents = qdrant_fetch_contents(cids)
    facts = gold.get("facts") or []
    # 每个事实点:证据块集合(相关性块中全文覆盖该事实的;无全文时退化为相关块)
    fact_ev = []
    for f in facts:
        ev_idx = set()
        for i, s in enumerate(pool):
            if not rel_flags[i]:
                continue
            txt = contents.get(s.get("chunk_id")) or s.get("content", "")
            if txt and fact_covered(f, txt):
                ev_idx.add(i)
        fact_ev.append(ev_idx)

    r = {"mrr": mrr, "n_rel_pool": n_rel_pool, "pool_size": len(pool)}
    for k in KS:
        top = rel_flags[:k]
        p = sum(top) / k if k else 0.0
        rc = (sum(top) / n_rel_pool) if n_rel_pool else 0.0
        f1 = (2 * p * rc / (p + rc)) if (p + rc) else 0.0
        dcg = sum((1.0 / math.log2(i + 2)) for i, f in enumerate(top) if f)
        idcg = sum((1.0 / math.log2(i + 2)) for i in range(min(k, n_rel_pool)))
        ndcg = dcg / idcg if idcg else 0.0
        r[f"P@{k}"] = round(p, 4)
        r[f"R@{k}"] = round(rc, 4)
        r[f"F1@{k}"] = round(f1, 4)
        r[f"NDCG@{k}"] = round(ndcg, 4)
        if facts:
            cov = sum(1 for ev in fact_ev if any(i < k for i in ev))
            r[f"Rfact@{k}"] = round(cov / len(facts), 4)
    # tier 路由准确
    r["tier"] = rec.get("tier")
    r["tier_ok"] = bool(rec.get("tier") and rec.get("tier") == gold.get("tier"))
    r["error"] = rec.get("error")
    r["wall_s"] = rec.get("wall_s")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=os.path.join(HERE, "results_concurrent"))
    ap.add_argument("--sets-dir", default=os.path.join(HERE, "qa_sets"))
    ap.add_argument("--no-qdrant", action="store_true",
                    help="跳过事实级指标(不取全文)")
    ap.add_argument("--exclude-zero-rel", action="store_true",
                    help="按 (user,id) 剔除 zero_rel_94.json 中的零相关题再统计")
    ap.add_argument("--users", type=int, default=5,
                    help="用户数上限(读 eval1..evalN)")
    args = ap.parse_args()

    excl = set()
    if args.exclude_zero_rel:
        zp = os.path.join(args.results_dir, "zero_rel_94.json")
        if os.path.exists(zp):
            excl = {(d["user"], d["id"])
                    for d in json.load(open(zp, encoding="utf-8"))}
            print(f"剔除零相关题 {len(excl)} 道")
        else:
            print(f"WARN 未找到 {zp},不剔除")

    per_user = {}
    for i in range(1, args.users + 1):
        u = f"eval{i}"
        res_path = os.path.join(args.results_dir, f"{u}.jsonl")
        if not os.path.exists(res_path):
            continue
        qa = {it["id"]: it for it in json.load(
            open(os.path.join(args.sets_dir, f"user{i}.json"), encoding="utf-8"))}
        rows = []
        with open(res_path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                gold = qa.get(rec["id"])
                if not gold or (u, rec["id"]) in excl:
                    continue
                rows.append(eval_record(rec, gold, not args.no_qdrant))
        per_user[u] = rows

    if not per_user:
        print("无结果文件")
        return

    def avg(rows, key):
        vals = [r[key] for r in rows if key in r and r[key] is not None
                and not isinstance(r[key], bool)]
        return round(sum(vals) / len(vals), 4) if vals else None

    metric_keys = (["mrr"] +
                   [f"{m}@{k}" for k in KS
                    for m in ("P", "R", "F1", "NDCG", "Rfact")] +
                   ["tier_ok", "wall_s"])
    report = {"per_user": {}, "overall": {}}
    allrows = [r for rows in per_user.values() for r in rows]
    header = "user    n    " + "  ".join(f"{k:>9}" for k in metric_keys)
    print(header)
    for u, rows in per_user.items():
        vals = {k: (round(sum(1 for r in rows if r.get(k)) / len(rows), 4)
                    if k == "tier_ok" else avg(rows, k)) for k in metric_keys}
        report["per_user"][u] = {"n": len(rows), **vals}
        print(f"{u}  {len(rows):4d}  " +
              "  ".join(f"{(vals[k] if vals[k] is not None else float('nan')):>9}"
                        for k in metric_keys))
    vals = {k: (round(sum(1 for r in allrows if r.get(k)) / len(allrows), 4)
                if k == "tier_ok" else avg(allrows, k)) for k in metric_keys}
    report["overall"] = {"n": len(allrows), **vals}
    print(f"OVERALL {len(allrows):4d}  " +
          "  ".join(f"{(vals[k] if vals[k] is not None else float('nan')):>9}"
                    for k in metric_keys))

    out = os.path.join(args.results_dir, "metrics_report.json")
    json.dump(report, open(out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
