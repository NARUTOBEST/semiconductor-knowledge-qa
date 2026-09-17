# -*- coding: utf-8 -*-
"""τ 定标扫描 v2:对评测集跑生产检索管线(召回80 → RRF top32 + exact 兜底 →
heading_path+content 重排),记录每个候选块的 rerank 分与 gold 标签,
用于动态截断阈值 RERANK_TAU 定值(Phase1 重排文本加入 heading_path 后分布已变)。

用法(GPU 机,qdrant 经隧道回到 VM):
  QDRANT_URL=http://127.0.0.1:6333 HF_HUB_OFFLINE=1 python eval/tau_sweep.py [--per-user 50]

输出: eval/results_concurrent/tau_sweep.jsonl
  每行一个查询: {id, uid, q, gold, candidates: [{rank, score, gold, stem}]}
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")

_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJECT, "config"))
sys.path.insert(0, os.path.join(_PROJECT, "RAG"))
sys.path.insert(0, _PROJECT)
sys.path.insert(0, os.path.join(_PROJECT, "mcp_servers"))

import config as C  # noqa: E402
import embed  # noqa: E402
from mcp_servers.retrieval import query as Q  # noqa: E402
from eval.concurrent_metrics import doc_match  # noqa: E402

_TOP_RERANK = getattr(C, "RERANK_RECALL_K", 32)  # 与生产 RERANK_RECALL_K 一致
_OUT = os.path.join(_PROJECT, "eval", "results_concurrent", "tau_sweep.jsonl")


def load_queries(per_user):
    qs = []
    for u in range(1, 6):
        data = json.load(open(
            os.path.join(_PROJECT, "eval", "qa_sets", f"user{u}.json"),
            encoding="utf-8"))
        items = data if isinstance(data, list) else data.get(
            "questions", data.get("items", []))
        for it in items[:per_user] if per_user else items:
            qs.append({"id": it.get("id"), "uid": f"u{u}_{it.get('id')}",
                       "q": it.get("q", ""),
                       "gold": it.get("source", "") or ""})
    return qs


def text_rerank_text(p):
    """与 engine_api._text_rerank_text 一致:标题路径 + 正文。"""
    pl = p.payload or {}
    hp = (pl.get("heading_path") or "").strip()
    content = pl.get("content", "") or ""
    return f"{hp}\n{content}" if hp else content


def candidates_for(q, gold, reranker):
    """复刻 engine_api.search_text 的候选构造(不截断,全量记录分数)。"""
    recall_k = _TOP_RERANK
    fused = Q.query_text(q, k=recall_k)
    exact = Q.exact_code_recall(q, limit=6)
    if fused:
        top_score = fused[0].score
        filtered = [p for p in fused if p.score >= top_score * 0.6]
    else:
        filtered = []
    have = {p.id for p in filtered}
    pts = [p for p in exact if p.id not in have] + filtered
    if not pts:
        return []
    docs = [text_rerank_text(p) for p in pts]
    with embed.RERANK_LOCK:
        scores = reranker.rerank(q, docs)
    order = sorted(zip(pts, scores), key=lambda x: x[1], reverse=True)
    cands = []
    for rank, (p, s) in enumerate(order, 1):
        stem = (p.payload or {}).get("source_stem", "")
        cands.append({"rank": rank, "score": round(float(s), 4),
                      "gold": bool(gold and doc_match(stem, gold)),
                      "stem": stem})
    return cands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-user", type=int, default=50)
    args = ap.parse_args()

    queries = load_queries(args.per_user)
    done = set()
    if os.path.exists(_OUT):
        with open(_OUT, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line).get("uid"))
                except Exception:
                    pass
        queries = [q for q in queries if q["uid"] not in done]
    print(f"queries: {len(queries)} (skip {len(done)} done)", flush=True)

    reranker = embed.get_reranker()
    print("models loaded", flush=True)

    t0 = time.time()
    with open(_OUT, "a", encoding="utf-8") as f:
        for i, item in enumerate(queries, 1):
            rec = {"id": item["id"], "uid": item["uid"],
                   "q": item["q"], "gold": item["gold"]}
            try:
                rec["candidates"] = candidates_for(item["q"], item["gold"], reranker)
            except Exception as e:
                rec["error"] = type(e).__name__ + ": " + str(e)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if i % 25 == 0:
                el = time.time() - t0
                print(f"[{i}/{len(queries)}] {el:.0f}s, {el/i:.1f}s/q, "
                      f"ETA {(el/i)*(len(queries)-i)/60:.0f}min", flush=True)
    print(f"DONE {time.time()-t0:.0f}s -> {_OUT}", flush=True)


if __name__ == "__main__":
    main()
