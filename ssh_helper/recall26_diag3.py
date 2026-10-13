# -*- coding: utf-8 -*-
"""GATE3 归因:5 题里 4B 精排压过金块的前几名是谁、分差多少。"""
import sys, json, re, io, contextlib
sys.path.insert(0, "/root/autodl-tmp/app/config")
sys.path.insert(0, "/root/autodl-tmp/app/RAG")
sys.path.insert(0, "/root/autodl-tmp/app")

Q = json.load(open(sys.argv[1], encoding="utf-8"))
WANT = {"dev_018", "dev_139", "dev_043", "dev_105", "dev_184"}

from mcp_servers.retrieval import query as Qd
from mcp_servers.retrieval import engine_api as E
import embed

re2 = embed.get_reranker2()
re1 = embed.get_reranker()

def norm(s):
    return re.sub(r"[^a-z0-9一-鿿]+", "", (s or "").lower())

def cov(fact, chunk):
    f = norm(fact)[:120]
    if len(f) < 6: return 0.0
    c = norm(chunk)
    gs = {f[i:i+2] for i in range(len(f)-1)}
    return sum(1 for g in gs if g in c) / len(gs)

for item in Q:
    if item["id"] not in WANT: continue
    q, facts = item["q"], item.get("facts") or []
    if isinstance(facts, str): facts = [facts]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        qs = E._recall_queries(q)
        fused = Qd.query_text(qs[0], k=96)
        for extra in qs[1:]:
            fused = E._rrf_merge(fused, Qd.query_text(extra, k=96), w_b=0.7)
        docs = [str((p.payload or {}).get("content", ""))[:800] for p in fused]
        best_c, best_i = max((max((cov(f, d) for f in facts), default=0.0), i) for i, d in enumerate(docs))
        m3s = re2.rerank(q, docs)
        keep = sorted(range(len(docs)), key=lambda i: -m3s[i])[:24]
        fine_docs = [docs[i] for i in keep]
        f1 = re1.rerank(q, fine_docs)
        order = sorted(range(len(fine_docs)), key=lambda i: -f1[i])
    print(f"===== {item['id']} {q[:46]}")
    print(f"  facts: {' / '.join((f[:60] if isinstance(f,str) else '') for f in facts)[:150]}")
    for rk, i in enumerate(order[:6], 1):
        pos = keep[i]
        gold = " <== 金块" if pos == best_i else ""
        print(f"  #{rk} fine={f1[i]:.3f} cov={max((cov(f, fine_docs[i]) for f in facts), default=0):.2f}"
              f" | {fine_docs[i][:70].replace(chr(10),' ')}{gold}")
