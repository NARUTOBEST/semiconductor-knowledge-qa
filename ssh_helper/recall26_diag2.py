# -*- coding: utf-8 -*-
"""三道关全定位 v2:池→m3粗筛(24)→4B精排(top-4)。金块用 facts 匹配,阈值放宽。"""
import sys, json, re, io, contextlib
sys.path.insert(0, "/root/autodl-tmp/app/config")
sys.path.insert(0, "/root/autodl-tmp/app/RAG")
sys.path.insert(0, "/root/autodl-tmp/app")

Q = json.load(open(sys.argv[1], encoding="utf-8"))

from mcp_servers.retrieval import query as Qd
from mcp_servers.retrieval import engine_api as E
import embed

re2 = embed.get_reranker2()   # m3 粗筛
re1 = embed.get_reranker()    # Qwen3-4B 精排

def norm(s):
    return re.sub(r"[^a-z0-9一-鿿]+", "", (s or "").lower())

def cov(fact, chunk):
    f = norm(fact)[:120]
    if len(f) < 6: return 0.0
    c = norm(chunk)
    gs = {f[i:i+2] for i in range(len(f)-1)}
    return sum(1 for g in gs if g in c) / len(gs)

for item in Q:
    q, facts = item["q"], item.get("facts") or []
    if isinstance(facts, str): facts = [facts]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        qs = E._recall_queries(q)
        fused = Qd.query_text(qs[0], k=96)
        for extra in qs[1:]:
            fused = E._rrf_merge(fused, Qd.query_text(extra, k=96), w_b=0.7)
        docs = [str((p.payload or {}).get("content", ""))[:800] for p in fused]
        cands = sorted(((max((cov(f, d) for f in facts), default=0.0), i)
                        for i, d in enumerate(docs)), reverse=True)
        best_c, best_i = cands[0]
        m3s = re2.rerank(q, docs)
        keep = sorted(range(len(docs)), key=lambda i: -m3s[i])[:24]
        fine_docs = [docs[i] for i in keep]
        f1 = re1.rerank(q, fine_docs)
    if best_c < 0.18:
        print(f"{item['id']}\tNO_MATCH\tbest_cov={best_c:.2f}\tpool={len(fused)}")
        continue
    m3_rank = sorted(range(len(docs)), key=lambda i: -m3s[i]).index(best_i) + 1
    rec_rank = sorted(range(len(fused)), key=lambda i: -fused[i].score).index(best_i) + 1
    if best_i not in keep:
        tag = "GATE2(m3筛掉)"
        fine_rank = -1
    else:
        pos = keep.index(best_i)
        fine_rank = sorted(range(len(fine_docs)), key=lambda i: -f1[i]).index(pos) + 1
        tag = "ALL_PASS" if fine_rank <= 4 else f"GATE3(精排第{fine_rank}/24)"
    print(f"{item['id']}\tcov={best_c:.2f}\trec={rec_rank}\tm3={m3_rank}\tfine={fine_rank}\t{tag}\t"
          f"{docs[best_i][:55].replace(chr(10),' ')}")
