# -*- coding: utf-8 -*-
"""级联粗筛漏杀定位:对回退题复现召回池,分别用 m3 粗筛分与召回分排序,
看金块(matching facts)在哪一道关被杀。"""
import sys, json, re
sys.path.insert(0, "/root/autodl-tmp/app/config")
sys.path.insert(0, "/root/autodl-tmp/app/RAG")
sys.path.insert(0, "/root/autodl-tmp/app")

Q = json.load(open(sys.argv[1], encoding="utf-8"))  # [{"id","q","facts"}]

from mcp_servers.retrieval import query as Qd
from mcp_servers.retrieval import engine_api as E
import embed

re2 = embed.get_reranker2()   # m3 粗筛

def norm(s):
    return re.sub(r"[^a-z0-9一-鿿]+", "", (s or "").lower())

def cov(fact, chunk):
    """fact 对 chunk 的字符 bigram 覆盖率(粗匹配金块)"""
    f = norm(fact)[:120]
    if len(f) < 6: return 0.0
    c = norm(chunk)
    gs = {f[i:i+2] for i in range(len(f)-1)}
    return sum(1 for g in gs if g in c) / len(gs)

for item in Q:
    q, facts = item["q"], item.get("facts") or []
    if isinstance(facts, str): facts = [facts]
    # 复现 search_text 召回:多路 + RRF(与 engine_api 相同参数)
    qs = E._recall_queries(q)
    fused = Qd.query_text(qs[0], k=96)
    for extra in qs[1:]:
        fused = E._rrf_merge(fused, Qd.query_text(extra, k=96), w_b=0.7)
    docs = [str((p.payload or {}).get("content", ""))[:800] for p in fused]
    # 1) 找金块:facts 覆盖率最高的块
    best_i, best_c = -1, 0.0
    for i, d in enumerate(docs):
        c = max((cov(f, d) for f in facts), default=0.0)
        if c > best_c: best_c, best_i = c, i
    if best_i < 0 or best_c < 0.25:
        print(f"{item['id']}\tGOLD_NOT_IN_POOL\tcov={best_c:.2f}\tpool={len(fused)}")
        continue
    # 2) 金块的召回分排名(RRF)与 m3 粗筛排名
    rec_rank = sorted(range(len(fused)), key=lambda i: -fused[i].score).index(best_i) + 1
    m3s = re2.rerank(q, docs)
    m3_rank = sorted(range(len(docs)), key=lambda i: -m3s[i]).index(best_i) + 1
    gate2 = "PASS" if m3_rank <= 24 else "KILLED"
    gate1 = "PASS" if rec_rank <= 32 else "OUT32"
    print(f"{item['id']}\tpool={len(fused)}\tgold_cov={best_c:.2f}\t"
          f"recall_rank={rec_rank}/{len(fused)} ({gate1})\t"
          f"m3_rank={m3_rank}/{len(fused)} ({gate2})\tm3={m3s[best_i]:.3f}\t"
          f"{docs[best_i][:60].replace(chr(10),' ')}")
