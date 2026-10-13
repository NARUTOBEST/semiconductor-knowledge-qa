# -*- coding: utf-8 -*-
"""NO_MATCH 四题 + dev_018 分块质量排查:单 fact 短查询复测。
单查能中(cov 高) → 复合 query 污染;单查也不中 → 分块/入库问题。"""
import sys, json, re, io, contextlib
sys.path.insert(0, "/root/autodl-tmp/app/config")
sys.path.insert(0, "/root/autodl-tmp/app/RAG")
sys.path.insert(0, "/root/autodl-tmp/app")

Q = json.load(open("/root/autodl-tmp/recall26_failing.json", encoding="utf-8"))
WANT = {"dev_018", "dev_176", "dev_196", "dev_069", "dev_070"}

from mcp_servers.retrieval import query as Qd

def norm(s):
    return re.sub(r"[^a-z0-9一-鿿]+", "", (s or "").lower())

def cov(fact, chunk):
    f = norm(fact)[:120]
    if len(f) < 6: return 0.0
    c = norm(chunk)
    gs = {f[i:i+2] for i in range(len(f)-1)}
    return sum(1 for g in gs if g in c) / len(gs)

# 每题的单 fact 探针查询(把复合 query 拆成最小单事实问法)
PROBES = {
    "dev_018": "Mo/Si 多层膜反射镜 短波截止波长 12.5nm",
    "dev_176": "热磷酸 蚀刻 氮化硅 氧化硅 选择比",
    "dev_196": "聚焦透镜 准直高斯光束 束腰位置 束腰大小",
    "dev_069": "检修 上锁挂牌 lockout 程序 步骤",
    "dev_070": "氦离子束光刻 HIBL HSQ 半间距",
}

items = {it["id"]: it for it in Q if it["id"] in WANT}
for qid, probe in PROBES.items():
    it = items[qid]
    facts = it.get("facts") or []
    if isinstance(facts, str): facts = [facts]
    with contextlib.redirect_stdout(io.StringIO()):
        pts = Qd.query_text(probe, k=30)
    docs = [str((p.payload or {}).get("content", "")) for p in pts]
    scored = sorted(((max((cov(f, d) for f in facts), default=0.0), i)
                     for i, d in enumerate(docs)), reverse=True)
    best_c, best_i = scored[0]
    stems = {(pts[i].payload or {}).get("source_stem", "?") for _, i in scored[:3]}
    print(f"===== {qid} 探针: {probe}")
    print(f"  best_cov={best_c:.2f} top3来源={sorted(stems)[:2]}")
    for c, i in scored[:3]:
        print(f"  cov={c:.2f} | {docs[i][:65].replace(chr(10),' ')}")
