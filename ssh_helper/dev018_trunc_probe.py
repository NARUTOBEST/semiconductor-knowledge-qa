# -*- coding: utf-8 -*-
"""dev_018 归因探针:金块(中文推理块)到底输在哪——
A) 800字截断把关键句切掉了?  B) 4B 真的偏词面?  用全文 vs 截断对照。"""
import sys, json, re, io, contextlib
sys.path.insert(0, "/root/autodl-tmp/app/config")
sys.path.insert(0, "/root/autodl-tmp/app/RAG")
sys.path.insert(0, "/root/autodl-tmp/app")

q = "EUVL 为什么最终选了 13.5 nm 这个波长?Mo/Si 多层膜反射镜的短波截止波大概是多少?"
from mcp_servers.retrieval import query as Qd
from mcp_servers.retrieval import engine_api as E
import embed

re1 = embed.get_reranker()

def norm(s):
    return re.sub(r"[^a-z0-9一-鿿]+", "", (s or "").lower())

def cov(fact, chunk):
    f = norm(fact)[:120]
    if len(f) < 6: return 0.0
    c = norm(chunk)
    gs = {f[i:i+2] for i in range(len(f)-1)}
    return sum(1 for g in gs if g in c) / len(gs)

fact = "Mo/Si 多层膜反射镜在短波侧的截止波长约 12.5 nm,但没有任何光源能有效利用这一窗口"
KEY = "12.5"

with contextlib.redirect_stdout(io.StringIO()):
    fused = Qd.query_text(q, k=96)
# 找含 12.5 的中文金块(全文)
gold = None
def content_of(p):
    pl = p.payload
    if isinstance(pl, str): return pl
    return str((pl or {}).get("content", ""))
for p in fused:
    c = content_of(p)
    if KEY in c and "多层膜" in c or (KEY in c and "截止" in c):
        gold = (p, c); break
if not gold:
    # 退而求其次:cov 最高且含 12.5
    for p in fused:
        c = content_of(p)
        if KEY in c:
            gold = (p, c); break
p, full = gold
print("金块全文长度:", len(full))
pos = full.find(KEY)
print(f"关键句位置: 第 {pos} 字符(截断800则{'在' if pos < 800 else '不在'}送入范围)")
print("关键句上下文:", full[max(0,pos-40):pos+80].replace(chr(10), " "))
# 对照:截断 vs 全文
t = full[:800]
s_trunc = re1.rerank(q, [t])[0]
s_full = re1.rerank(q, [full])[0]
print(f"\n4B 打分: 截断800={s_trunc:.3f}  全文={s_full:.3f}")
# 竞品块:复述 13.5 的英文块
comp = [content_of(pp) for pp in fused if "13.5 nm" in content_of(pp)][:2]
for i, c in enumerate(comp):
    print(f"竞品{i} 截断800分={re1.rerank(q, [c[:800]])[0]:.3f} 全文分={re1.rerank(q, [c])[0]:.3f} | {c[:60].replace(chr(10),' ')}")
