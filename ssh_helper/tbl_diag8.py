# -*- coding: utf-8 -*-
"""第二轮:4 题精确事实存在性(打印全部命中,判定 分块缺失 vs 检索失败)。"""
import sys, re
sys.path.insert(0, "/root/autodl-tmp/app/config")
sys.path.insert(0, "/root/autodl-tmp/app/RAG")
sys.path.insert(0, "/root/autodl-tmp/app")
from qdrant_client import QdrantClient
import config as C

client = QdrantClient(url="http://127.0.0.1:6333", timeout=90)

def scan(stem_pat):
    chunks, offset = [], None
    while True:
        pts, offset = client.scroll(
            collection_name=C.TEXT_COLLECTION, limit=128, offset=offset,
            with_payload=True, with_vectors=False,
            scroll_filter={"must": [{"key": "source_stem", "match": {"text": stem_pat}}]})
        for p in pts:
            pl = p.payload or {}
            chunks.append((pl.get("content") or "", pl))
        if offset is None or not pts:
            break
    return chunks

# dev_043: magnatran placement repeatability / TIR
ch = scan("双臂取晶圆机器人magnatran")
print("### dev_043 magnatran 块数", len(ch), flush=True)
for c, pl in ch:
    if re.search(r"repeatab", c, re.I) and re.search(r"placement|TIR|0\.1", c, re.I):
        mark = "TBL" if (pl.get("has_table") or pl.get("table_html")) else "txt"
        print(f"  <{mark}> {c[:200]}".replace(chr(10), " ⏎ "), flush=True)

# dev_094: Mi30 peel force 相关段
ch = scan("WAFER  HANDLER Mi30")
print("### dev_094 Mi30 peel 命中", flush=True)
for c, pl in ch:
    if re.search(r"peel", c, re.I) and re.search(r"seal|compress|temperature|force", c, re.I):
        mark = "TBL" if (pl.get("has_table") or pl.get("table_html")) else "txt"
        print(f"  <{mark}> {c[:200]}".replace(chr(10), " ⏎ "), flush=True)

# dev_033: 7AF utility requirements / GPM 表
ch = scan("晶圆减薄strasbaugh Model 7AF")
print("### dev_033 7AF GPM/PSI 命中", flush=True)
for c, pl in ch:
    if re.search(r"GPM", c) and re.search(r"PSI|coolant|D\.I", c, re.I):
        mark = "TBL" if (pl.get("has_table") or pl.get("table_html")) else "txt"
        print(f"  <{mark}> {c[:240]}".replace(chr(10), " ⏎ "), flush=True)

# dev_198: MKS 光栅方程
ch = scan("MKS 仪器手册光子技术")
print("### dev_198 MKS 光栅方程命中", flush=True)
for c, pl in ch:
    if re.search(r"grating equation|m\\\\?lambda|d\s*G|mλ|sin\s?α|sin\s?β", c) or \
       (re.search(r"groove", c, re.I) and re.search(r"equation|sin", c, re.I)):
        mark = "TBL" if (pl.get("has_table") or pl.get("table_html")) else "txt"
        print(f"  <{mark}> {c[:240]}".replace(chr(10), " ⏎ "), flush=True)
