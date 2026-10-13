# -*- coding: utf-8 -*-
"""从 qdrant ald_text 倒出全部含表 chunk(has_table/table_html),供表格分块治理离线分析。
用法: python dump_tables.py [输出路径]  (GPU 上跑,默认 /root/autodl-tmp/tables_dump.json)"""
import sys, json, os
sys.path.insert(0, "/root/autodl-tmp/app/config")
sys.path.insert(0, "/root/autodl-tmp/app/RAG")
sys.path.insert(0, "/root/autodl-tmp/app")

from qdrant_client import QdrantClient
import config as C

OUT = sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/tables_dump.json"
client = QdrantClient(url="http://127.0.0.1:6333", timeout=60)

info = client.get_collection(C.TEXT_COLLECTION)
total = info.points_count
print("total points:", total, flush=True)

tables, n = [], 0
offset = None
while True:
    pts, offset = client.scroll(
        collection_name=C.TEXT_COLLECTION, limit=256, offset=offset,
        with_payload=True, with_vectors=False)
    for p in pts:
        pl = p.payload or {}
        if pl.get("has_table") or pl.get("table_html"):
            tables.append({
                "chunk_id": pl.get("chunk_id") or str(p.id),
                "source_stem": pl.get("source_stem"),
                "source_path": pl.get("source_path"),
                "page_start": pl.get("page_start"),
                "heading_path": pl.get("heading_path"),
                "content_type": pl.get("content_type"),
                "table_html": pl.get("table_html"),
                "content": pl.get("content"),
            })
    n += len(pts)
    if n % 4096 == 0:
        print(f"scanned {n}/{total}, tables so far {len(tables)}", flush=True)
    if offset is None or not pts:
        break

json.dump(tables, open(OUT, "w", encoding="utf-8"), ensure_ascii=False)
stems = {t["source_stem"] for t in tables}
print(f"DONE: scanned {n}, table chunks {len(tables)}, docs {len(stems)} -> {OUT}")
