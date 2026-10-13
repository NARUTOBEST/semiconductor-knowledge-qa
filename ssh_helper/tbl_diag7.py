# -*- coding: utf-8 -*-
"""7 题硬失败诊断:在其来源文档的全部 chunk 中按关键词找事实文本,
输出块形态(是否表格块/内容摘要),判定 分块缺失 vs 检索失败。"""
import sys, re, json
sys.path.insert(0, "/root/autodl-tmp/app/config")
sys.path.insert(0, "/root/autodl-tmp/app/RAG")
sys.path.insert(0, "/root/autodl-tmp/app")
from qdrant_client import QdrantClient
import config as C

client = QdrantClient(url="http://127.0.0.1:6333", timeout=90)

# (qid, source_stem 前缀, [(标签, 关键词regex)...])
PROBES = [
    ("dev_043", "双臂取晶圆机器人magnatran", [
        ("placement_repeatability", r"repeatab"),
        ("TIR", r"TIR|0\.1\s*mm"),
        ("spec_table", r"Specification|Rated|Item"),
    ]),
    ("dev_094", "WAFER  HANDLER Mi30", [
        ("peel_force", r"peel force|peelforce"),
        ("seal_factors", r"Seal Time|over.?compress"),
    ]),
    ("dev_098", "WAFER  HANDLER Mi30", [
        ("overspeed_def", r"1\.1 times|Overspeed"),
        ("servo_params", r"PA1_37|PA1_52|PA1_56"),
    ]),
    ("dev_114", "晶圆减薄strasbaugh Model 7AF", [
        ("vid_139", r"Coarse Wheel Wear"),
        ("vid_140", r"Fine Wheel Wear"),
        ("vid_num", r"\b139\b|\b140\b"),
        ("unreferenced", r"-1\.0[eE]\+?06|unreferenced|1\.0E\+6"),
    ]),
    ("dev_128", "双臂取晶圆机器人magnatran", [
        ("millidegrees", r"millidegree|decimal"),
        ("micrometers", r"micrometer"),
        ("units_table", r"unit"),
    ]),
    ("dev_033", "晶圆减薄strasbaugh Model 7AF", [
        ("coolant", r"grinding coolant|coolant"),
        ("gpm", r"6 GPM|30 PSI|50 PSI"),
        ("utility", r"UTILITY REQUIREMENTS|STEP 4"),
    ]),
    ("dev_198", "MKS 仪器手册光子技术", [
        ("grating_eq", r"sin|grating equation|dG"),
        ("grooves", r"groove"),
        ("diffraction_angle", r"diffraction"),
    ]),
]

for qid, stem_pat, probes in PROBES:
    print(f"\n########## {qid} (source~{stem_pat})", flush=True)
    # 滚动该文档全部块
    chunks, offset, scanned = [], None, 0
    while True:
        pts, offset = client.scroll(
            collection_name=C.TEXT_COLLECTION, limit=128, offset=offset,
            with_payload=True, with_vectors=False,
            scroll_filter={"must": [{"key": "source_stem", "match": {"text": stem_pat}}]})
        for p in pts:
            pl = p.payload or {}
            chunks.append((pl.get("content") or "", pl))
        scanned += len(pts)
        if offset is None or not pts:
            break
    print(f"  文档块数: {scanned}", flush=True)
    for label, pat in probes:
        hits = [(c, pl) for c, pl in chunks if re.search(pat, c, re.I)]
        tbl = sum(1 for _, pl in hits if pl.get("has_table") or pl.get("table_html"))
        print(f"  [{label}] 命中 {len(hits)} 块 (表块 {tbl})", flush=True)
        for c, pl in hits[:2]:
            mark = "TBL" if (pl.get("has_table") or pl.get("table_html")) else "txt"
            print(f"    <{mark}> {c[:150]}".replace(chr(10), " ⏎ "), flush=True)
