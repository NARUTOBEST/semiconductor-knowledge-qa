# -*- coding: utf-8 -*-
"""按《Agent评测集生成方案》白名单文档,从 Qdrant(ald_text)抓取出题素材。

仅抓取通过 source 隔离校验的 6 篇文档(dev 金标/途经文档全部排除,见
source_isolation_prep.json)。按 heading_path 分组落盘,另出标题索引供选读。

用法(项目根): python eval/pipeline/dev/fetch_material.py  # 本 track 文档;题集输出 eval/qa_sets/
依赖: 本地端口转发 127.0.0.1:16333 -> VM qdrant (ssh_helper/forward_6333.py 16333)
"""
import json
import os
import re
import sys
from collections import defaultdict

import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
PAGE = re.compile(r"__p\d+-\d+$|_p\d+-\d+$")
QDRANT = "http://127.0.0.1:16333"

# 旧 6 篇(20261001 首批) + 20261002 重建新增 11 篇(取消隔离红线后全库遴选)
TARGETS = {
    "01_CMP_6DS-SP": "晶圆CMP研磨机Strasbaugh 6DS-SP Production CMP Planarizer",
    "02_DB800_固晶机": "日立固晶机DB800_Manual-eng_ver01_012_00",
    "03_WAFER_HANDLER_Mi30": "WAFER  HANDLER Mi30 User Manual V5",
    "04_清洗技术_pv94": "半导体器件制造中的清洗技术pv_94_07",
    "05_减薄_7AF": "晶圆减薄strasbaugh Model 7AF",
    # ---- dev 200 新增 4 篇 ----
    "07_机器人_magnatran": "双臂取晶圆机器人magnatran_71",
    "08_清洗技术_VIII": "半导体器件制造中的清洗技术 VIII",
    "09_MKS仪器手册": "MKS 仪器手册光子技术原理与应用",
    # ---- heldout 200 新增 7 篇 ----
}


def scroll_stem(stem):
    """拉取单个 source_stem 的全部 chunk(content/heading/页码/表格标记)。"""
    out, offset = [], None
    while True:
        body = {
            "filter": {"must": [{"key": "source_stem", "match": {"value": stem}}]},
            "limit": 128,
            "with_payload": True,
        }
        if offset is not None:
            body["offset"] = offset
        req = urllib.request.Request(
            QDRANT + "/collections/ald_text/points/scroll",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        d = json.loads(urllib.request.urlopen(req, timeout=60).read())
        pts = d["result"]["points"]
        for p in pts:
            pl = p["payload"]
            out.append({
                "heading": pl.get("heading_path") or "",
                "page": pl.get("page_start"),
                "content": pl.get("content") or "",
                "has_table": bool(pl.get("has_table")),
            })
        nxt = d["result"].get("next_page_offset")
        if nxt is None or not pts:
            break
        offset = nxt
    return out


def main():
    facet = json.load(open(os.path.join(ROOT, "facet_ald_text.json"), encoding="utf-8"))
    mat_dir = os.path.join(HERE, "draft_material")
    os.makedirs(mat_dir, exist_ok=True)

    for tag, base in TARGETS.items():
        if len(sys.argv) > 1 and tag not in sys.argv[1:]:
            continue  # 命令行可传 tag 列表,只抓指定文档
        stems = [k for k in facet if PAGE.sub("", k).strip() == base]
        chunks = []
        for s in sorted(stems):
            chunks.extend(scroll_stem(s))
        # 按标题分组(保持文档顺序)
        groups = defaultdict(list)
        for c in chunks:
            h = c["heading"].strip() or "(无标题)"
            groups[h].append(c)
        fp = os.path.join(mat_dir, tag + ".txt")
        with open(fp, "w", encoding="utf-8") as f:
            for h, items in groups.items():
                pages = sorted({i["page"] for i in items if i["page"] is not None})
                f.write(f"\n===== [p{pages[0]}-{pages[-1]}] {h} ({len(items)}块) =====\n".replace("[pNone", "[p?"))
                for i in items:
                    mark = " [表]" if i["has_table"] else ""
                    pg = f"p{i['page']}" if i["page"] is not None else "p?"
                    f.write(f"--[{pg}{mark}]-- {i['content']}\n")
        index = [(h, len(v)) for h, v in sorted(groups.items(), key=lambda x: -len(x[1]))]
        with open(os.path.join(mat_dir, tag + "_index.json"), "w", encoding="utf-8") as f:
            json.dump({"doc": base, "chunks": len(chunks), "headings": index}, f, ensure_ascii=False, indent=1)
        print(f"{tag}: {len(chunks)} chunks / {len(groups)} headings -> {os.path.basename(fp)}")


if __name__ == "__main__":
    sys.exit(main())
