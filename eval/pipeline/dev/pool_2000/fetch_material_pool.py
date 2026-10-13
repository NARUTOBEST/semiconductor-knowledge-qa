# -*- coding: utf-8 -*-
"""备选池 2000 题素材抓取:selection.json 的 80 篇 -> draft_material/P*.txt。

与 pipeline/dev/fetch_material.py 同逻辑(scroll ald_text,按 heading 分组),
仅 TARGETS 换成 pool_2000/selection.json 遴选的 80 篇。
依赖:本地端口转发 127.0.0.1:16333 -> VM qdrant。
用法(任意目录): python fetch_material_pool.py [P01 P02 ...]
"""
import json
import os
import re
import sys
from collections import defaultdict

import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
PAGE = re.compile(r"__p\d+-\d+$|_p\d+-\d+$")
QDRANT = "http://127.0.0.1:16333"
OUT = os.path.join(HERE, "draft_material")


def scroll_stem(stem):
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
    sel = json.load(open(os.path.join(HERE, "selection.json"), encoding="utf-8"))
    facet = json.load(open(r"C:\project3\facet_ald_text.json", encoding="utf-8"))
    os.makedirs(OUT, exist_ok=True)
    only = set(sys.argv[1:])

    for i, (base, _c) in enumerate(sel, 1):
        tag = f"P{i:02d}"
        if only and tag not in only:
            continue
        stems = [k for k in facet if PAGE.sub("", k).strip() == base]
        chunks = []
        for s in sorted(stems):
            chunks.extend(scroll_stem(s))
        groups = defaultdict(list)
        for c in chunks:
            h = c["heading"].strip() or "(无标题)"
            groups[h].append(c)
        fp = os.path.join(OUT, tag + ".txt")
        with open(fp, "w", encoding="utf-8") as f:
            for h, items in groups.items():
                pages = sorted({x["page"] for x in items if x["page"] is not None})
                f.write(f"\n===== [p{pages[0]}-{pages[-1]}] {h} ({len(items)}块) =====\n".replace("[pNone", "[p?"))
                for x in items:
                    mark = " [表]" if x["has_table"] else ""
                    pg = f"p{x['page']}" if x["page"] is not None else "p?"
                    f.write(f"--[{pg}{mark}]-- {x['content']}\n")
        index = [(h, len(v)) for h, v in sorted(groups.items(), key=lambda x: -len(x[1]))]
        json.dump({"doc": base, "chunks": len(chunks), "headings": index[:80]},
                  open(os.path.join(OUT, tag + "_index.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print(f"{tag}: {len(chunks)} chunks / {len(groups)} headings / stems {len(stems)} -> {base[:40]}", flush=True)


if __name__ == "__main__":
    main()
