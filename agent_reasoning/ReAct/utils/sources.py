# -*- coding: utf-8 -*-
"""从检索结果中提取来源条目(供前端显示引用条)。"""


def sources_from_result(result):
    """从检索工具(search_text/search_image)的返回里提取来源条目。

    兼容文本与图像块。每条截断 content 到 160 字。
    """
    out = []
    if not isinstance(result, list):
        return out
    for r in result:
        if not isinstance(r, dict) or not r.get("source_stem"):
            continue
        page = r.get("page_num") or r.get("page_start")
        heading = r.get("heading_path") or r.get("caption") or ""
        out.append({
            "chunk_id": r.get("chunk_id", ""),
            "source_stem": r["source_stem"],
            "page": f"p{page}" if page else "",
            "heading": heading,
            "score": round(float(r.get("score") or 0), 4),
            "content": (r.get("content", "") or "")[:160],
        })
    return out
