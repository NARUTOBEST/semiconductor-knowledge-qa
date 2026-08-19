# -*- coding: utf-8 -*-
"""工具层共享基础设施:stage2 路径注入 + query/config 导入 + 块->结构化 dict 转换。

各工具文件(search_text / search_image / get_chunk)只 import 本模块,
不直接依赖 stage2 的裸 import 约定,保持工具文件干净。
"""
import os
import re
import sys
import uuid

# stage2 内部用裸 import(import config / import embed / import query),
# 把 stage2 目录加到 sys.path 首位,使其内部裸 import 能正确解析。
_HERE = os.path.dirname(os.path.abspath(__file__))                 # tools
_PROJECT = os.path.dirname(_HERE)                 # project root
_RAG = os.path.join(_PROJECT, "RAG")
_CONFIG = os.path.join(_PROJECT, "config")
for _p in (_RAG, _CONFIG):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import query as Q      # noqa: E402
import config as C     # noqa: E402


def _pid(chunk_id):
    """chunk_id -> Qdrant 点 id(与 ingest._pid 完全一致:uuid5(NAMESPACE_DNS, chunk_id))。"""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))


def _text_dict(payload, score=None):
    """文本块 payload -> 结构化 dict。"""
    payload = payload or {}
    return {
        "chunk_id": payload.get("chunk_id"),
        "source_stem": payload.get("source_stem", ""),
        "source_path": payload.get("source_path", ""),
        "page_num": payload.get("page_start"),        # 归一化页码(与 page_start 一致)
        "page_start": payload.get("page_start"),
        "page_end": payload.get("page_end"),
        "heading_path": payload.get("heading_path", ""),
        "content": payload.get("content", ""),         # 内容
        "char_count": payload.get("char_count", 0),
        "chunk_index": payload.get("chunk_index"),
        "has_table": bool(payload.get("has_table", False)),
        "table_html": payload.get("table_html"),
        "image_paths": payload.get("image_paths") or [],
        "image_descriptions": payload.get("image_descriptions") or [],
        "score": score,
    }


def _image_dict(payload, score=None):
    """图像块 payload -> 结构化 dict。内容 = caption + description。"""
    payload = payload or {}
    caption = payload.get("caption", "") or ""
    description = payload.get("description", "") or ""
    content = "\n".join(s for s in (caption, description) if s)
    return {
        "chunk_id": payload.get("chunk_id"),
        "source_stem": payload.get("source_stem", ""),
        "source_path": payload.get("source_path", ""),
        "page_num": payload.get("page_num"),
        "item_type": payload.get("item_type", ""),
        "content": content,                            # 内容(caption + description)
        "caption": caption,
        "description": description,
        "image_path": payload.get("image_path", ""),
        "parent_text_chunk_id": payload.get("parent_text_chunk_id"),
        "chunk_index": payload.get("chunk_index"),
        "content_type": payload.get("content_type"),   # portrait 标签(若有)
        "score": score,
    }


def _collection_hint(chunk_id):
    """由 chunk_id 后缀推断所属库:__t##### -> 文本库;__i##### -> 图像库;未知 -> None。"""
    if re.search(r"__t\d+$", chunk_id):
        return C.TEXT_COLLECTION
    if re.search(r"__i\d+$", chunk_id):
        return C.IMAGE_COLLECTION
    return None
