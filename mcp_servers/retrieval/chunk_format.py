# -*- coding: utf-8 -*-
"""检索服务的 chunk 数据格式配置:Qdrant payload -> 结构化 dict。

供检索服务(service.py / engine_api.py)把命中点归一化成对外响应结构。
写入端(RAG/pdf/ingest.py)的字段命名与 _pid 算法必须与本文件保持一致。
"""
import os
import re
import sys
import uuid

# 裸 import config:把 config/ 加入 sys.path(服务启动时也会注入,此处兜底,
# 保证本模块被包方式导入时也能解析 config)。注意:不得把本目录(_HERE)插入
# sys.path——agent 进程会经 MCP 桥加载本包,目录里有 tools.py,插入会让
# ``import tools`` 解析到检索服务的工具模块而非 agent 工具包。
_HERE = os.path.dirname(os.path.abspath(__file__))                 # mcp_servers/retrieval
_PROJECT = os.path.dirname(os.path.dirname(_HERE))                 # project root
_CONFIG = os.path.join(_PROJECT, "config")
if _CONFIG not in sys.path:
    sys.path.insert(0, _CONFIG)

import config as C  # noqa: E402
from . import image_s3  # noqa: E402


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
        # 本地绝对路径签名为 TOS 时效 HTTPS 链接;未配置对象存储时原样返回本地路径
        "image_urls": [image_s3.image_url(p) for p in (payload.get("image_paths") or [])],
        "image_descriptions": payload.get("image_descriptions") or [],
        "content_type": payload.get("content_type") or "",
        # 视频块(mp4 入库):video_path 走独立视频存储(VID_*,缺省回落 TOS)时效签名
        "video_url": image_s3.video_url(payload["video_path"])
        if payload.get("video_path") else "",
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
        # 签名为 TOS 时效 HTTPS 链接;未配置对象存储时回退本地绝对路径
        "image_url": image_s3.image_url(payload.get("image_path", "")),
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
