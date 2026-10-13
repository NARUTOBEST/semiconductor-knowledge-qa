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


# 大 PDF 清洗拆分段 stem(如 SEMI 标准合集_p3501-4000):payload 页码是段内 0 基,
# 在线打开要换算回原合集的全局物理页 = 段起始页 + 段内页码。
_SPLIT_STEM_RE = re.compile(r"^(.*)_p(\d+)-(\d+)$")


def _doc_page(source_stem, page_num):
    """payload 页码 -> 原始 PDF 的 1 基物理页(前端 #page= 锚点用)。

    - 拆分段 stem:_p<x>-<y> 的 x 即段起始全局页,全局页 = x + 段内 0 基页码;
    - 普通文档:minerU page_idx 为 0 基,物理页 = page_idx + 1;
    - 页码缺失/异常返回 None(前端不追加锚点)。
    """
    if not page_num or page_num < 0:
        return None
    m = _SPLIT_STEM_RE.match(str(source_stem or ""))
    try:
        return int(m.group(2)) + int(page_num) if m else int(page_num) + 1
    except (ValueError, TypeError):
        return None


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
        # 视频块(mp4 入库):优先用入库关联的已验证 video_key(TOS)直签;
        # 缺失再按 video_path 路径换算(统一 TOS,VID_* 已下线)
        "video_url": (
            image_s3.presigned(payload.get("video_key"))
            or image_s3.video_url(payload["video_path"])
        ) if (payload.get("video_path") or payload.get("video_key")) else "",
        # 原文档在线打开(答案来源可点击):签名直链 + 原始 PDF 1 基物理页
        # 优先用入库关联的已验证 doc_key(TOS),缺失再按 source_path 换算
        "doc_url": (image_s3.presigned(payload.get("doc_key"), inline=True)
                    or image_s3.doc_url(payload.get("source_path", ""))),
        "doc_page": _doc_page(payload.get("source_stem"), payload.get("page_start")),
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
        # 优先用入库关联的已验证 image_key(TOS)直签;缺失再按路径换算
        "image_url": (image_s3.presigned(payload.get("image_key"))
                      or image_s3.image_url(payload.get("image_path", ""))),
        # 原文档在线打开(同文本块)
        "doc_url": (image_s3.presigned(payload.get("doc_key"), inline=True)
                    or image_s3.doc_url(payload.get("source_path", ""))),
        "doc_page": _doc_page(payload.get("source_stem"), payload.get("page_num")),
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
