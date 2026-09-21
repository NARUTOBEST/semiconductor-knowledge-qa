# -*- coding: utf-8 -*-
"""MCP 工具的 Agent 侧策略表。

MCP 协议只传 name / description / inputSchema;ToolSpec 其余能力字段
(分类、来源提取、截断策略、超时重试、输入长度上限)是 Agent 侧关注点,
由本表按 ``"{server}:{tool}"`` 键补充;未命中的键回退 ToolSpec 默认值。

原 tools/retrieval 的 source_extractor(extract_doc_sources)与截断策略
(_RETRIEVAL_TRUNCATE)随迁移收编到这里——它们服务于来源卡片/上下文截断,
本就是 Agent 侧职责,不该塞进 MCP 工具服务。
"""
from .base import Category, TruncatePolicy


def extract_doc_sources(result):
    """从检索工具返回里提取来源条目。

    兼容文本与图像块。每条 content 截断到 160 字。
    返回字段与历史格式完全一致(不新增 source_type,缺省按本地文档处理):
      chunk_id / source_stem / page / heading / score / content
    多媒体透传字段(前端缩略图/放大与视频入口用,缺失则不下发):
      image_url(图像块签名链接,或文本块内嵌图取首张) / item_type /
      description(图像描述,截 160 字) / video_url(视频块签名链接)
    """
    out = []
    if not isinstance(result, list):
        return out
    for r in result:
        if not isinstance(r, dict) or not r.get("source_stem"):
            continue
        page = r.get("page_num") or r.get("page_start")
        heading = r.get("heading_path") or r.get("caption") or ""
        item = {
            "chunk_id": r.get("chunk_id", ""),
            "source_stem": r["source_stem"],
            "page": f"p{page}" if page else "",
            "heading": heading,
            "score": round(float(r.get("score") or 0), 4),
            "content": (r.get("content", "") or "")[:160],
        }
        # 多媒体透传:仅在有值且是可访问 URL(http(s))时下发,本地残留路径不给前端
        image_url = r.get("image_url") or next(
            iter(r.get("image_urls") or []), None)
        if image_url and str(image_url).startswith("http"):
            item["image_url"] = image_url
        if r.get("video_url") and str(r["video_url"]).startswith("http"):
            item["video_url"] = r["video_url"]
        if r.get("item_type"):
            item["item_type"] = r["item_type"]
        if r.get("description"):
            item["description"] = (r.get("description") or "")[:160]
        out.append(item)
    return out


# 截断策略(与原 context_management 硬编码一致)
_RETRIEVAL_TRUNCATE = TruncatePolicy(
    long_fields={"content", "table_html", "description"},
    short_fields={
        "chunk_id", "source_stem", "source_path", "source_path_orig",
        "page_num", "page_start", "page_end", "page",
        "score", "heading_path", "caption", "content_type", "image_path",
        "has_table", "heading", "char_count", "chunk_index", "item_type",
        "parent_text_chunk_id", "image_url", "video_url", "image_urls",
    },
    list_long_fields={"image_descriptions", "image_paths"},
)


# 策略表:键 "{server}:{tool}",值 merge 进 ToolSpec 构造参数。
# 未列出的字段用 ToolSpec 默认;未列出的工具用 category=EXTERNAL(故障独立隔离)。
POLICIES = {
    "retrieval:search_text": {
        "category": Category.RETRIEVAL,        # 与检索类别熔断/降档联动(原行为不变)
        "produces_sources": True,
        "source_extractor": extract_doc_sources,
        "truncate": _RETRIEVAL_TRUNCATE,
        "timeout": 30.0,
        "retry_times": 1,
        "max_input_length": {"query": 500},
    },
    "retrieval:search_image": {
        "category": Category.RETRIEVAL,
        "produces_sources": True,
        "source_extractor": extract_doc_sources,
        "truncate": _RETRIEVAL_TRUNCATE,
        "timeout": 30.0,
        "retry_times": 1,
        "max_input_length": {"query": 500},
    },
    "retrieval:get_chunk": {
        "category": Category.RETRIEVAL,
        "produces_sources": False,
        "source_extractor": None,
        "truncate": _RETRIEVAL_TRUNCATE,
        "timeout": 30.0,
        "retry_times": 1,
        "max_input_length": {"chunk_id": 200},
    },
}


def policy_for(server: str, tool: str) -> dict:
    """取某 server 上某工具的策略:优先 "{server}:{tool}",未命中返回空表(全默认)。"""
    return POLICIES.get(f"{server}:{tool}", {})
