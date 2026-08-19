# -*- coding: utf-8 -*-
"""检索工具集合:三个工具函数(HTTP 调用检索微服务)+ OpenAI function-calling schema。

工具函数通过 HTTP 调用检索微服务(retrieval_service.py, :8002),
不再在 Agent 进程中加载 BGE-m3/CLIP/Reranker。
超时 = 关 HTTP 连接,无僵尸线程。

对外(经 tools 包再导出):
    search_text(query, k=3)          文本库检索
    search_image(query, k=5, ...)    图像库检索
    get_chunk(chunk_id)              取完整块
    search_tools                     function-calling schema 列表
    SEARCH_TOOLS_BY_NAME             name -> schema
"""
import os
import logging
import httpx

logger = logging.getLogger("tools")

# 检索微服务地址
_RETRIEVAL_URL = os.getenv("RETRIEVAL_SERVICE_URL", "http://127.0.0.1:8002")
_client = httpx.Client(base_url=_RETRIEVAL_URL, timeout=30)


# ==================== 工具实现(HTTP 调用)====================

def search_text(query, k=3, score_ratio=0.6):
    """文本库检索(HTTP 调用检索微服务)。"""
    try:
        resp = _client.post("/search_text", json={
            "query": query, "k": k, "score_ratio": score_ratio,
        })
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.warning(f"search_text 失败: {e}")
        return {"error": "检索服务不可用"}


def search_image(query, k=3, include_portraits=False, score_ratio=0.6):
    """图像库检索(HTTP 调用检索微服务)。"""
    try:
        resp = _client.post("/search_image", json={
            "query": query, "k": k,
            "include_portraits": include_portraits, "score_ratio": score_ratio,
        })
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.warning(f"search_image 失败: {e}")
        return {"error": "检索服务不可用"}


def get_chunk(chunk_id):
    """按 chunk_id 取完整块(HTTP 调用检索微服务)。"""
    try:
        resp = _client.post("/get_chunk", json={"chunk_id": chunk_id})
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.warning(f"get_chunk 失败: {e}")
        return {"error": "检索服务不可用"}


# ==================== OpenAI function-calling schema(不变)====================
search_tools = [
    {
        "type": "function",
        "function": {
            "name": "search_text",
            "description": (
                "在半导体设备/工艺文档的【文本库】做混合检索(dense+sparse,RRF 融合)。"
                "适合查概念、原理、操作流程、术语、设备型号等文本内容。"
                "返回结构化文本块列表(按相关度过滤,只留高相关结果,通常 1-3 条)。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "检索查询,中文/英文/术语均可,如 'TMA 前驱体' 或 'wafer chuck 温度控制'。",
                    },
                    "k": {
                        "type": "integer",
                        "description": "返回的条数(按相关度降序)。",
                        "default": 3,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_image",
            "description": (
                "在【图像库】做跨模态检索(CLIP 文本->图 + BGE-m3 描述 RRF 融合)。"
                "适合查示意图、曲线图、设备外观照片、流程图等。"
                "返回结构化图块列表,每条含 source_stem / page_num / caption / description / image_path 等。"
                "默认过滤人像(content_type=portrait)。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "图像检索查询,描述想找的图,如 'ALD 工艺原理示意图' 或 '前驱体饱和曲线图'。",
                    },
                    "k": {
                        "type": "integer",
                        "description": "返回的条数(按相关度降序)。",
                        "default": 3,
                    },
                    "include_portraits": {
                        "type": "boolean",
                        "description": "是否包含人像/证件照(默认 False,过滤掉)。",
                        "default": False,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_chunk",
            "description": (
                "按 chunk_id 从 Qdrant 取【完整】的单个块/图(含全部字段,不截断)。"
                "用于拿到 search_text/search_image 命中块的完整内容(如完整 table_html、完整 description)。"
                "chunk_id 命名约定:<source_stem>__t##### 文本块 / __i##### 图像块。找不到返回 None。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "chunk_id": {
                        "type": "string",
                        "description": "块ID,形如 'FIJI_F200_ALD__i00072' 或 'Oxford ALD Operation Manual__t00001'。",
                    },
                },
                "required": ["chunk_id"],
            },
        },
    },
]

SEARCH_TOOLS_BY_NAME = {t["function"]["name"]: t["function"] for t in search_tools}
