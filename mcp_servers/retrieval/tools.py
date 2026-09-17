# -*- coding: utf-8 -*-
"""检索三件套的 MCP 工具声明(search_text / search_image / get_chunk)。

纯工具层:只声明 MCP 工具(名称/描述/参数 schema),业务执行全部委托
engine_api(向量检索引擎在本服务进程内,不在 Agent 进程)。刻意不 import
fastapi/uvicorn,便于 Agent 侧测试经 memory 传输在本进程直连
(list_tools -> 注册,不触网、不加载模型)。

Agent 侧经 tools/mcp_bridge.py 连接本服务取回工具;MCP 协议只传
name/description/inputSchema,ToolSpec 其余能力字段(分类/来源提取/截断/
超时重试)由 agent 侧 tools/mcp_policies.py 策略表补充。
"""
from typing import Annotated

from pydantic import Field

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from . import engine_api


def _run(fn, *args, **kw):
    """执行引擎函数;业务异常转 ToolError(消息原样送达客户端,不丢原因)。

    不转 ToolError 时 mcp 2.x 会包成 UnexpectedToolError,客户端只见
    "Error executing tool ..." 而丢失具体错误文本(重试分类/排障都靠它)。
    """
    try:
        return fn(*args, **kw)
    except ToolError:
        raise
    except Exception as e:
        raise ToolError(f"{type(e).__name__}: {e}") from e

mcp = MCPServer(
    name="retrieval",
    instructions=(
        "半导体设备/工艺知识库检索工具集:文本混合检索(search_text)、"
        "图像跨模态检索(search_image)、按块ID取完整内容(get_chunk)。"
    ),
)


@mcp.tool(
    name="search_text",
    description=(
        "在半导体设备/工艺文档的【文本库】做混合检索(dense+sparse,RRF 融合)。"
        "适合查概念、原理、操作流程、术语、设备型号等文本内容。"
        "返回结构化文本块列表(按相关度过滤,只留高相关结果,通常 1-3 条)。"
    ),
)
def search_text(
    query: Annotated[str, Field(
        description="检索查询,中文/英文/术语均可,如 'TMA 前驱体' 或 'wafer chuck 温度控制'。")],
    k: Annotated[int, Field(description="返回的条数(按相关度降序)。")] = 3,
    score_ratio: Annotated[float, Field(
        description="相关分阈值系数:保留 score >= 最高分*score_ratio 的结果,默认 0.6。")] = 0.6,
) -> list[dict]:
    """文本库混合检索(dense+sparse RRF)+ 精确代码兜底 + 重排。"""
    return _run(engine_api.search_text, query, k=k, score_ratio=score_ratio)


@mcp.tool(
    name="search_image",
    description=(
        "在【图像库】做跨模态检索(CLIP 文本->图 + BGE-m3 描述 RRF 融合)。"
        "适合查示意图、曲线图、设备外观照片、流程图等。"
        "返回结构化图块列表,每条含 source_stem / page_num / caption / description / image_path 等。"
        "默认过滤人像(content_type=portrait)。"
    ),
)
def search_image(
    query: Annotated[str, Field(
        description="图像检索查询,描述想找的图,如 'ALD 工艺原理示意图' 或 '前驱体饱和曲线图'。")],
    k: Annotated[int, Field(description="返回的条数(按相关度降序)。")] = 3,
    include_portraits: Annotated[bool, Field(
        description="是否包含人像/证件照(默认 False,过滤掉)。")] = False,
    score_ratio: Annotated[float, Field(
        description="相关分阈值系数:保留 score >= 最高分*score_ratio 的结果,默认 0.6。")] = 0.6,
) -> list[dict]:
    """图像库检索(CLIP+BGE-m3 双路 RRF)+ 重排。"""
    return _run(
        engine_api.search_image,
        query, k=k, include_portraits=include_portraits, score_ratio=score_ratio)


@mcp.tool(
    name="get_chunk",
    description=(
        "按 chunk_id 从 Qdrant 取【完整】的单个块/图(含全部字段,不截断)。"
        "用于拿到 search_text/search_image 命中块的完整内容(如完整 table_html、完整 description)。"
        "chunk_id 命名约定:<source_stem>__t##### 文本块 / __i##### 图像块。找不到返回 None。"
    ),
)
def get_chunk(
    chunk_id: Annotated[str, Field(
        description="块ID,形如 'FIJI_F200_ALD__i00072' 或 'Oxford ALD Operation Manual__t00001'。")],
) -> dict | None:
    """按 chunk_id 取完整块/图。"""
    return _run(engine_api.get_chunk, chunk_id)
