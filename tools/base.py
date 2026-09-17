# -*- coding: utf-8 -*-
"""工具层统一抽象:Category / ErrorType / TruncatePolicy / ToolSpec。

每个工具用 ToolSpec 声明元信息、schema、执行 handler、来源/截断策略、
超时重试策略,注册到 Registry 后即可被 dispatch 与 nodes 自动识别。

本模块不依赖 registry/dispatch,避免循环 import。
"""
from __future__ import annotations

import contextvars
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx


# 工具 handler 在命中结果缓存时置 True,dispatch 读取后复位,用于 metrics 记账。
# 用 ContextVar 保证并发(多请求线程/协程)下互不干扰。
cache_hit_var: "contextvars.ContextVar[bool]" = contextvars.ContextVar(
    "tool_cache_hit", default=False)


def mark_cache_hit() -> None:
    """handler 命中结果缓存时调用,标记本次 dispatch 调用为缓存命中。"""
    cache_hit_var.set(True)


# ==================== 分类 ====================
class Category:
    """工具 category 常量。"""
    RETRIEVAL = "retrieval"      # 本地知识库检索
    MEMORY = "memory"            # 记忆召回(长期偏好/近期对话);故障独立隔离,不影响检索
    EXTERNAL = "external"        # 外部 MCP 工具(默认);故障独立隔离,不影响检索/记忆


ALL_CATEGORIES = (Category.RETRIEVAL, Category.MEMORY, Category.EXTERNAL)


# ==================== 错误分类 ====================
class ErrorType:
    """dispatch 统一错误结构的 error_type 取值。"""
    TIMEOUT = "timeout"
    RETRYABLE = "retryable"      # 网络抖动 / 5xx / 限流(429)
    FATAL = "fatal"              # 参数错误 / 4xx(非 429) / 内容违规 / 未知工具
    CIRCUIT_OPEN = "circuit_open"
    UNKNOWN = "unknown"


def make_error(message: str, error_type: str, tool: str) -> dict:
    """构造统一错误结构。

    向后兼容:仍是 dict 且含 "error" 键,现有 ``"error" in result`` 判定不变;
    新增 error_type / tool 为增量字段。
    """
    return {"error": message, "error_type": error_type, "tool": tool}


# ==================== 重试谓词 ====================
def default_retry_on(exc: BaseException) -> bool:
    """默认重试判定:网络超时 / 传输层故障(连接拒绝/重置/断连) / 5xx / 429 可重试,
    其余(含其它 4xx、协议/URL 配置错误)不重试。

    不能只用异常类型元组——httpx.HTTPStatusError 覆盖所有 4xx/5xx,
    必须在谓词内检查 response.status_code。
    httpx.TransportError 是 TimeoutException 与 ConnectError/ReadError 等网络故障的基类
    (不含 HTTPStatusError),微服务瞬时不可用/连接被拒属可重试。
    """
    if isinstance(exc, httpx.HTTPStatusError):
        resp = getattr(exc, "response", None)
        code = getattr(resp, "status_code", 0) or 0
        return code >= 500 or code == 429
    if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
        return True
    return False


# ==================== 截断策略 ====================
@dataclass
class TruncatePolicy:
    """工具结果截断策略(供 context_management.truncate_tool_result 使用)。

    - long_fields:      这些字符串字段截到 max_chars
    - short_fields:     这些字段原样保留(即使很长也不截断)
    - list_long_fields: 这些 list 字段先 join 再按长文本截断
    - max_chars_override: 覆盖全局 CONTEXT_TOOL_RESULT_MAX_CHARS
    - custom_fn:        自定义截断函数 (result, max_chars) -> result,优先级最高
    """
    long_fields: set = field(default_factory=set)
    short_fields: set = field(default_factory=set)
    list_long_fields: set = field(default_factory=set)
    max_chars_override: int | None = None
    custom_fn: Callable | None = None
    # True 且结果为纯字符串时:按 max_chars 原样截断返回(不走 json.dumps,保留换行),
    # 用于 recall_memory 这类返回"给 LLM 读的文本块"而非结构化 JSON 的工具。
    passthrough_text: bool = False


# ==================== ToolSpec ====================
@dataclass
class ToolSpec:
    """一个工具的完整声明。

    handler 约定:
      - 只负责业务执行,不做重试/熔断/参数校验;
      - 成功返回裸结果(list/dict/str/...);
      - 失败抛异常,由 ReAct 层韧性中间件(tool_resilience)分类并重试/熔断;
        或返回已自带 "error" 键的 dict(视为业务失败,不重试,交中间件上交 reflect)。
    """
    name: str
    description: str
    category: str
    parameters: dict                       # OpenAI function-calling JSON Schema
    handler: Callable[..., Any]

    produces_sources: bool = False         # True -> 结果进 collected_sources(来源卡片/引用)
    enabled: bool = True                   # False = 不 bind 给 LLM

    # 执行策略
    timeout: float = 30.0
    retry_times: int = 0
    retry_on: Callable[[BaseException], bool] = field(default=default_retry_on)

    # 结果处理
    truncate: TruncatePolicy = field(default_factory=TruncatePolicy)
    source_extractor: Callable[[Any], list] | None = None

    # 安全:参数名 -> 最大字符长度(超限返回 FATAL;未列出的字符串参数不校验)
    max_input_length: dict = field(default_factory=dict)

    def to_schema(self) -> dict:
        """导出 OpenAI function-calling schema dict。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def required_params(self) -> set:
        """parameters.declared 的 required 参数名集合。"""
        return set((self.parameters or {}).get("required") or [])
