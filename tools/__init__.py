# -*- coding: utf-8 -*-
"""工具层包:统一注册表 + dispatch 分发器 + MCP 工具桥。

目录结构:
  base.py           ToolSpec / Category / ErrorType / TruncatePolicy
  registry.py       全局 registry 单例
  dispatch.py       按 (name,args) 单次路由执行(从 registry 取 spec,调一次 handler)
  mcp_bridge.py     MCP 工具桥:连接外部 MCP server,把工具取回注册进 registry
  mcp_policies.py   MCP 工具的 Agent 侧策略表(分类/来源提取/截断/超时重试)

注:检索三件套(search_text/search_image/get_chunk)已迁至外部 MCP 服务
  mcp_servers/retrieval,经本包的 MCP 桥在 import 期注册,不再是本地子包。
  以后新增工具一律:外部服务以 MCP 暴露 -> 本桥自动取回注册(MCP_SERVERS 配置)。

注:重试/熔断/限流/错误分类等「机械重试」韧性机制已上移到
  agent_reasoning.ReAct.support.tool_resilience(中间件)+ tool_circuit(熔断器);
  参数校验/schema 检查在 ReAct 的 validate_runtime 节点。tools 层只保留工具能力本身。

运行时 schema 统一经 ``registry.schemas()`` 动态获取(含已注册且启用的全部工具)。
本包在导入时把 config/、RAG/ 加入 sys.path,使子模块的裸 import 可解析。

对外接口:
    from tools import dispatch, registry
    from tools import Category, ErrorType, ToolSpec, TruncatePolicy, ALL_CATEGORIES
"""
import os as _os
import sys as _sys

# 子模块用裸 import config(config 是无 __init__.py 的目录,需把 config/、RAG/
# 显式加入 sys.path 才能解析到 config.py)。生产链由 chat 包、测试由 conftest
# 注入;此处兜底,保证裸 ``import tools`` 也可用。
_HERE = _os.path.dirname(_os.path.abspath(__file__))           # tools/
_PROJECT = _os.path.dirname(_HERE)                             # project root
for _p in (_os.path.join(_PROJECT, "config"),
           _os.path.join(_PROJECT, "RAG")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

from . import memory_tool as _memory_tool_mod  # noqa: F401  (recall_memory:Category.MEMORY)
from . import mcp_bridge as _mcp_bridge  # noqa: F401

from .base import (  # noqa: F401
    Category, ErrorType, ToolSpec, TruncatePolicy, ALL_CATEGORIES,
)
from .registry import registry  # noqa: F401
from .dispatch import dispatch  # noqa: F401

# 启动 MCP 工具桥(连接外部 MCP server,取回工具注册进 registry):
# 有界等待首次注册(MCP_CONNECT_TIMEOUT_S,默认 3s),连不上后台重试、不阻塞。
# 必须在本包 import 期同步执行,保证 agent 图编译取 schema 前工具已就位。
_mcp_bridge.start_all()

__all__ = [
    "dispatch",
    "registry", "ToolSpec", "Category", "ErrorType", "TruncatePolicy",
    "ALL_CATEGORIES",
]
