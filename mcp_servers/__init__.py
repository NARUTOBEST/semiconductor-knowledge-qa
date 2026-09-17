# -*- coding: utf-8 -*-
"""外部 MCP 服务包:独立于 Agent 主进程的工具服务。

当前包含:
  retrieval/   半导体设备知识库检索服务(BGE-m3 + CLIP + Reranker + Qdrant),
               同时提供 MCP 工具接口(/mcp)与传统 HTTP 端点(:8002)。

以后所有 agent 工具都在此包(或同类外部服务)以 MCP 注册,由 agent 侧
tools/mcp_bridge.py 取回注册进全局 registry 使用。
"""
