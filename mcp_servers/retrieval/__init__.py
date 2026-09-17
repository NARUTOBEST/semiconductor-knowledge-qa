# -*- coding: utf-8 -*-
"""检索 MCP 服务:search_text / search_image / get_chunk 三件套。

目录结构:
  service.py      FastAPI 入口(:8002):MCP 挂载(/mcp)+ 传统 HTTP 端点
                  (/embed_text /rerank /ingest_document 供记忆嵌入/管理后台复用)
  tools.py        MCP 工具声明(纯工具层,不依赖 fastapi,可被进程内直连)
  engine_api.py   检索业务实现(向量召回 + 重排 + payload 归一化)
  query.py        Qdrant 检索引擎(自 RAG/query.py 迁来)
  chunk_format.py Qdrant payload -> 结构化 dict
  image_s3.py     图片对象存储(火山引擎 TOS)路径↔时效签名链接

Agent 侧不再持有任何检索工具代码(tools/retrieval 已删除),经 tools/mcp_bridge.py
以 MCP 客户端身份连接本服务,把工具取回注册进全局 registry。
"""
