# -*- coding: utf-8 -*-
"""长期记忆存储层(PostgreSQL + pgvector,按 user 哈希分表)。

- long_term:用户偏好 DAO(结构化 upsert + 向量语义召回 + 画像 + 注销级联)
- get_pg / ping_pg:PG 连接工厂(不可用返回 None,旁路降级)
- embed_texts / embed_one:复用检索微服务 BGE-m3 嵌入(失败返回 None)
"""
from .long_term import LongTermMemory, long_term
from .pg import get_pg, ping_pg
from .sharding import shard_for, table_for, all_shard_tables, SHARD_COUNT
from .embed import embed_texts, embed_one

__all__ = [
    "LongTermMemory", "long_term",
    "get_pg", "ping_pg",
    "shard_for", "table_for", "all_shard_tables", "SHARD_COUNT",
    "embed_texts", "embed_one",
]
