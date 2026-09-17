# -*- coding: utf-8 -*-
"""长期记忆【分表路由】:按 username 哈希把用户固定落到一张分片表。

策略(单库分表):长期偏好表按用户分片,共 LONG_MEM_SHARD_COUNT 张(默认 16),
表名 long_mem_00 .. long_mem_{N-1:02d}。同一 username 恒定落同一张表,因此该用户的
向量召回 / 去重 / 注销清理都只需打一张表;未来要扩到分库时,把 shard -> (库,表) 的
映射在 table_for / 连接选择处扩展即可。

哈希用 blake2b(确定性、分布均匀、无需第三方库);取模后落 [0, SHARD_COUNT)。
"""
from __future__ import annotations

import hashlib
import os
import sys

_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "config"))
import config as C  # noqa: E402

SHARD_COUNT = max(1, int(getattr(C, "LONG_MEM_SHARD_COUNT", 16)))

_TABLE_PREFIX = "long_mem_"


def shard_for(username: str) -> int:
    """返回 username 对应的分片号 [0, SHARD_COUNT)。空名落到 0。"""
    if not username:
        return 0
    digest = hashlib.blake2b(str(username).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % SHARD_COUNT


def table_for(username: str) -> str:
    """返回 username 对应的分片表名,如 long_mem_07。"""
    width = max(2, len(str(SHARD_COUNT - 1)))
    return f"{_TABLE_PREFIX}{shard_for(username):0{width}d}"


def all_shard_tables() -> list[str]:
    """全部分片表名(建表 / 兜底遍历用)。"""
    width = max(2, len(str(SHARD_COUNT - 1)))
    return [f"{_TABLE_PREFIX}{i:0{width}d}" for i in range(SHARD_COUNT)]
