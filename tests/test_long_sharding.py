# -*- coding: utf-8 -*-
"""长期记忆【分表路由】单测:确定性、稳定、落 [0,N)、表名命名。

不依赖 PG / 网络,纯函数验证 blake2b(username) % N 分片。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memories.storage.long import sharding as sh  # noqa: E402
from memories.storage.long import (  # noqa: E402
    shard_for, table_for, all_shard_tables, SHARD_COUNT,
)


def test_shard_in_range():
    for name in ["alice", "bob", "carol", "dave", "用户A", "用户B", "x" * 50]:
        s = shard_for(name)
        assert 0 <= s < SHARD_COUNT


def test_shard_deterministic_and_stable():
    # 同一 username 多次 / 任意进程都落同一分片(纯哈希,无随机)
    assert shard_for("alice") == shard_for("alice")
    assert table_for("alice") == table_for("alice")


def test_empty_username_shard_zero():
    assert shard_for("") == 0
    assert shard_for(None) == 0


def test_table_name_matches_shard():
    for name in ["alice", "bob", "carol"]:
        tbl = table_for(name)
        assert tbl.startswith("long_mem_")
        # 表名尾部的分片号应与 shard_for 一致
        suffix = tbl.split("long_mem_")[1]
        assert int(suffix) == shard_for(name)


def test_all_shard_tables_complete():
    tables = all_shard_tables()
    assert len(tables) == SHARD_COUNT
    assert len(set(tables)) == SHARD_COUNT  # 无重名
    for i, t in enumerate(tables):
        assert t.startswith("long_mem_")
    # table_for 的结果一定在全量枚举里(召回/删除只打存在的表)
    assert table_for("alice") in tables


def test_distribution_spreads():
    # 大量用户名不应全部挤在同一分片(冒烟:至少落到 >1 张表)
    seen = {shard_for(f"user_{i}") for i in range(200)}
    assert len(seen) > 1
