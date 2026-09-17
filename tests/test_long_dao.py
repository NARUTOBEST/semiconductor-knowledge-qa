# -*- coding: utf-8 -*-
"""长期记忆 DAO 单测:用 fake PG 连接/游标,断言打到正确分片表、SQL 与参数。

不依赖真实 PostgreSQL:monkeypatch long_term._pg.get_pg 返回内存 fake,
记录所有 cur.execute(sql, params) 调用供断言。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib  # noqa: E402

# long_term 内部 `from . import pg as _pg`;取模块对象以便 monkeypatch 其 get_pg。
lt_module = importlib.import_module("memories.storage.long.long_term")
from memories.storage.long.sharding import table_for, all_shard_tables  # noqa: E402


class FakeCursor:
    def __init__(self):
        self.executes = []          # [(sql, params)]
        self.rowcount = 0
        self._fetchone_q = []       # fetchone 依次返回
        self._fetchall = None

    def execute(self, sql, params=None):
        self.executes.append((sql, params))

    def fetchone(self):
        return self._fetchone_q.pop(0) if self._fetchone_q else None

    def fetchall(self):
        return self._fetchall or []

    # context manager
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeConn:
    def __init__(self):
        self.cursors = []

    def cursor(self):
        c = FakeCursor()
        self.cursors.append(c)
        return c


def _patch_pg(monkeypatch):
    conn = FakeConn()
    monkeypatch.setattr(lt_module._pg, "get_pg", lambda force=False: conn)
    return conn, lt_module.long_term


def test_setup_creates_static_and_all_shards(monkeypatch):
    conn, mem = _patch_pg(monkeypatch)
    ok = mem.setup()
    assert ok is True
    sqls = [s for c in conn.cursors for (s, _) in c.executes]
    # 第一条是静态部分(扩展 + user_profile),之后每张分片表一条
    assert any("CREATE EXTENSION" in s for s in sqls)
    assert any("user_profile" in s for s in sqls)
    for t in all_shard_tables():
        assert any(t in s for s in sqls), f"缺少分片表 {t} 的建表语句"
    # 静态 1 条 + 每分片 1 条
    assert len(sqls) == 1 + len(all_shard_tables())


def test_setup_returns_false_when_pg_unavailable(monkeypatch):
    monkeypatch.setattr(lt_module._pg, "get_pg", lambda force=False: None)
    assert lt_module.long_term.setup() is False


def test_upsert_with_key_hits_shard_and_on_conflict(monkeypatch):
    conn, mem = _patch_pg(monkeypatch)
    mem.upsert_memory("alice", category="language", key="response_language",
                      content="默认用中文回答", importance=0.8)
    cur = conn.cursors[-1]
    sql, params = cur.executes[-1]
    assert table_for("alice") in sql
    assert "INSERT" in sql and "ON CONFLICT" in sql
    # 参数顺序:username, category, key, content, vec, importance, source_thread
    assert params[0] == "alice"
    assert params[1] == "language"
    assert params[2] == "response_language"
    assert params[3] == "默认用中文回答"
    assert params[5] == 0.8


def test_upsert_without_key_no_embedding_inserts_null_key(monkeypatch):
    conn, mem = _patch_pg(monkeypatch)
    mem.upsert_memory("bob", category="fact", key=None,
                      content="用户关注 ALD 设备", embedding=None)
    cur = conn.cursors[-1]
    sql, params = cur.executes[-1]
    assert table_for("bob") in sql
    assert "INSERT" in sql and "ON CONFLICT" not in sql
    # 无 key 插入参数:username, category, content, vec(None), importance, source_thread
    assert params[0] == "bob"
    assert params[1] == "fact"
    assert params[2] == "用户关注 ALD 设备"
    assert params[3] is None  # 无向量


def test_upsert_without_key_dedup_miss_then_insert(monkeypatch):
    conn, mem = _patch_pg(monkeypatch)
    cur_holder = {}
    orig_cursor = conn.cursor

    def cursor():
        c = orig_cursor()
        cur_holder["c"] = c
        return c
    conn.cursor = cursor
    vec = [0.1] * 1024
    mem.upsert_memory("carol", category="topic_interest", key=None,
                      content="关注刻蚀工艺", embedding=vec)
    cur = cur_holder["c"]
    # 近邻查询 fetchone -> None(无近邻)=> 判重未命中 => 插入
    sqls = [s for (s, _) in cur.executes]
    assert any("ORDER BY embedding <=>" in s for s in sqls)  # 近邻查询
    assert any("INSERT" in s for s in sqls)                  # 随后插入


def test_search_relevant_parses_rows_and_bumps_hits(monkeypatch):
    conn, mem = _patch_pg(monkeypatch)
    cur_holder = {}
    orig = conn.cursor

    def cursor():
        c = orig()
        c._fetchall = [
            (10, "language", "response_language", "默认中文", 0.8, 0.12),
            (11, "topic_interest", None, "关注 ALD", 0.6, 0.31),
        ]
        cur_holder["c"] = c
        return c
    conn.cursor = cursor
    hits = mem.search_relevant("alice", [0.2] * 1024, k=5)
    cur = cur_holder["c"]
    assert len(hits) == 2
    assert hits[0]["id"] == 10 and hits[0]["content"] == "默认中文"
    assert abs(hits[0]["distance"] - 0.12) < 1e-6
    sql, params = cur.executes[0]
    assert table_for("alice") in sql and "ORDER BY embedding <=>" in sql
    # 命中后有热度更新
    assert any("hit_count = hit_count + 1" in s for (s, _) in cur.executes)


def test_search_empty_when_no_query_vec(monkeypatch):
    conn, mem = _patch_pg(monkeypatch)
    assert mem.search_relevant("alice", None) == []
    assert conn.cursors == []  # 根本没碰 PG


def test_get_profile_parses_tuple(monkeypatch):
    conn, mem = _patch_pg(monkeypatch)
    cur_holder = {}
    orig = conn.cursor

    def cursor():
        c = orig()
        c._fetchone_q = [({"lang": "zh"}, "半导体工程师", ["ALD", "刻蚀"], 7)]
        cur_holder["c"] = c
        return c
    conn.cursor = cursor
    prof = mem.get_profile("alice")
    assert prof["summary"] == "半导体工程师"
    assert prof["fact_count"] == 7
    assert prof["top_interests"] == ["ALD", "刻蚀"]
    sql, _ = cur_holder["c"].executes[0]
    assert "FROM user_profile" in sql


def test_get_profile_none_when_no_row(monkeypatch):
    conn, mem = _patch_pg(monkeypatch)
    assert mem.get_profile("nobody") is None


def test_delete_user_cascades_profile_and_all_shards(monkeypatch):
    conn, mem = _patch_pg(monkeypatch)
    # 每张分片 DELETE 的 rowcount 记为 1
    cur_holder = {}
    orig = conn.cursor

    def cursor():
        c = orig()
        c.rowcount = 1
        cur_holder["c"] = c
        return c
    conn.cursor = cursor
    deleted = mem.delete_user("alice")
    cur = cur_holder["c"]
    sqls = [s for (s, _) in cur.executes]
    assert any("DELETE FROM user_profile" in s for s in sqls)
    for t in all_shard_tables():
        assert any(f"DELETE FROM {t}" in s for s in sqls)
    # user_profile 删除不计入,只计分片条目数
    assert deleted == len(all_shard_tables())


def test_delete_user_zero_when_pg_down(monkeypatch):
    monkeypatch.setattr(lt_module._pg, "get_pg", lambda force=False: None)
    assert lt_module.long_term.delete_user("alice") == 0
