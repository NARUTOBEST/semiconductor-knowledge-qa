# -*- coding: utf-8 -*-
"""长期记忆 NULL 向量 backfill 单测:mock PG 游标 + 注入 embed_fn,不触网/不依赖真库。

覆盖:有候选且模型正常 -> 批量补嵌并 UPDATE;模型不可用(None)/抛错 -> 保留 NULL 下轮
再试、不更新;无候选 -> 不调用模型;PG 不可用 -> 全 0 不抛。
"""
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

lt_module = importlib.import_module("memories.storage.long.long_term")


class _FakeCursor:
    def __init__(self, conn):
        self.conn = conn
        self.rowcount = 0
        self._rows = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        if sql.strip().upper().startswith("SELECT"):
            self._rows = list(self.conn.candidates)
            self.conn.select_calls += 1
        else:  # UPDATE
            self.conn.update_calls += 1
            self.rowcount = 1

    def fetchall(self):
        return self._rows


class _FakeConn:
    def __init__(self, candidates):
        self.candidates = candidates
        self.select_calls = 0
        self.update_calls = 0

    def cursor(self):
        return _FakeCursor(self)


@pytest.fixture
def _patched(monkeypatch):
    def _set(candidates):
        conn = _FakeConn(candidates)
        monkeypatch.setattr(lt_module._pg, "get_pg", lambda force=False: conn)
        monkeypatch.setattr(lt_module, "all_shard_tables", lambda: ["long_mem_00"])
        return conn
    return _set


class TestBackfill:
    def test_fills_null_embeddings(self, _patched):
        conn = _patched([(1, "事实A"), (2, "事实B")])
        out = lt_module.long_term.backfill_missing_embeddings(
            lambda texts: [[0.1, 0.2, 0.3, 0.4] for _ in texts], limit=8)
        assert out["candidate"] == 2 and out["filled"] == 2
        assert out["remain"] == 0 and conn.update_calls == 2

    def test_embed_none_keeps_null(self, _patched):
        conn = _patched([(1, "事实A"), (2, "事实B")])
        out = lt_module.long_term.backfill_missing_embeddings(
            lambda texts: None, limit=8)
        assert out["filled"] == 0 and out["remain"] == 2
        assert out["skipped"] == 2 and conn.update_calls == 0

    def test_embed_raises_is_swallowed(self, _patched):
        _patched([(1, "事实A")])

        def _boom(texts):
            raise RuntimeError("8002 down")
        out = lt_module.long_term.backfill_missing_embeddings(_boom, limit=8)
        assert out["filled"] == 0 and out["remain"] == 1

    def test_no_candidate_does_not_call_embed(self, _patched):
        conn = _patched([])
        called = {"n": 0}

        def _embed(texts):
            called["n"] += 1
            return []
        out = lt_module.long_term.backfill_missing_embeddings(_embed, limit=8)
        assert out["candidate"] == 0 and out["filled"] == 0
        assert called["n"] == 0 and conn.update_calls == 0

    def test_pg_unavailable_returns_zero(self, monkeypatch):
        monkeypatch.setattr(lt_module._pg, "get_pg", lambda force=False: None)
        out = lt_module.long_term.backfill_missing_embeddings(
            lambda texts: [], limit=8)
        assert out == {"candidate": 0, "filled": 0, "remain": 0, "skipped": 0}
