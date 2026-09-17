# -*- coding: utf-8 -*-
"""短期记忆【事实表】(facts.py,Redis memf:*)单测:fakeredis,不依赖真实 redis。

覆盖:seq/键落位/TTL、规范化指纹去重(二次只 HINCRBY)、容量淘汰(ZPOPMIN、
promoted 优先)、降级裸写 degraded=1、mark_promoted、thread/user 级联删除。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib  # noqa: E402

import fakeredis  # noqa: E402
import config as C  # noqa: E402

facts_module = importlib.import_module("memories.storage.short.facts")
from memories.storage.short.facts import FactTable, fingerprint  # noqa: E402

TID = "alice|t1"


def _make(monkeypatch):
    fake = fakeredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(facts_module, "get_redis", lambda: fake)
    return FactTable(), fake


def test_fingerprint_normalizes():
    # 大小写/标点/空白差异应得同一指纹
    f1 = fingerprint("ALD 是什么？", "ALD 是原子层沉积。")
    f2 = fingerprint("ald是什么", "ALD是原子层沉积")
    assert f1 == f2
    f3 = fingerprint("完全不同的问题", "另一个答案")
    assert f3 != f1


class TestInsertAndDedup:
    def test_insert_keys_and_ttl(self, monkeypatch):
        ft, fake = _make(monkeypatch)
        r = ft.add_or_touch(TID, q="问题", a="答案")
        assert r["fid"] == "f1" and r["duplicate"] is False
        assert fake.zcard("memf:facts:alice|t1") == 1
        h = fake.hgetall("memf:fact:alice|t1:f1")
        assert h["q"] == "问题" and h["a"] == "答案"
        assert h["freq"] == "1" and h["promoted"] == "0" and h["degraded"] == "0"
        # 指纹索引 + keys 集合 + user 集合
        assert fake.hget("memf:idx:fp:alice|t1", h["fp"]) == "f1"
        assert "memf:fact:alice|t1:f1" in fake.smembers("memf:keys:alice|t1")
        assert "alice|t1" in fake.smembers("memf:user:alice")
        # TTL 已设置(24h)
        assert fake.ttl("memf:fact:alice|t1:f1") > 0

    def test_duplicate_refreshes_freq_no_new(self, monkeypatch):
        ft, fake = _make(monkeypatch)
        ft.add_or_touch(TID, q="重复问题", a="重复答案")
        r2 = ft.add_or_touch(TID, q="重复问题", a="重复答案")
        assert r2["duplicate"] is True and r2["fid"] == "f1"
        assert fake.zcard("memf:facts:alice|t1") == 1
        assert fake.hget("memf:fact:alice|t1:f1", "freq") == "2"

    def test_touch_if_exists(self, monkeypatch):
        ft, _ = _make(monkeypatch)
        assert ft.touch_if_exists(TID, fingerprint("q", "a")) is None
        ft.add_or_touch(TID, q="q", a="a")
        fp = fingerprint("q", "a")
        assert ft.touch_if_exists(TID, fp) == "f1"
        assert ft.touch_if_exists(TID, fingerprint("其他", "x")) is None


class TestDegraded:
    def test_degraded_raw_insert_skips_dedup(self, monkeypatch):
        ft, fake = _make(monkeypatch)
        ft.add_or_touch(TID, q="q", a="a", degraded=True)
        # 同样内容再裸写:跳过【查重】直接新增(不 HINCRBY 到 f1),共 2 条
        r = ft.add_or_touch(TID, q="q", a="a", degraded=True)
        assert r["duplicate"] is False and r["fid"] == "f2"
        assert ft.count(TID) == 2
        assert fake.hget("memf:fact:alice|t1:f1", "degraded") == "1"
        assert fake.hget("memf:fact:alice|t1:f2", "freq") == "1"


class TestPromotion:
    def test_mark_promoted_penalizes_score(self, monkeypatch):
        ft, fake = _make(monkeypatch)
        ft.add_or_touch(TID, q="q", a="a")
        s_before = fake.zscore("memf:facts:alice|t1", "f1")
        ft.mark_promoted(TID, "f1", long_ref="shard3")
        h = fake.hgetall("memf:fact:alice|t1:f1")
        assert h["promoted"] == "1" and h["long_ref"] == "shard3"
        assert fake.zscore("memf:facts:alice|t1", "f1") < s_before


class TestEviction:
    def test_cap_evicts_lowest_score(self, monkeypatch):
        # 淘汰下限为 10(_max_per_thread 内 max(10, ...)),故用 cap=10、插 11 条
        monkeypatch.setattr(C, "MEM_FACT_MAX_PER_THREAD", 10, raising=False)
        ft, fake = _make(monkeypatch)
        # 首条 promoted(惩罚分最低,应最先被淘汰)
        ft.add_or_touch(TID, q="q0", a="a0", importance=0.9)
        ft.mark_promoted(TID, "f1")
        for i in range(1, 11):
            ft.add_or_touch(TID, q=f"q{i}", a=f"a{i}", importance=0.9)
        # 已到 cap=10;第 11 条(f11)触发淘汰
        assert ft.count(TID) == 10
        ft.add_or_touch(TID, q="q11", a="a11", importance=0.9)
        assert ft.count(TID) == 10
        # f1(promoted,惩罚分最低)被淘汰,f11 保留
        assert fake.zscore("memf:facts:alice|t1", "f1") is None
        assert not fake.exists("memf:fact:alice|t1:f1")
        assert fake.zscore("memf:facts:alice|t1", "f11") is not None


class TestCascade:
    def test_delete_thread(self, monkeypatch):
        ft, fake = _make(monkeypatch)
        ft.add_or_touch(TID, q="q", a="a")
        n = ft.delete_thread(TID)
        assert n == 1
        assert ft.count(TID) == 0
        assert not fake.exists("memf:fact:alice|t1:f1")
        assert "alice|t1" not in fake.smembers("memf:user:alice")

    def test_delete_user(self, monkeypatch):
        ft, _ = _make(monkeypatch)
        ft.add_or_touch("alice|tA", q="q", a="a")
        ft.add_or_touch("alice|tB", q="q2", a="a2")
        r = ft.delete_user("alice")
        assert r["facts"] == 2
        assert ft.count("alice|tA") == 0 and ft.count("alice|tB") == 0
