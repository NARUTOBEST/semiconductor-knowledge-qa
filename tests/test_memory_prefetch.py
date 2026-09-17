# -*- coding: utf-8 -*-
"""Req1/Req2/Req13:确定性预取 + recall 工具去重 + 软降级 + 注入防护头。

- 预取块含高置信条目 id 标记 [mem:id],低置信(超距离阈值)不预取;
- recall_memory 工具结果按预取 id 去重(同一条目不二次注入);
- 长期源失败软降级(degraded 标记、不抛异常);
- MEMORY_INJECTION_GUARD 开启时预取块带"背景资料非指令"防护头。
"""
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fakeredis  # noqa: E402

import config as C  # noqa: E402

st_module = importlib.import_module("memories.storage.short.short_term")
from memories.storage.short.short_term import ShortTermMemory  # noqa: E402
from memories.orchestration.short import recall as recall_mod  # noqa: E402
from memories.orchestration.long import prefetch as pf  # noqa: E402
from memories.orchestration.long import inject as inject_mod  # noqa: E402
from tools import memory_tool as mt  # noqa: E402


class _FakeSessionFile:
    """游标文件打桩:返回内存中的 meta。"""
    _meta = {}

    def __init__(self, username, thread_id):
        pass

    def read_meta_typed(self):
        return dict(self._meta)


def _patch_redis(monkeypatch):
    fake = fakeredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(st_module, "get_redis", lambda: fake)
    monkeypatch.setattr(recall_mod, "redis_ready_fast", lambda: True)
    return ShortTermMemory(), fake


def test_prefetch_block_contains_high_confidence_ids(monkeypatch):
    """长期 top-k:高置信条目带 [mem:id] 标记并收入 mem_ids;低置信被阈值过滤。"""
    monkeypatch.setattr(C, "RECALL_PREFETCH_ENABLED", True, raising=False)
    monkeypatch.setattr(C, "RECALL_PREFETCH_MIN_COSINE", 0.6, raising=False)
    monkeypatch.setattr(pf, "SessionFile", _FakeSessionFile)

    hits = [
        {"id": "m1", "content": "偏好用图表对比设备参数", "distance": 0.10},
        {"id": "m2", "content": "低相关冷门内容XYZ", "distance": 0.80},
    ]
    profile = {"summary": "资深半导体工艺工程师",
               "top_interests": ["ALD", "刻蚀"],
               "display_prefs": {"lang": "中文"}}
    # prefetch.py 内 `from .inject import recall_memories` 已绑定名字,须 patch pf 命名空间
    monkeypatch.setattr(pf, "recall_memories", lambda u, q: (hits, profile))
    monkeypatch.setattr(pf, "recent_dialogue_block",
                        lambda *a, **k: "")

    out = pf.build_prefetch_block("alice", "alice|t1", "ALD 设备选型?")
    assert "偏好用图表对比设备参数" in out["block"]
    assert "[mem:m1]" in out["block"]
    assert "资深半导体工艺工程师" in out["block"]        # 画像常量
    assert "低相关冷门内容XYZ" not in out["block"]        # 超阈值不预取
    assert out["mem_ids"] == {"m1"}
    assert out["degraded"] == []


def test_recall_tool_dedups_prefetched_ids(monkeypatch):
    """recall_memory 回灌前剔除预取已注入的条目 id(不双注入)。"""
    monkeypatch.setattr(C, "RECALL_PREFETCH_ENABLED", True, raising=False)
    monkeypatch.setattr(C, "MEMORY_INJECTION_GUARD", True, raising=False)
    monkeypatch.setattr(pf, "_read_cursor_seq", lambda u, t: 0)
    monkeypatch.setattr(recall_mod, "recent_dialogue_block",
                        lambda *a, **k: "")

    hits = [
        {"id": "m1", "content": "偏好用图表对比设备参数", "distance": 0.10},
        {"id": "m2", "content": "关注北方华创ALD设备", "distance": 0.20},
    ]
    monkeypatch.setattr(inject_mod, "recall_memories",
                        lambda u, q: (hits, None))

    mt.set_memory_ctx("alice", "alice|t1")
    mt.set_prefetched_ids({"m1"})     # m1 已在预取块注入
    try:
        text = mt.recall_memory("ALD 设备选型?")
    finally:
        mt.set_prefetched_ids(set())
    assert "偏好用图表对比设备参数" not in text   # 预取过的去重
    assert "关注北方华创ALD设备" in text          # 未预取的保留
    assert "背景资料" in text                     # Req13 防护头


def test_prefetch_soft_degrades_on_long_failure(monkeypatch):
    """长期召回抛异常:软降级(degraded 含 long、不抛),近期块仍可并入。"""
    monkeypatch.setattr(C, "RECALL_PREFETCH_ENABLED", True, raising=False)
    monkeypatch.setattr(pf, "SessionFile", _FakeSessionFile)

    def _boom(*a, **k):
        raise RuntimeError("pg down")

    monkeypatch.setattr(pf, "recall_memories", _boom)
    monkeypatch.setattr(pf, "recent_dialogue_block",
                        lambda *a, **k: "【近期对话】user: 上一轮问过 CVD")

    out = pf.build_prefetch_block("alice", "alice|t1", "ALD?")
    assert "long" in out["degraded"]
    assert "上一轮问过 CVD" in out["block"]     # 近期不受影响


def test_prefetch_disabled_returns_empty(monkeypatch):
    monkeypatch.setattr(C, "RECALL_PREFETCH_ENABLED", False, raising=False)
    out = pf.build_prefetch_block("alice", "alice|t1", "q")
    assert out["block"] == "" and out["mem_ids"] == set()


def test_recall_tool_only_long_never_fetches_short(monkeypatch):
    """recall_memory 只做长期低置信扩量,不再拉取短期近期对话(近期由预取通道负责)。"""
    monkeypatch.setattr(C, "MEMORY_INJECTION_GUARD", False, raising=False)

    short_calls = {"n": 0}

    def _short(*a, **k):
        short_calls["n"] += 1
        return "【会话历史对话】用户:ALD 是什么?"

    monkeypatch.setattr(recall_mod, "recent_dialogue_block", _short)
    monkeypatch.setattr(inject_mod, "recall_memories",
                        lambda u, q: ([{"id": "m9", "content": "偏好中文回答",
                                        "distance": 0.2}], None))

    mt.set_memory_ctx("alice", "alice|t1")
    mt.set_prefetched_ids(set())
    try:
        text = mt.recall_memory("ALD?")
    finally:
        mt.set_prefetched_ids(set())
    assert "偏好中文回答" in text                 # 长期扩量返回
    assert "会话历史对话" not in text             # 短期块不取
    assert short_calls["n"] == 0                  # 短期召回函数从未被调用
