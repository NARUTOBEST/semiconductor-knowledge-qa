# -*- coding: utf-8 -*-
"""Req1/D1:预取游标互斥测试。

摘要落盘时记录短期流水高水位 recent_cursor_seq;之后预取的近期窗口只读 seq > 游标
的事件——游标之前的对话已被摘要覆盖,严格互斥、不重叠:
  - 存储层 recent_dialogue(since_seq=...) 只回游标之后;
  - recent_high_watermark 返回当前高水位;
  - build_prefetch_block 经 SessionFile 读游标,近期块只含游标后内容(游标前不出现)。
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

TID = "alice|tcur"


class _CursorFile:
    """游标文件打桩:模拟摘要已落盘、高水位为 2。"""
    def __init__(self, username, thread_id):
        pass

    def read_meta_typed(self):
        return {"recent_cursor_seq": 2, "pending_remove_ids": []}


def _seed(monkeypatch):
    fake = fakeredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(st_module, "get_redis", lambda: fake)
    monkeypatch.setattr(recall_mod, "redis_ready_fast", lambda: True)
    m = ShortTermMemory()
    # seq1/2 = 游标前(已进摘要);seq3/4 = 游标后;seq5 = 本轮当前问题
    m.append_event(TID, "user_message", {"content": "旧问题A"})
    m.append_event(TID, "assistant_message", {"content": "旧回答A"})
    m.append_event(TID, "user_message", {"content": "新问题B"})
    m.append_event(TID, "assistant_message", {"content": "新回答B"})
    m.append_event(TID, "user_message", {"content": "现在的问题"})
    return m


def test_storage_recent_dialogue_respects_cursor(monkeypatch):
    m = _seed(monkeypatch)
    all_rows = m.recent_dialogue(TID, since_seq=0)
    post = m.recent_dialogue(TID, since_seq=2)
    assert [r["content"] for r in all_rows] == [
        "旧问题A", "旧回答A", "新问题B", "新回答B", "现在的问题"]
    # 游标后:严格 seq > 2,不含旧问题/旧回答
    assert [r["content"] for r in post] == ["新问题B", "新回答B", "现在的问题"]


def test_high_watermark_returns_current_seq(monkeypatch):
    _seed(monkeypatch)
    assert recall_mod.recent_high_watermark(TID) == 5


def test_prefetch_block_recent_excludes_pre_cursor(monkeypatch):
    _seed(monkeypatch)
    monkeypatch.setattr(C, "RECALL_PREFETCH_ENABLED", True, raising=False)
    monkeypatch.setattr(pf, "SessionFile", _CursorFile)
    monkeypatch.setattr(pf, "recall_memories",
                        lambda u, q: ([], None))   # 长期为空,只验近期块

    out = pf.build_prefetch_block("alice", TID, "现在的问题")
    block = out["block"]
    assert "新回答B" in block          # 游标后内容进近期块
    assert "旧回答A" not in block      # 游标前内容已进摘要,不重复注入
    assert "旧问题A" not in block
    assert "现在的问题" not in block   # 当前问题已在 HumanMessage,剔除
    assert out["degraded"] == []
