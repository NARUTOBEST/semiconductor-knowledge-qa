# -*- coding: utf-8 -*-
"""Req6:摘要/compact【先写后删】幂等测试。

- 触发 compact 时:只有会话文件(摘要正文 + 游标 + recent_cursor_seq + pending_remove_ids)
  落盘成功后才返回 RemoveMessage;返回的删除 id 与文件里 pending_remove_ids 完全一致;
- 重跑同轮(文件已记录 pending,消息尚未被 checkpoint 删除):不重复生成 RemoveMessage;
- 落盘失败:绝不返回 RemoveMessage(route=degrade / compacted=False)。
"""
import contextlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import (  # noqa: E402
    AIMessage, HumanMessage, RemoveMessage, ToolMessage,
)

from memories.orchestration.memory_loop import session_summary as SS  # noqa: E402


def _big_turn(i, tool_chars=2000, ai_chars=600):
    return [
        HumanMessage(content=f"q{i} 请检索半导体相关资料", id=f"h{i}"),
        AIMessage(content="", id=f"a{i}-call",
                  tool_calls=[{"id": f"c{i}", "name": "search_text",
                               "args": {"query": "x", "k": 3}}]),
        ToolMessage(content=("半导体材料与工艺" * (tool_chars // 8)),
                    tool_call_id=f"c{i}", id=f"t{i}"),
        AIMessage(content=("根据检索结果作答" * (ai_chars // 8)), id=f"a{i}"),
    ]


def _big_dialog(n):
    out = []
    for i in range(n):
        out.extend(_big_turn(i))
    return out


class _FakeSF:
    """内存版 SessionFile;fail_write=True 时写盘抛错模拟磁盘失败。"""
    fail_write = False

    def __init__(self, username, thread_id):
        self.key = (username, thread_id)

    def _st(self):
        if not hasattr(_FakeSF, "_store"):
            _FakeSF._store = {}
        return _FakeSF._store.setdefault(
            self.key, {"exists": False, "meta": {}, "body": "", "writes": 0})

    def exists(self):
        return self._st()["exists"]

    def read(self):
        st = self._st()
        return dict(st["meta"]), st["body"]

    def read_meta_typed(self):
        return dict(self._st()["meta"])

    def write(self, meta, body):
        if _FakeSF.fail_write:
            raise RuntimeError("disk full")
        st = self._st()
        st["meta"] = dict(meta)
        st["body"] = body
        st["exists"] = True
        st["writes"] += 1

    def lock(self, timeout=None):
        """与真实现同签名(整周期锁);测试内无并发,直接放行。"""
        return contextlib.nullcontext()


def _setup(monkeypatch):
    _FakeSF._store = {}
    _FakeSF.fail_write = False
    monkeypatch.setattr(SS, "SessionFile", _FakeSF)
    # 不调真实 LLM:子代理级摘要直接返回(level=subagent,允许 compact)
    monkeypatch.setattr(SS, "_llm_subagent",
                        lambda transcript, prev, deadline=None: "会话摘要:讨论了 ALD/CVD 工艺选型。")
    monkeypatch.setattr(SS, "recent_high_watermark", lambda tid: 42)


def _run(username="alice", thread_id="alice|t1", n=16):
    return SS.run_session_maintenance(username, thread_id, _big_dialog(n))


def test_remove_only_after_file_persisted(monkeypatch):
    _setup(monkeypatch)
    out = _run()
    assert out["compacted"] is True
    remove = out["remove"]
    assert remove and all(isinstance(m, RemoveMessage) for m in remove)

    # 文件确实先落盘,且记录了游标 / 流水高水位 / 待删 id 列表
    st = _FakeSF._store[("alice", "alice|t1")]
    assert st["writes"] >= 1
    meta = st["meta"]
    assert meta.get("recent_cursor_seq") == 42
    pending = meta.get("pending_remove_ids") or []
    # 返回的 RemoveMessage id 与落盘的 pending_remove_ids 完全一致(先写后删)
    assert sorted(m.id for m in remove) == sorted(pending)
    assert st["body"]  # 摘要正文已落盘


def test_replay_same_turn_is_idempotent(monkeypatch):
    _setup(monkeypatch)
    first = _run()
    assert first["compacted"] is True
    removed_first = {m.id for m in first["remove"]}

    # 模拟重放:RemoveMessage 尚未应用(checkpoint 前崩溃),同一份消息再跑一次
    second = _run()
    # 已落盘 pending 的 id 不重复删除;无新增 RemoveMessage
    assert second["remove"] == []
    assert second["compacted"] is False
    # 没有新 id 被误删(幂等:两轮删除集合一致)
    assert {m.id for m in second["remove"]} <= removed_first


def test_write_failure_emits_no_remove(monkeypatch):
    _setup(monkeypatch)
    _FakeSF.fail_write = True
    out = _run()
    # 落盘失败:绝不返回 RemoveMessage,降级
    assert out["remove"] == []
    assert out["compacted"] is False
    st = _FakeSF._store[("alice", "alice|t1")]
    assert st["exists"] is False and st["writes"] == 0
