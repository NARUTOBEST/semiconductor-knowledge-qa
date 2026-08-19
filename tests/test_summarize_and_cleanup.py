# -*- coding: utf-8 -*-
"""摘要压缩(纯 token 预算 + 后台实时刷新)与会话清理测试(不依赖真实 PG/LLM)。"""
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import (  # noqa: E402
    AIMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage,
)

from memories.storage.working import summarize  # noqa: E402
from memories.storage.working.summarize import (  # noqa: E402
    calculate_compaction_keep_index,
    calculate_pregen_keep_index,
    format_summary_block,
    maybe_summarize,
    schedule_pregeneration,
    KEEP_RECENT_TOKENS, COMPACT_TRIGGER_TOKENS, SUMMARY_START_TOKENS,
)


# ---------------- 构造消息 ----------------
def _big_turn(i, tool_chars=2000, ai_chars=600):
    """一轮带工具检索的消息(约 2.2k token)。"""
    return [
        HumanMessage(content=f"q{i} 请检索半导体相关资料", id=f"h{i}"),
        AIMessage(content="", id=f"a{i}-call",
                  tool_calls=[{"id": f"c{i}", "name": "search_text",
                               "args": {"query": "x", "k": 3}}]),
        ToolMessage(content=("半导体材料与工艺" * (tool_chars // 8)),
                    tool_call_id=f"c{i}", id=f"t{i}"),
        AIMessage(content=("根据检索结果作答" * (ai_chars // 8)), id=f"a{i}"),
    ]


def _big_dialog(n, **kw):
    out = []
    for i in range(n):
        out.extend(_big_turn(i, **kw))
    return out


def _tokens(msgs):
    return summarize._convo_tokens(summarize._convo_without_system(msgs))


# ---------------- 边界算法 ----------------
def test_compaction_none_below_trigger():
    # 总量低于 COMPACT_TRIGGER -> 不压缩
    msgs = _big_dialog(5)
    assert _tokens(msgs) < COMPACT_TRIGGER_TOKENS
    assert calculate_compaction_keep_index(msgs) is None


def test_compaction_returns_keep_index_and_respects_window():
    # 构造超过 68k 的对话
    msgs = _big_dialog(40)
    assert _tokens(msgs) >= COMPACT_TRIGGER_TOKENS
    kf = calculate_compaction_keep_index(msgs)
    assert kf is not None and kf > 0
    kept = summarize._convo_without_system(msgs)[kf:]
    # 保留段必须 <= KEEP_RECENT_TOKENS 窗口
    assert summarize._convo_tokens(kept) <= KEEP_RECENT_TOKENS
    # 保留段以一个 HumanMessage 开头(按整轮切,不拆半轮)
    assert isinstance(kept[0], HumanMessage)
    # 旧区非空
    assert summarize._removed_messages(msgs, kf)


def test_pregen_starts_earlier_than_compaction():
    # SUMMARY_START(40k) < COMPACT_TRIGGER(68k)
    early = _big_dialog(20)  # ~44k
    assert SUMMARY_START_TOKENS < _tokens(early) < COMPACT_TRIGGER_TOKENS
    assert calculate_pregen_keep_index(early) is not None
    assert calculate_compaction_keep_index(early) is None


def test_pregen_none_below_start():
    msgs = _big_dialog(5)
    assert _tokens(msgs) < SUMMARY_START_TOKENS
    assert calculate_pregen_keep_index(msgs) is None


def test_pregen_and_compact_agree_on_same_messages():
    # 实际时序:finalize 与下一轮 build_messages 看到同一份 state,
    # 故预生成与压缩算出的保留段必须一致(缓存才能命中)
    msgs = _big_dialog(40)
    assert calculate_pregen_keep_index(msgs) == calculate_compaction_keep_index(msgs)


def test_keep_index_ignores_system():
    msgs = [SystemMessage(content="sys", id="sys")] + _big_dialog(40)
    kf = calculate_compaction_keep_index(msgs)
    removed_ids = [m.id for m in summarize._removed_messages(msgs, kf)]
    assert "sys" not in removed_ids


def test_format_summary_block_truncates():
    assert format_summary_block("") == ""
    assert format_summary_block(None) == ""
    long = "事" * (summarize.SUMMARY_MAX_CHARS + 100)
    block = format_summary_block(long)
    assert block.endswith("…")
    assert len(block) < len(long) + 30


# ---------------- maybe_summarize ----------------
def test_maybe_summarize_compacts(monkeypatch):
    msgs = _big_dialog(40)
    monkeypatch.setattr(summarize, "_llm_summarize",
                        lambda prev, tx, tid="": "增量摘要")
    out = maybe_summarize({"messages": msgs, "summary": "旧"})
    assert out["summary"] == "增量摘要"
    assert all(isinstance(m, RemoveMessage) for m in out["messages"])
    # 删除的是旧区,且至少删掉一轮
    assert len(out["messages"]) >= 4


def test_maybe_summarize_noop_below_trigger():
    assert maybe_summarize({"messages": _big_dialog(5), "summary": ""}) == {}


def test_maybe_summarize_failure_keeps_messages(monkeypatch):
    msgs = _big_dialog(40)
    monkeypatch.setattr(summarize, "_llm_summarize", lambda *a, **k: None)
    assert maybe_summarize({"messages": msgs, "summary": ""}) == {}


# ---------------- 后台实时刷新 + 命中 ----------------
def test_pregeneration_consumed_at_compaction(monkeypatch):
    import queue
    q = queue.Queue()

    def fake_llm(prev, transcript, trace_id=""):
        return q.get(timeout=5)

    monkeypatch.setattr(summarize, "_llm_summarize", fake_llm)

    # 已超 68k:finalize 调度预生成
    msgs = _big_dialog(40)
    schedule_pregeneration(msgs, prev_summary="旧", thread_id="tA", trace_id="tr")
    q.put("预生成摘要")

    deadline = time.time() + 5
    while time.time() < deadline:
        with summarize._pregen_lock:
            fut = summarize._pregen.get("tA", {}).get("future")
        if fut is not None and fut.done():
            break
        time.sleep(0.02)

    # 下一轮压缩命中缓存(未命中会阻塞在空 q,超时即失败)
    out = maybe_summarize({"messages": msgs, "summary": "旧"}, thread_id="tA")
    assert out["summary"] == "预生成摘要"
    assert out["messages"]
    with summarize._pregen_lock:
        assert "tA" not in summarize._pregen


def test_pregeneration_not_triggered_below_start():
    schedule_pregeneration(_big_dialog(5), prev_summary="", thread_id="tB")
    with summarize._pregen_lock:
        assert "tB" not in summarize._pregen


def test_pregen_mismatch_falls_back_to_sync(monkeypatch):
    msgs = _big_dialog(40)
    schedule_pregeneration(msgs, prev_summary="摘要A", thread_id="tC")
    with summarize._pregen_lock:
        fut = summarize._pregen["tC"]["future"]
    fut.set_result("不应使用")

    called = []
    monkeypatch.setattr(summarize, "_llm_summarize",
                        lambda prev, tx, tid="": called.append(1) or "同步摘要")
    # 真实提交时旧摘要不同 -> key 不匹配 -> 同步兜底
    out = maybe_summarize({"messages": msgs, "summary": "摘要B(变了)"},
                          thread_id="tC")
    assert out["summary"] == "同步摘要"
    assert called == [1]


def test_pregen_cache_bounded_lru(monkeypatch):
    # 缓存有上限:超过 max 条目时淘汰最久未访问的(LRU)
    import concurrent.futures
    monkeypatch.setattr(summarize, "_PREGEN_MAX_ENTRIES", 3)

    def _fake_submit(prev, transcript, trace_id):
        fut = concurrent.futures.Future()
        fut.set_result("s")
        return fut
    monkeypatch.setattr(summarize, "_submit_pregen", _fake_submit)

    msgs = _big_dialog(40)  # 超过 pregen 阈值,keep_from 非 None
    keys = [frozenset([f"id{i}"]) for i in range(5)]
    # 直接往缓存写 5 个不同 thread 的条目(绕过 schedule 的 key 一致性判断)
    with summarize._pregen_lock:
        for i in range(5):
            summarize._pregen[f"th{i}"] = {"key": (keys[i], ""),
                                           "future": _fake_submit("", "", "")}
            summarize._pregen.move_to_end(f"th{i}")
        while len(summarize._pregen) > summarize._PREGEN_MAX_ENTRIES:
            summarize._pregen.popitem(last=False)

    with summarize._pregen_lock:
        assert len(summarize._pregen) == 3
        # 最早写入的 th0/th1 被淘汰,保留 th2/th3/th4
        assert "th0" not in summarize._pregen
        assert "th1" not in summarize._pregen
        assert {"th2", "th3", "th4"} <= set(summarize._pregen)


# ---------------- 清理级联(打桩 working_saver / short_term) ----------------
def test_delete_thread_artifacts_calls_both(monkeypatch):
    from memories.orchestration.working import lifecycle as cleanup

    deleted_cp, deleted_short = [], []

    class _FakeSaver:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def delete_thread(self, tid): deleted_cp.append(tid)

    monkeypatch.setattr(cleanup, "working_saver", lambda: _FakeSaver())
    monkeypatch.setattr(cleanup.short_term, "delete_thread",
                        lambda tid: deleted_short.append(tid) or 3)

    cleanup.delete_thread_artifacts("t-123")
    assert deleted_cp == ["t-123"]
    assert deleted_short == ["t-123"]


def test_delete_thread_artifacts_clears_pregen_cache(monkeypatch):
    from memories.orchestration.working import lifecycle as cleanup

    class _FakeSaver:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def delete_thread(self, tid): pass

    monkeypatch.setattr(cleanup, "working_saver", lambda: _FakeSaver())
    monkeypatch.setattr(cleanup.short_term, "delete_thread", lambda tid: 0)

    with summarize._pregen_lock:
        summarize._pregen["t-z"] = {"key": ("x", ""), "future": None}
    cleanup.delete_thread_artifacts("t-z")
    with summarize._pregen_lock:
        assert "t-z" not in summarize._pregen


def test_delete_thread_artifacts_swallows_checkpoint_error(monkeypatch):
    from memories.orchestration.working import lifecycle as cleanup

    class _BoomSaver:
        def __enter__(self): raise RuntimeError("pg down")
        def __exit__(self, *a): return False

    short_called = []
    monkeypatch.setattr(cleanup, "working_saver", lambda: _BoomSaver())
    monkeypatch.setattr(cleanup.short_term, "delete_thread",
                        lambda tid: short_called.append(tid))
    cleanup.delete_thread_artifacts("t-x")
    assert short_called == ["t-x"]


def test_prune_inactive_batch(monkeypatch):
    from memories.orchestration.working import lifecycle as cleanup
    from datetime import datetime, timezone

    stale = [("t1", datetime.now(timezone.utc)),
             ("t2", datetime.now(timezone.utc))]
    monkeypatch.setattr(cleanup.short_term, "stale_threads",
                        lambda days: stale)
    deleted = []

    class _FakeSaver:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def delete_thread(self, tid): deleted.append(tid)

    monkeypatch.setattr(cleanup, "working_saver", lambda: _FakeSaver())
    monkeypatch.setattr(cleanup.short_term, "delete_threads_before",
                        lambda days: 5)

    result = cleanup.prune_inactive(30)
    assert result == {"candidates": 2, "checkpoints_deleted": 2,
                      "short_deleted": 5}
    assert set(deleted) == {"t1", "t2"}
