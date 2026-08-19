# -*- coding: utf-8 -*-
"""tier 开关测试(1.3):主图前置节点可按 tier 跳过 recall/rewrite,react 可不绑 tools。

这些开关是 simple 路径(阶段 2)的基础,本测试先锁定其在主图上的行为。
"""
import os
import sys
import time
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_reasoning.ReAct.core import nodes  # noqa: E402
from agent_reasoning.ReAct.core.graph import build_graph  # noqa: E402
from agent_reasoning.ReAct.trace import TraceRecorder  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402


class _Delta:
    def __init__(self, content=None, tool_calls=None):
        self.content = content; self.tool_calls = tool_calls
class _Choice:
    def __init__(self, delta, finish_reason=None):
        self.delta = delta; self.finish_reason = finish_reason
class _Chunk:
    def __init__(self, choice=None, usage=None):
        self.choices = [choice] if choice is not None else []; self.usage = usage
def _usage(t=30):
    return types.SimpleNamespace(prompt_tokens=10, completion_tokens=20, total_tokens=t)
def _answer_stream(text="你好"):
    for ch in text:
        yield _Chunk(_Choice(_Delta(content=ch)))
    yield _Chunk(_Choice(_Delta(), "stop"), _usage())
def _make_client(script):
    it = iter(script)
    class C:
        @staticmethod
        def create(**k): return next(it)
    class Chat: completions = C
    return types.SimpleNamespace(chat=Chat())


def _run(monkeypatch, inputs, client_script, *, recall_fn=None, rewrite_fn=None):
    fake = _make_client(client_script)
    monkeypatch.setattr(nodes, "get_client", lambda: fake)
    monkeypatch.setattr(nodes, "dispatch", lambda n, a: [])
    monkeypatch.setattr(nodes, "recall_memories",
                        recall_fn or (lambda *a, **k: []))
    rewrite_called = {"n": 0}
    real_rewrite = rewrite_fn or (lambda m, h=None: [m])

    def counting_rewrite(m, h=None):
        rewrite_called["n"] += 1
        return real_rewrite(m, h)

    monkeypatch.setattr(nodes, "rewrite_query", counting_rewrite)

    recorder = TraceRecorder("tr", time.time(), inputs["question"])
    cfg = {"configurable": {"thread_id": "t1", "user_id": "alice",
                            "trace_recorder": recorder}}
    g = build_graph(checkpointer=InMemorySaver())
    base = {"question": inputs["question"], "history": [],
            "started_at": time.time(), "max_steps": 6, "max_total_seconds": 60}
    base.update(inputs)
    evs = list(g.stream(base, config=cfg, stream_mode="custom"))
    st = g.get_state(cfg).values
    return evs, st, rewrite_called


def test_skip_rewrite_avoids_llm_rewrite(monkeypatch):
    # skip_rewrite=True:rewrite_query 一次都不应被调用,sub_queries 直接用原问题。
    # 无检索来源会触发一次反思重生成,因此准备两个回答流。
    evs, st, called = _run(
        monkeypatch, {"question": "你好", "skip_rewrite": True},
        [_answer_stream("你好"), _answer_stream("你好")],
    )
    assert called["n"] == 0
    assert st["sub_queries"] == ["你好"]
    assert st["final_reason"] == "answer"


def test_skip_recall_avoids_long_term_recall(monkeypatch):
    calls = []

    def spy_recall(user_id, query, **k):
        calls.append((user_id, query))
        return []

    _run(monkeypatch, {"question": "你好", "skip_recall": True},
         [_answer_stream("你好")], recall_fn=spy_recall)
    assert calls == []  # 召回被跳过


def test_default_flags_keep_rewrite_and_recall(monkeypatch):
    # 不传开关(默认 medium 行为):rewrite 与 recall 仍照常调用
    rewrite_called = {"n": 0}
    recall_calls = []

    def real_rewrite(m, h=None):
        rewrite_called["n"] += 1
        return [m]

    def spy_recall(user_id, query, **k):
        recall_calls.append((user_id, query))
        return []

    _run(monkeypatch, {"question": "你好"},
         [_answer_stream("你好"), _answer_stream("你好")],
         recall_fn=spy_recall, rewrite_fn=real_rewrite)
    assert rewrite_called["n"] == 1
    assert recall_calls == [("alice", "你好")]
