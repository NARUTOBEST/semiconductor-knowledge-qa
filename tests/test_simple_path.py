# -*- coding: utf-8 -*-
"""simple 路径测试(阶段 2)。

mock 掉 LLM 流与长期记忆召回,验证:
  - 一次 LLM 调用,不绑定工具(不传 tools/tool_choice)
  - 模型用 config.TIER_MODEL_SIMPLE
  - 事件序列 status -> token* -> assistant_message(run_path 包 done)
  - 不调用 dispatch/检索工具
  - 长期记忆召回被调用(保留 recall)
  - 短期流水落库 user_message / assistant_message / done
"""
import os
import sys
import time
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_reasoning.simple import stream as simple_mod  # noqa: E402
from agent_reasoning.simple.stream import simple_answer_stream  # noqa: E402
from agent_reasoning.simple import runner as simple_runner  # noqa: E402
from agent_reasoning.ReAct.support import runner as runner_mod  # noqa: E402
from agent_reasoning.ReAct.trace import TraceRecorder  # noqa: E402


class _Delta:
    def __init__(self, content=None):
        self.content = content
        self.tool_calls = None
class _Choice:
    def __init__(self, delta, finish_reason=None):
        self.delta = delta
        self.finish_reason = finish_reason
class _Chunk:
    def __init__(self, choice=None, usage=None):
        self.choices = [choice] if choice is not None else []
        self.usage = usage
def _usage(t=20):
    return types.SimpleNamespace(prompt_tokens=5, completion_tokens=15, total_tokens=t)
def _answer_stream(text="你好呀"):
    for ch in text:
        yield _Chunk(_Choice(_Delta(content=ch)))
    yield _Chunk(_Choice(_Delta(), "stop"), _usage())


def test_simple_answer_stream_emits_tokens_and_message(monkeypatch):
    captured = {}

    def fake_create(client, trace_id="", retries=1, **kwargs):
        captured["kwargs"] = kwargs
        return _answer_stream("你好呀"), None

    monkeypatch.setattr(simple_mod, "llm_create_with_retry", fake_create)

    t0 = time.time()
    recorder = TraceRecorder("tr", t0, "你好")
    evs = list(simple_answer_stream(
        "你好", history=[], recalled_memories=[],
        recorder=recorder, trace_id="tr", t0=t0,
    ))
    types = [e["type"] for e in evs]
    assert types[0] == "status"
    assert types.count("token") == 3  # 你好呀
    # 以 meta + done 收尾(与 medium 路径 SSE 契约一致)
    assert evs[-1]["type"] == "done"
    assert evs[-1]["trace"]["final_reason"] == "answer"
    amsg = next(e for e in evs if e["type"] == "assistant_message")
    assert amsg["content"] == "你好呀"
    assert "meta" in types
    # 不绑定工具
    kw = captured["kwargs"]
    assert "tools" not in kw
    assert "tool_choice" not in kw
    # 用 lite 模型
    assert kw["model"] == "doubao-seed-2.0-lite"
    assert recorder.final_reason == "answer"


def test_run_simple_persists_events_and_recalls(monkeypatch):
    # LLM 走 simple_answer_stream 内部的 llm_create_with_retry
    monkeypatch.setattr(simple_mod, "llm_create_with_retry",
                        lambda c, trace_id="", retries=1, **k: (_answer_stream("你好"), None))

    # 召回被调用
    recall_calls = []

    def fake_recall(user_id, query, **k):
        recall_calls.append((user_id, query))
        return []

    import memories.storage.long as long_mod
    monkeypatch.setattr(long_mod, "recall_memories", fake_recall)
    # runner 里是 `from memories.storage.long import recall_memories`(函数内 import),
    # patch 模块属性即可被函数内 import 取到。

    # 短期流水
    persisted = []
    monkeypatch.setattr(runner_mod, "persist_event",
                        lambda ev, **kw: persisted.append(ev["type"]))
    # 升迁
    promoted = []
    monkeypatch.setattr(runner_mod, "after_stream",
                        lambda *a, **k: promoted.append(a))

    # short_term.append_event
    import memories.storage.short as short_mod
    user_msg_seen = []
    monkeypatch.setattr(short_mod.short_term, "append_event",
                        lambda tid, etype, payload, **k: user_msg_seen.append(etype),
                        raising=False)

    evs = list(simple_runner.run_simple(
        "你好", history=[],
        thread_id="t1", username="alice", session_id="s1",
    ))
    types = [e["type"] for e in evs]
    assert types[0] == "status"
    assert "token" in types
    assert types[-1] == "done"
    assert "assistant_message" in types

    # recall 用 username + 原问题调用
    assert recall_calls == [("alice", "你好")]
    # 流开始前落了 user_message
    assert user_msg_seen == ["user_message"]
    # 关键事件落短期流水
    assert "assistant_message" in persisted
    # 结束后触发升迁
    assert promoted == [("t1", "alice", "s1")]


def test_run_simple_llm_error_emits_error_and_still_promotes(monkeypatch):
    monkeypatch.setattr(simple_mod, "llm_create_with_retry",
                        lambda c, trace_id="", retries=1, **k: (None, RuntimeError("boom")))
    import memories.storage.long as long_mod
    monkeypatch.setattr(long_mod, "recall_memories", lambda *a, **k: [])
    monkeypatch.setattr(runner_mod, "persist_event", lambda ev, **k: True)
    promoted = []
    monkeypatch.setattr(runner_mod, "after_stream", lambda *a, **k: promoted.append(1))
    # 屏蔽 user_message 落库
    import memories.storage.short as short_mod
    monkeypatch.setattr(short_mod.short_term, "append_event",
                        lambda *a, **k: None, raising=False)

    evs = list(simple_runner.run_simple("x", thread_id="t", username=None))
    types = [e["type"] for e in evs]
    assert "error" in types
    assert "boom" in next(e["message"] for e in evs if e["type"] == "error")
    assert promoted == [1]  # 异常也升迁
