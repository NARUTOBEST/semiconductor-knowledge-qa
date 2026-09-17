# -*- coding: utf-8 -*-
"""验证 simple / react 两条路径都:
  - 通过通用 run_path 把白名单事件落短期流水(persist_event)
  - 流开始前落一条 user_message
  - 内部流异常时补发 error 事件、不向上抛

不调真实 LLM / Qdrant / PG:用桩替换内部图/流,只验证接线。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_reasoning.ReAct.support import runner as runner_mod  # noqa: E402
import agent_reasoning.simple.runner as simple_runner  # noqa: E402
import agent_reasoning.simple.stream as simple_stream  # noqa: E402


def _patch_common(monkeypatch, *, persisted):
    # persist_event 在 runner_mod 中被 run_path 以模块全局名调用,
    # 两条路径都复用同一个 run_path,patch runner_mod 即可全部命中。
    monkeypatch.setattr(runner_mod, "persist_event",
                        lambda ev, **kw: persisted.append(ev))

    # user_message 落库(不写真实存储)
    import memories.storage.short as short_mod
    user_msgs = []
    monkeypatch.setattr(short_mod.short_term, "append_event",
                        lambda tid, etype, payload, **k: user_msgs.append((tid, etype)),
                        raising=False)
    return user_msgs


# ---------------- simple ----------------
def test_simple_path_persists(monkeypatch):
    import types as _types

    class _Delta:
        def __init__(self, content):
            self.content = content

    class _Choice:
        def __init__(self, content, finish=None):
            self.delta = _Delta(content)
            self.finish_reason = finish

    class _Chunk:
        def __init__(self, content, finish=None, usage=False):
            self.choices = [_Choice(content, finish)]
            self.usage = _types.SimpleNamespace(prompt_tokens=1, completion_tokens=1,
                                                total_tokens=2) if usage else None

    def _fake_llm(c, trace_id="", retries=1, **k):
        def _g():
            for ch in "你好":
                yield _Chunk(ch)
            yield _Chunk(None, finish="stop", usage=True)
        return _g(), None

    monkeypatch.setattr(simple_stream, "llm_create_with_retry", _fake_llm)

    persisted = []
    user_msgs = _patch_common(monkeypatch, persisted=persisted)

    evs = list(simple_runner.run_simple("你好", history=[],
                                        thread_id="t1", username="alice", session_id="s1"))

    types = [e["type"] for e in evs]
    assert types[-1] == "done"
    assert "assistant_message" in types
    # user_message 在流开始前落库(键按用户隔离: "用户名|会话id")
    assert ("alice|t1", "user_message") in user_msgs
    # 白名单关键事件经 persist_event 落库
    ptypes = [e["type"] for e in persisted]
    assert "assistant_message" in ptypes
    assert "done" in ptypes


# ---------------- react ----------------
def test_react_path_persists(monkeypatch):
    persisted = []
    user_msgs = _patch_common(monkeypatch, persisted=persisted)

    class _FakeGraph:
        def stream(self, *a, **k):
            yield {"type": "status", "message": "理解问题中…"}
            yield {"type": "assistant_message", "content": "答案"}
            yield {"type": "done", "trace": {}}

        def get_state(self, *a, **k):
            class _S:
                values = {}
            return _S()

    monkeypatch.setattr(runner_mod, "build_graph", lambda *a, **k: _FakeGraph())
    monkeypatch.setattr(runner_mod, "working_saver",
                        lambda: _CtxManager())

    evs = list(runner_mod.run_agent_graph("ALD 原理", history=[],
                                          thread_id="t2", username="bob",
                                          session_id="s2"))

    types = [e["type"] for e in evs]
    assert types[-1] == "done"
    assert ("bob|t2", "user_message") in user_msgs
    ptypes = [e["type"] for e in persisted]
    assert "assistant_message" in ptypes and "done" in ptypes


# ---------------- 异常路径补发 error ----------------
def test_react_path_error_event_on_exception(monkeypatch):
    persisted = []
    _patch_common(monkeypatch, persisted=persisted)

    class _BoomGraph:
        def stream(self, *a, **k):
            yield {"type": "status"}
            raise RuntimeError("graph boom")

        def get_state(self, *a, **k):
            class _S:
                values = {}
            return _S()

    monkeypatch.setattr(runner_mod, "build_graph",
                        lambda *a, **k: _BoomGraph())
    monkeypatch.setattr(runner_mod, "working_saver",
                        lambda: _CtxManager())

    evs = list(runner_mod.run_agent_graph("q", history=[],
                                          thread_id="tX", username="u"))
    # run_path 在内部流异常时补发 error 事件(done 由 HTTP 层补),不向上抛
    assert any(e["type"] == "error" for e in evs)
    assert all(e["type"] != "done" for e in evs)


# ---------------- helpers ----------------
class _CtxManager:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False
