# -*- coding: utf-8 -*-
"""阶段 8.5:验证 simple / medium(react)/ complex(P&E)三条路径都:
  - 通过通用 run_path 把白名单事件落短期流水(persist_event)
  - 流开始前落一条 user_message
  - 结束后触发一次 after_stream 长期升迁(正常与异常都触发)

不调真实 LLM / Qdrant / PG:用桩替换内部图/流,只验证接线。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_reasoning.ReAct.support import runner as runner_mod  # noqa: E402


def _patch_common(monkeypatch, *, persisted, promoted):
    monkeypatch.setattr(runner_mod, "persist_event",
                        lambda ev, **kw: persisted.append(ev))
    monkeypatch.setattr(runner_mod, "after_stream",
                        lambda *a, **k: promoted.append(a))

    # user_message 落库(不写真实存储)
    import memories.storage.short as short_mod
    user_msgs = []
    monkeypatch.setattr(short_mod.short_term, "append_event",
                        lambda tid, etype, payload, **k: user_msgs.append((tid, etype)),
                        raising=False)
    return user_msgs


# ---------------- simple ----------------
def test_simple_path_persists_and_promotes(monkeypatch):
    import types as _types
    from agent_reasoning.ReAct.paths import simple as simple_mod

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

    monkeypatch.setattr(simple_mod, "llm_create_with_retry", _fake_llm)
    import memories.storage.long as long_mod
    monkeypatch.setattr(long_mod, "recall_memories", lambda *a, **k: [])

    persisted, promoted = [], []
    user_msgs = _patch_common(monkeypatch, persisted=persisted, promoted=promoted)

    evs = list(runner_mod.run_simple("你好", history=[],
                                     thread_id="t1", username="alice", session_id="s1"))

    types = [e["type"] for e in evs]
    assert types[-1] == "done"
    assert "assistant_message" in types
    # user_message 在流开始前落库
    assert ("t1", "user_message") in user_msgs
    # 白名单关键事件经 persist_event 落库
    ptypes = [e["type"] for e in persisted]
    assert "assistant_message" in ptypes
    assert "done" in ptypes
    # 升迁恰好一次
    assert promoted == [("t1", "alice", "s1")]


# ---------------- medium (react) ----------------
def test_react_path_persists_and_promotes(monkeypatch):
    persisted, promoted = [], []
    user_msgs = _patch_common(monkeypatch, persisted=persisted, promoted=promoted)

    # 屏蔽 CoverageTracker 守护线程
    monkeypatch.setattr(runner_mod, "CoverageTracker",
                        lambda *a, **k: _FakeTracker())

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
    assert ("t2", "user_message") in user_msgs
    ptypes = [e["type"] for e in persisted]
    assert "assistant_message" in ptypes and "done" in ptypes
    assert promoted == [("t2", "bob", "s2")]


# ---------------- complex (plan_execute) ----------------
def test_plan_execute_path_persists_and_promotes(monkeypatch):
    persisted, promoted = [], []
    user_msgs = _patch_common(monkeypatch, persisted=persisted, promoted=promoted)

    monkeypatch.setattr(runner_mod, "generate_plan",
                        lambda *a, **k: (["查 ALD", "查 CVD"], None))

    def _fake_pe_stream(*a, **k):
        yield {"type": "plan", "steps": ["查 ALD", "查 CVD"]}
        yield {"type": "assistant_message", "content": "综合答案"}
        yield {"type": "done", "trace": {}}

    monkeypatch.setattr(runner_mod, "plan_execute_stream", _fake_pe_stream)

    evs = list(runner_mod.run_plan_execute("对比 ALD 和 CVD", history=[],
                                           thread_id="t3", username="carol",
                                           session_id="s3"))

    types = [e["type"] for e in evs]
    assert "plan" in types
    assert types[-1] == "done"
    assert ("t3", "user_message") in user_msgs
    ptypes = [e["type"] for e in persisted]
    assert "assistant_message" in ptypes
    assert "done" in ptypes
    assert promoted == [("t3", "carol", "s3")]


# ---------------- 异常路径也升迁 ----------------
def test_all_paths_promote_even_on_exception(monkeypatch):
    for path_name, runner_fn in [
        ("react", runner_mod.run_agent_graph),
        ("plan_execute", runner_mod.run_plan_execute),
    ]:
        persisted, promoted = [], []
        _patch_common(monkeypatch, persisted=persisted, promoted=promoted)

        if path_name == "react":
            monkeypatch.setattr(runner_mod, "CoverageTracker",
                                lambda *a, **k: _FakeTracker())

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
        else:
            monkeypatch.setattr(runner_mod, "generate_plan",
                                lambda *a, **k: (["s1"], None))

            def _boom_pe(*a, **k):
                yield {"type": "plan", "steps": ["s1"]}
                raise RuntimeError("pe boom")

            monkeypatch.setattr(runner_mod, "plan_execute_stream", _boom_pe)

        evs = list(runner_fn("q", history=[], thread_id="tX", username="u"))
        # run_path 在内部流异常时补发 error 事件(done 由 HTTP 层补),不向上抛
        assert any(e["type"] == "error" for e in evs), path_name
        assert all(e["type"] != "done" for e in evs), path_name
        # 即使异常也触发升迁一次
        assert len(promoted) == 1, (path_name, promoted)


# ---------------- helpers ----------------
class _FakeTracker:
    def start(self):
        return self

    def close(self):
        pass


class _CtxManager:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False
