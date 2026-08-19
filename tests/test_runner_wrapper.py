# -*- coding: utf-8 -*-
"""通用 run_path 包装器测试(1.4)。

验证它能包裹任意事件流,并统一处理:
  - 事件透传 + persist_event 落短期流水 + on_event 回调
  - 正常结束后触发 after_stream(长期升迁)且只触发一次
  - 流抛异常时补发 error 事件,仍触发 after_stream,且不向上抛
  - on_error / on_finally 回调被调用
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_reasoning.ReAct.support import runner as runner_mod  # noqa: E402


def _collect(gen):
    return list(gen)


def test_run_path_persists_and_yields_events(monkeypatch):
    persisted = []
    on_events = []
    promoted = []

    monkeypatch.setattr(runner_mod, "persist_event",
                        lambda ev, **kw: persisted.append((ev, kw)))
    monkeypatch.setattr(runner_mod, "after_stream",
                        lambda *a, **k: promoted.append(a))

    def fake_stream():
        yield {"type": "status", "msg": "a"}
        yield {"type": "token", "delta": "x"}
        yield {"type": "done"}

    evs = _collect(runner_mod.run_path(
        "simple", fake_stream(),
        thread_id="t1", username="alice", session_id="s1", trace_id="tr",
        on_event=lambda ev: on_events.append(ev),
    ))

    assert [e["type"] for e in evs] == ["status", "token", "done"]
    # 每个事件都落了短期流水,主键透传
    assert len(persisted) == 3
    assert persisted[0][1] == {"thread_id": "t1", "user_id": "alice",
                               "session_id": "s1"}
    assert on_events == evs
    # 结束后触发一次长期升迁
    assert promoted == [("t1", "alice", "s1")]


def test_run_path_emits_error_on_exception_and_still_promotes(monkeypatch):
    persisted = []
    on_errors = []
    finals = []
    promoted = []

    monkeypatch.setattr(runner_mod, "persist_event",
                        lambda ev, **kw: persisted.append(ev))
    monkeypatch.setattr(runner_mod, "after_stream",
                        lambda *a, **k: promoted.append(a))

    def boom():
        yield {"type": "status"}
        raise RuntimeError("kaboom")

    # 不应向上抛
    evs = _collect(runner_mod.run_path(
        "react", boom(),
        thread_id="t1", trace_id="tr",
        on_error=lambda e: on_errors.append(type(e).__name__),
        on_finally=lambda: finals.append(1),
    ))

    types = [e["type"] for e in evs]
    assert types[0] == "status"
    assert types[-1] == "error"
    err = evs[-1]
    assert "kaboom" in err["message"]
    assert err["trace_id"] == "tr"
    # error 事件也落短期流水
    assert any(e.get("type") == "error" for e in persisted)
    # 即使异常也触发升迁 + finally
    assert promoted == [("t1", None, None)]
    assert on_errors == ["RuntimeError"]
    assert finals == [1]


def test_run_path_on_event_exception_does_not_break_stream(monkeypatch):
    monkeypatch.setattr(runner_mod, "persist_event", lambda ev, **kw: True)
    monkeypatch.setattr(runner_mod, "after_stream", lambda *a, **k: None)

    def bad_cb(ev):
        raise ValueError("cb broken")

    def stream():
        yield {"type": "a"}
        yield {"type": "b"}

    evs = _collect(runner_mod.run_path(
        "pe", stream(), thread_id="t", on_event=bad_cb,
    ))
    # on_event 抛异常不影响主流透传
    assert [e["type"] for e in evs] == ["a", "b"]
