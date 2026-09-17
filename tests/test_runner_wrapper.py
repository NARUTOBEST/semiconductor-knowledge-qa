# -*- coding: utf-8 -*-
"""通用 run_path 包装器测试。

验证它能包裹任意事件流,并统一处理:
  - 事件透传 + persist_event 落短期流水 + on_event 回调
  - 流抛异常时补发 error 事件且不向上抛;on_error 回调被调用
  - error 事件不外泄异常类型/文本,只回通用提示 + trace_id
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

    monkeypatch.setattr(runner_mod, "persist_event",
                        lambda ev, **kw: persisted.append((ev, kw)))

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


def test_run_path_emits_error_on_exception(monkeypatch):
    persisted = []
    on_errors = []

    monkeypatch.setattr(runner_mod, "persist_event",
                        lambda ev, **kw: persisted.append(ev))

    def boom():
        yield {"type": "status"}
        raise RuntimeError("kaboom")

    # 不应向上抛
    evs = _collect(runner_mod.run_path(
        "react", boom(),
        thread_id="t1", trace_id="tr",
        on_error=lambda e: on_errors.append(type(e).__name__),
    ))

    types = [e["type"] for e in evs]
    assert types[0] == "status"
    assert types[-1] == "error"
    err = evs[-1]
    # 安全:错误事件不外泄异常文本/类型(避免泄露内部路径/细节),
    # 只回通用提示 + 可用于排查的 trace_id(详细堆栈仅记服务端日志)。
    assert "kaboom" not in err["message"]
    assert "RuntimeError" not in err["message"]
    assert err["trace_id"] == "tr" and err["trace_id"] in err["message"]
    # error 事件也落短期流水
    assert any(e.get("type") == "error" for e in persisted)
    assert on_errors == ["RuntimeError"]


def test_run_path_on_event_exception_does_not_break_stream(monkeypatch):
    monkeypatch.setattr(runner_mod, "persist_event", lambda ev, **kw: True)

    def bad_cb(ev):
        raise ValueError("cb broken")

    def stream():
        yield {"type": "a"}
        yield {"type": "b"}

    evs = _collect(runner_mod.run_path(
        "react", stream(), thread_id="t", on_event=bad_cb,
    ))
    # on_event 抛异常不影响主流透传
    assert [e["type"] for e in evs] == ["a", "b"]
