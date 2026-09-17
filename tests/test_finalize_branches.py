# -*- coding: utf-8 -*-
"""Req5:finalize 终态分支测试。

- answer          :正常 assistant_message + sources 卡;
- max_steps/timeout:assistant_message 带 incomplete:true / incomplete_reason;
- error           :只发 error 事件(agent 节点已发),finalize 不发答案卡/来源;
- done(emit_done):强制携带 final_reason。
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_reasoning.ReAct.core import nodes  # noqa: E402
from agent_reasoning.ReAct.trace import TraceRecorder  # noqa: E402


def _run_node(monkeypatch, state):
    sink = []
    monkeypatch.setattr(nodes, "get_stream_writer",
                        lambda: (lambda ev: sink.append(ev)), raising=False)
    recorder = TraceRecorder("tr", time.time(), "q")
    config = {"configurable": {"trace_recorder": recorder}}
    base = {"trace_id": "tr", "started_at": time.time(), "step": 1,
            "collected_sources": {}, "retrieval_max_score": 0.0,
            "search_count": 0}
    base.update(state)
    patch = nodes.finalize_node(base, config)
    done = {}
    # emit_done 在 finalize 之后,读 finalize 回写的 final_reason
    merged = dict(base)
    merged.update(patch)
    nodes.emit_done_node(merged, config)
    done_ev = [e for e in sink if e["type"] == "done"]
    return sink, patch, (done_ev[0] if done_ev else {})


def test_answer_branch_emits_message_and_sources(monkeypatch):
    sink, patch, done = _run_node(monkeypatch, {
        "final_reason": "answer",
        "full_reply": "ALD 是原子层沉积工艺。",
        "collected_sources": {"c1": {"chunk_id": "c1", "content": "ALD 是…",
                                     "score": 0.9, "source_stem": "doc1"}},
    })
    msgs = [e for e in sink if e["type"] == "assistant_message"]
    assert len(msgs) == 1
    assert msgs[0]["content"] == "ALD 是原子层沉积工艺。"
    assert msgs[0].get("incomplete") is not True
    assert done["final_reason"] == "answer"
    assert patch["final_reason"] == "answer"


def test_max_steps_branch_marks_incomplete(monkeypatch):
    sink, patch, done = _run_node(monkeypatch, {
        "final_reason": "max_steps",
        "full_reply": "目前查到部分资料…",
    })
    msgs = [e for e in sink if e["type"] == "assistant_message"]
    assert len(msgs) == 1
    assert msgs[0]["incomplete"] is True
    assert msgs[0]["incomplete_reason"] == "max_steps"
    assert msgs[0]["note"]
    assert done["final_reason"] == "max_steps"


def test_timeout_branch_marks_incomplete(monkeypatch):
    sink, patch, done = _run_node(monkeypatch, {
        "final_reason": "timeout",
        "full_reply": "部分结果",
    })
    msgs = [e for e in sink if e["type"] == "assistant_message"]
    assert msgs[0]["incomplete"] is True
    assert msgs[0]["incomplete_reason"] == "timeout"
    assert done["final_reason"] == "timeout"


def test_error_branch_no_answer_card_or_sources(monkeypatch):
    sink, patch, done = _run_node(monkeypatch, {
        "final_reason": "error",
        "full_reply": "",
        "error": {"phase": "llm_create", "message": "boom"},
    })
    assert [e for e in sink if e["type"] == "assistant_message"] == []
    assert [e for e in sink if e["type"] == "sources"] == []
    assert done["final_reason"] == "error"
    assert patch["error"] == {"phase": "llm_create", "message": "boom"}
