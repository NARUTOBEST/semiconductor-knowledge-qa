# -*- coding: utf-8 -*-
"""Req10:validate 条件回边测试。

generation/runtime 全非法 -> 直接回 agent(不空转经过 execute);有合法 -> 进下一节点;
混合批 -> 合法的继续、非法的回灌错误 ToolMessage。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import AIMessage  # noqa: E402

from agent_reasoning.ReAct.core import validate_nodes as vn  # noqa: E402
from agent_reasoning.ReAct.core import loop  # noqa: E402


def _writer(monkeypatch, sink):
    monkeypatch.setattr(vn, "get_stream_writer",
                        lambda: (lambda ev: sink.append(ev)), raising=False)


def _state(tool_calls):
    return {"trace_id": "tr", "step": 1,
            "messages": [AIMessage(content="", tool_calls=tool_calls)]}


def test_generation_all_invalid_routes_back_to_agent(monkeypatch):
    _writer(monkeypatch, [])
    # 工具名幻觉:不存在的工具
    state = _state([{"id": "x1", "name": "nonexistent_tool", "args": {}}])
    out = vn.validate_generation_node(state, {})
    assert out["pending_tool_calls"] == []           # 无合法调用
    assert out["messages"]                           # 回灌了错误反馈(ToolMessage)
    # 条件路由:全非法 -> 回 agent
    merged = {**state, **out}
    assert loop._route_after_generation(merged) == "agent"


def test_generation_valid_routes_to_runtime(monkeypatch):
    _writer(monkeypatch, [])
    state = _state([{"id": "x1", "name": "search_text",
                     "args": {"query": "ALD"}}])
    out = vn.validate_generation_node(state, {})
    assert [c["name"] for c in out["pending_tool_calls"]] == ["search_text"]
    merged = {**state, **out}
    assert loop._route_after_generation(merged) == "runtime"


def test_generation_mixed_batch_keeps_valid(monkeypatch):
    _writer(monkeypatch, [])
    state = _state([
        {"id": "a", "name": "search_text", "args": {"query": "ALD"}},
        {"id": "b", "name": "hallucinated_xyz", "args": {}},
    ])
    out = vn.validate_generation_node(state, {})
    # 合法的保留进 pending,非法的不进
    assert [c["name"] for c in out["pending_tool_calls"]] == ["search_text"]
    assert loop._route_after_generation({**state, **out}) == "runtime"


def test_runtime_all_invalid_routes_back_to_agent(monkeypatch):
    _writer(monkeypatch, [])
    # search_text 缺必填 query -> schema 校验失败
    state = {"trace_id": "tr", "step": 1,
             "messages": [],
             "pending_tool_calls": [{"id": "r1", "name": "search_text", "args": {}}],
             "tool_parse_errors": {}}
    out = vn.validate_runtime_node(state, {})
    assert out["pending_tool_calls"] == []
    assert any(getattr(m, "type", "") == "tool" for m in out["messages"])
    assert loop._route_after_runtime({**state, **out}) == "agent"


def test_runtime_valid_routes_to_execute(monkeypatch):
    _writer(monkeypatch, [])
    state = {"trace_id": "tr", "step": 1, "messages": [],
             "pending_tool_calls": [{"id": "r1", "name": "search_text",
                                     "args": {"query": "ALD"}}],
             "tool_parse_errors": {}}
    out = vn.validate_runtime_node(state, {})
    assert [c["name"] for c in out["pending_tool_calls"]] == ["search_text"]
    assert loop._route_after_runtime({**state, **out}) == "execute"
