# -*- coding: utf-8 -*-
"""工具故障自适应决策测试(缺口③)。

验证 agent 能感知工具健康度并调整策略:
  - 本轮故障类别(tool_status down)与跨轮熔断(breaker open)都会把对应工具摘出 schema;
  - system 提示注入故障说明与替代策略(检索挂了 -> 基于通用知识谨慎作答 / 告知用户);
  - 全部工具不可用时不传 tools(纯文本作答);特性开关关闭时不干预。
"""
import os
import sys
import time
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_reasoning.ReAct.core import nodes  # noqa: E402
from agent_reasoning.ReAct.core.graph import build_graph  # noqa: E402
from agent_reasoning.ReAct.trace import TraceRecorder  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from tools import Category  # noqa: E402


# ---------------- 假 LLM 流(复用结构) ----------------
class _Delta:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _Choice:
    def __init__(self, delta, finish_reason=None):
        self.delta = delta
        self.finish_reason = finish_reason


class _Chunk:
    def __init__(self, choice=None, usage=None):
        self.choices = [choice] if choice is not None else []
        self.usage = usage


class _TCDelta:
    def __init__(self, index, id=None, name=None, arguments=None):
        self.index = index
        self.id = id
        self.function = types.SimpleNamespace(name=name, arguments=arguments)


def _usage():
    return types.SimpleNamespace(prompt_tokens=10, completion_tokens=20, total_tokens=30)


def _answer_stream(text="答案"):
    for ch in text:
        yield _Chunk(_Choice(_Delta(content=ch)))
    yield _Chunk(_Choice(_Delta(), finish_reason="stop"), usage=_usage())


def _tool_then_answer_stream(tc_id="call_1", name="search_text", args='{"query": "ALD"}'):
    yield _Chunk(_Choice(_Delta(tool_calls=[
        _TCDelta(0, id=tc_id, name=name, arguments=args),
    ]), finish_reason="tool_calls"))
    yield _Chunk(usage=_usage())


def _make_capture_client(script):
    """每次 create 记录 kwargs,返回 (client, captured_list)。"""
    it = iter(script)
    captured: list = []

    class _Completions:
        @staticmethod
        def create(**kwargs):
            captured.append(kwargs)
            return next(it)

    class _Chat:
        completions = _Completions()

    return types.SimpleNamespace(chat=_Chat()), captured


def _run_graph(inputs, client_script, *, dispatch_fn=None):
    recorder = TraceRecorder("trace0", inputs["started_at"], inputs["question"])
    cfg = {"configurable": {
        "thread_id": "t1", "user_id": "alice", "trace_recorder": recorder}}
    g = build_graph(checkpointer=InMemorySaver())
    client, captured = _make_capture_client(client_script)
    old_client = nodes.get_client
    old_dispatch = nodes.dispatch
    nodes.get_client = lambda: client
    nodes.dispatch = dispatch_fn or (lambda n, a: [])
    try:
        evs = list(g.stream({
            "question": inputs["question"],
            "history": [],
            "started_at": inputs["started_at"],
            "max_steps": 6, "max_total_seconds": 60,
        }, config=cfg, stream_mode="custom"))
    finally:
        nodes.get_client = old_client
        nodes.dispatch = old_dispatch
    return evs, captured


def _tool_names(kwargs):
    return {t["function"]["name"] for t in (kwargs.get("tools") or [])}


# ---------------- 单元:健康度汇总 ----------------
def test_unavailable_tools_from_down_category():
    unavailable, cats = nodes._unavailable_tools({Category.RETRIEVAL: "down"})
    # 检索三件套都应被摘除
    assert {"search_text", "search_image", "get_chunk"} <= unavailable
    assert Category.RETRIEVAL in cats


def test_unavailable_tools_all_up():
    unavailable, cats = nodes._unavailable_tools({c: "up" for c in nodes.ALL_CATEGORIES})
    assert unavailable == set()
    assert cats == []


def test_unavailable_tools_from_open_breaker(monkeypatch):
    # 模拟检索工具熔断打开(跨轮持久信号)
    monkeypatch.setattr(nodes, "circuit_snapshot",
                        lambda: {"search_text": "open", "search_image": "closed"})
    unavailable, cats = nodes._unavailable_tools({})
    assert "search_text" in unavailable
    assert "search_image" not in unavailable
    assert Category.RETRIEVAL in cats


def test_health_half_open_not_removed(monkeypatch):
    # half_open 是试探态,仍可调用,不应摘除
    monkeypatch.setattr(nodes, "circuit_snapshot",
                        lambda: {"search_text": "half_open"})
    unavailable, cats = nodes._unavailable_tools({})
    assert "search_text" not in unavailable


def test_health_adapt_disabled(monkeypatch):
    import config as C
    monkeypatch.setattr(C, "TOOL_HEALTH_ADAPT_ENABLED", False)
    monkeypatch.setattr(nodes, "circuit_snapshot",
                        lambda: {"search_text": "open"})
    unavailable, cats = nodes._unavailable_tools({})
    assert "search_text" not in unavailable
    assert cats == []


def test_health_block_content():
    block = nodes._tool_health_block({"search_text", "search_image", "get_chunk"},
                                    [Category.RETRIEVAL])
    assert "本地知识库" in block
    assert "通用知识谨慎作答" in block
    assert nodes._tool_health_block(set(), []) == ""


# ---------------- 真实熔断器端到端 ----------------
def test_real_breaker_open_excludes_tool():
    import importlib
    # 熔断器已从 tools.dispatch 迁到 ReAct 支撑层 tool_circuit(由韧性中间件驱动)
    circuit_mod = importlib.import_module(
        "agent_reasoning.ReAct.support.tool_circuit")
    brk = circuit_mod._breakers.get("search_text")
    for _ in range(99):
        brk.on_failure("retryable")
    try:
        assert nodes.circuit_snapshot().get("search_text") == "open"
        unavailable, cats = nodes._unavailable_tools({})
        assert "search_text" in unavailable
        assert Category.RETRIEVAL in cats
    finally:
        circuit_mod._breakers.reset_all()


# ---------------- 图级:检索故障后第二轮摘 schema + 注入提示 ----------------
def test_graph_retrieval_down_filters_schema_and_advises():
    # 第一轮模型调 search_text,dispatch 返回重试类错误 -> 检索类别标 down;
    # 第二轮 agent 应摘掉检索工具并在 system 提示中告知替代策略。
    def failing_dispatch(name, args):
        return {"error": "检索服务连接失败", "error_type": "retryable", "tool": name}

    evs, captured = _run_graph(
        {"question": "ALD 是什么", "started_at": time.time()},
        [_tool_then_answer_stream(), _answer_stream("通用知识答案")],
        dispatch_fn=failing_dispatch,
    )

    assert len(captured) == 2
    first_tools = _tool_names(captured[0])
    second_tools = _tool_names(captured[1])

    # 第一轮 schema 完整(含检索工具)
    assert "search_text" in first_tools
    # 第二轮检索三件套被摘除
    assert "search_text" not in second_tools
    assert "search_image" not in second_tools
    assert "get_chunk" not in second_tools

    # 第二轮 messages 末尾追加了健康度 system 提示
    last_msgs = captured[1]["messages"]
    sys_msgs = [m for m in last_msgs if m["role"] == "system"]
    assert sys_msgs and "本地知识库" in sys_msgs[-1]["content"]
    assert "通用知识谨慎作答" in sys_msgs[-1]["content"]

    # 前端收到切换策略的 status 事件
    assert any("备用策略" in e.get("message", "") for e in evs if e["type"] == "status")


def test_graph_healthy_binds_all_tools():
    # 正常路径:无故障,每轮都绑定完整 schema,不注入健康度提示
    evs, captured = _run_graph(
        {"question": "ALD 是什么", "started_at": time.time()},
        [_tool_then_answer_stream(), _answer_stream("ALD 答案")],
        dispatch_fn=lambda n, a: [
            {"chunk_id": "c1", "source_stem": "doc1", "page": "p1",
             "score": 0.3, "content": "ALD..."}],
    )
    for kw in captured:
        names = _tool_names(kw)
        assert "search_text" in names
        sys_msgs = [m for m in kw["messages"] if m["role"] == "system"]
        assert not any("工具健康度提示" in m["content"] for m in sys_msgs)
