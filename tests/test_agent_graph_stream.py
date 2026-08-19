# -*- coding: utf-8 -*-
"""LangGraph 重写版 ReAct 的事件契约测试。

mock 掉 LLM 流、检索工具、召回,断言:
  - answer / 单工具 / LLM失败 / 超时 / max_steps 各路径 yield 的事件序列
  - state 正确累积(messages/usage/final_reason/collected_sources)
  - 记忆接线:recall 被调用、persist_event 落短期、after_stream 触发升迁
"""
import os
import sys
import time
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_reasoning.ReAct.core import nodes  # noqa: E402
from agent_reasoning.ReAct.core.graph import build_graph  # noqa: E402
from memories.orchestration.short import events as short_events  # noqa: E402
from agent_reasoning.ReAct.trace import TraceRecorder  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402


# ---------------- 假 LLM 流 ----------------
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


def _usage(total_tokens=30):
    return types.SimpleNamespace(prompt_tokens=10, completion_tokens=20,
                                 total_tokens=total_tokens)


def _answer_stream(text="你好世界"):
    """逐 token 文本,无 tool_calls。"""
    for ch in text:
        yield _Chunk(_Choice(_Delta(content=ch)))
    yield _Chunk(_Choice(_Delta(), finish_reason="stop"), usage=_usage())


def _tool_then_answer_stream(tc_id="call_1", args='{"query": "ALD"}'):
    """第一轮:一个 search_text 工具调用;无正文。(由脚本控制第二轮再 answer)"""
    yield _Chunk(_Choice(_Delta(tool_calls=[
        _TCDelta(0, id=tc_id, name="search_text", arguments=args),
    ]), finish_reason="tool_calls"))
    yield _Chunk(usage=_usage())


def _make_client(script):
    """script: 可迭代对象列表,每次 create 返回下一个 (stream, None)。"""
    it = iter(script)

    class _Completions:
        @staticmethod
        def create(**kwargs):
            # 模拟真实 SDK:只返回 stream;由 llm_create_with_retry 包成 (stream, err)
            return next(it)

    class _Chat:
        completions = _Completions()

    return types.SimpleNamespace(chat=_Chat())


def _run_graph(inputs, client_script, *, dispatch_fn=None, recall_fn=None,
               max_steps=6, max_total_seconds=60, thread_id="t1"):
    """跑内存 checkpointer 的图,返回 (事件列表, 最终 state)。"""
    recorder = TraceRecorder("trace0", inputs["started_at"], inputs["question"])
    cfg = {"configurable": {
        "thread_id": thread_id, "user_id": "alice",
        "trace_recorder": recorder,
    }}
    g = build_graph(checkpointer=InMemorySaver())
    old_client = nodes.get_client
    old_dispatch = nodes.dispatch
    old_recall = nodes.recall_memories
    old_prompt = nodes.rewrite_query
    # get_client 在节点内每步都调用,必须返回同一个 client 单例,
    # 否则每个新 client 的 script 迭代器都会从头开始,导致永远拿到第一个响应。
    fake_client = _make_client(client_script)
    nodes.get_client = lambda: fake_client
    nodes.dispatch = dispatch_fn or (lambda n, a: [])
    nodes.recall_memories = recall_fn or (lambda *a, **k: [])
    nodes.rewrite_query = lambda m, h=None: [m]
    try:
        evs = list(g.stream({
            "question": inputs["question"],
            "history": inputs.get("history", []),
            "started_at": inputs["started_at"],
            "max_steps": max_steps,
            "max_total_seconds": max_total_seconds,
        }, config=cfg, stream_mode="custom"))
        state = g.get_state(cfg).values
    finally:
        nodes.get_client = old_client
        nodes.dispatch = old_dispatch
        nodes.recall_memories = old_recall
        nodes.rewrite_query = old_prompt
    return evs, state


def _types(evs):
    return [e["type"] for e in evs]


# ---------------- 路径测试 ----------------
def test_answer_path():
    # 无检索来源 + 检索服务正常 -> grounding 标记不通过并触发一次反思重生成
    # (缺口1修复:模型未检索就作答需被拦截);第二轮仍直接作答,达 MAX_REFLECT 后收尾。
    evs, state = _run_graph(
        {"question": "你好", "started_at": time.time()},
        [_answer_stream("你好世界"), _answer_stream("你好世界")],
    )
    types = _types(evs)
    # 关键事件齐全且有序
    assert types[0] == "status"
    assert "token" in types
    assert "llm_response" in types
    # 无检索来源现在也会发 grounding(passed=False)并触发 reflect
    g = [e for e in evs if e["type"] == "grounding"]
    assert g and g[0]["passed"] is False
    assert "reflect" in types
    assert types[-1] == "done"
    assert state["final_reason"] == "answer"
    assert state["step"] == 2  # 反思后多走一轮
    assert state["usage"]["total_tokens"] == 60
    # 最后一条是 assistant 消息(第二轮重生成的答案)
    assert state["messages"][-1].type == "ai"
    assert state["messages"][-1].content == "你好世界"


def test_single_tool_then_answer():
    # score < GROUNDING_FAITHFULNESS_THRESHOLD(0.5):有 grounding 事件但不触发忠实度 LLM 调用
    sources = [{"chunk_id": "c1", "source_stem": "doc1", "page": "p1",
                "score": 0.3, "content": "ALD 是..."}]

    def fake_dispatch(name, args):
        return list(sources)

    evs, state = _run_graph(
        {"question": "ALD 是什么", "started_at": time.time()},
        [_tool_then_answer_stream(), _answer_stream("ALD 答案")],
        dispatch_fn=fake_dispatch,
    )
    types = _types(evs)
    assert "tool_call" in types
    assert "tool_result" in types
    assert "sources" in types
    assert "grounding" in types          # 有来源 -> 做 grounding
    # 顺序:第一次 agent -> tools -> 第二次 agent -> done
    first_tool = types.index("tool_call")
    done_idx = types.index("done")
    # 第二次 token 出现在工具之后
    token_after_tool = [i for i, t in enumerate(types)
                        if t == "token" and i > first_tool]
    assert token_after_tool and token_after_tool[-1] < done_idx
    assert state["final_reason"] == "answer"
    assert state["step"] == 2
    assert "c1" in state["collected_sources"]
    # 有 tool 消息回传
    assert any(getattr(m, "type", "") == "tool" for m in state["messages"])


def test_llm_create_error(monkeypatch):
    # 不重试,直接返回错误,避免测试等待指数退避
    monkeypatch.setattr(
        nodes, "llm_create_with_retry",
        lambda client, trace_id="", retries=1, **k: (None, RuntimeError("boom")))
    evs, state = _run_graph(
        {"question": "x", "started_at": time.time()},
        [iter([])],  # 不会被消费
    )
    types = _types(evs)
    assert "error" in types
    assert state["final_reason"] == "error"
    assert state["error"]["phase"] == "llm_create"
    assert types[-1] == "done"


def test_timeout_routes_to_finalize():
    evs, state = _run_graph(
        {"question": "x", "started_at": time.time() - 100},  # 已超时
        [_answer_stream("ignored")],
        max_total_seconds=60,
    )
    assert any("超时" in e.get("message", "") for e in evs if e["type"] == "status")
    assert state["final_reason"] == "timeout"
    assert _types(evs)[-1] == "done"


def test_max_steps():
    # agent 每轮都要求调工具,直到超过 max_steps。
    # 两轮工具参数必须不同,否则会被"重复工具调用检测"提前判 error 终止(P0③)。
    evs, state = _run_graph(
        {"question": "x", "started_at": time.time()},
        [_tool_then_answer_stream(tc_id="c1", args='{"query": "ALD"}'),   # step1 -> tool
         _tool_then_answer_stream(tc_id="c2", args='{"query": "CVD"}'),   # step2 -> tool
         _answer_stream("never")],                                          # step3 不会执行
        dispatch_fn=lambda n, a: [],
        max_steps=2,
    )
    assert state["final_reason"] == "max_steps"
    types = _types(evs)
    # 恰好两次工具调用,没有第三次 agent 正文 token
    assert types.count("tool_call") == 2
    assert types[-1] == "done"


def test_recall_invoked_with_username():
    called = {}

    def fake_recall(user_id, query, **k):
        called["user_id"] = user_id
        called["query"] = query
        return []

    _run_graph({"question": "你好", "started_at": time.time()},
               [_answer_stream("hi")], recall_fn=fake_recall)
    assert called == {"user_id": "alice", "query": "你好"}


# ---------------- 短期/长期接线 ----------------
def test_persist_event_whitelist():
    assert short_events.persist_event({"type": "token", "delta": "x"},
                                      thread_id="t") is False  # 跳过
    ok = short_events.persist_event(
        {"type": "assistant_message", "content": "hi", "trace": {"big": 1}},
        thread_id="t", user_id="alice")
    assert ok is True  # 白名单内


def test_after_stream_triggers_promote(monkeypatch):
    from memories.orchestration.long import handlers as long_handlers
    calls = []
    monkeypatch.setattr(long_handlers, "promote_thread",
                        lambda tid, uid, session_id=None: calls.append((tid, uid)))
    long_handlers.after_stream("t1", "alice", session_id="s1")
    time.sleep(0.3)  # 等 daemon 线程
    assert calls == [("t1", "alice")]


def test_after_stream_skips_anonymous(monkeypatch):
    from memories.orchestration.long import handlers as long_handlers
    called = []
    monkeypatch.setattr(long_handlers, "promote_thread",
                        lambda *a, **k: called.append(1))
    long_handlers.after_stream("t1", None)
    time.sleep(0.1)
    assert called == []
