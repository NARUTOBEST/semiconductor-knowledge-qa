# -*- coding: utf-8 -*-
"""LangGraph 版 ReAct 的事件契约测试。

mock 掉 LLM 流、检索工具,断言:
  - answer / 单工具 / LLM失败 / 超时 / max_steps 各路径 yield 的事件序列
  - state 正确累积(messages/usage/final_reason/collected_sources)
  - 两级范式无旁路:一轮直答即 done(无 grounding/reflect);短期流水白名单接线
"""
import os
import sys
import time
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_reasoning.ReAct.core import nodes  # noqa: E402
from agent_reasoning.ReAct.core.graph import build_graph  # noqa: E402
from memories.orchestration import persist_event  # noqa: E402
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


def _tool_then_answer_stream(tc_id="call_1", args='{"query": "ALD"}',
                             name="search_text"):
    """第一轮:一个工具调用;无正文。(由脚本控制第二轮再 answer)"""
    yield _Chunk(_Choice(_Delta(tool_calls=[
        _TCDelta(0, id=tc_id, name=name, arguments=args),
    ]), finish_reason="tool_calls"))
    yield _Chunk(usage=_usage())


def _multi_tool_stream(calls):
    """一轮里并发发出多个工具调用;calls: [(tc_id, name, args_json), ...]。"""
    deltas = [_TCDelta(i, id=tc_id, name=name, arguments=args)
              for i, (tc_id, name, args) in enumerate(calls)]
    yield _Chunk(_Choice(_Delta(tool_calls=deltas), finish_reason="tool_calls"))
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


def _run_graph(inputs, client_script, *, dispatch_fn=None,
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
    # get_client 在节点内每步都调用,必须返回同一个 client 单例,
    # 否则每个新 client 的 script 迭代器都会从头开始,导致永远拿到第一个响应。
    fake_client = _make_client(client_script)
    nodes.get_client = lambda: fake_client
    nodes.dispatch = dispatch_fn or (lambda n, a: [])
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
    return evs, state


def _types(evs):
    return [e["type"] for e in evs]


# ---------------- 路径测试 ----------------
def test_answer_path():
    # 两级范式无旁路:模型一轮直答即 done(无 grounding/reflect)。
    evs, state = _run_graph(
        {"question": "你好", "started_at": time.time()},
        [_answer_stream("你好世界")],
    )
    types = _types(evs)
    # 关键事件齐全且有序
    assert types[0] == "status"
    assert "token" in types
    assert "llm_response" in types
    assert "grounding" not in types
    assert "reflect" not in types
    assert types[-1] == "done"
    assert state["final_reason"] == "answer"
    assert state["step"] == 1
    assert state["usage"]["total_tokens"] == 30
    # 最后一条是 assistant 消息
    assert state["messages"][-1].type == "ai"
    assert state["messages"][-1].content == "你好世界"


def test_single_tool_then_answer():
    sources = [{"chunk_id": "c1", "source_stem": "doc1", "page": "p1",
                "score": 0.9, "content": "ALD 是..."}]

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
    assert "grounding" not in types
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


def test_parallel_tool_calls_fan_out():
    # 一轮里发出 3 个无依赖的检索工具调用:应并发执行(线程池),而非串行累加。
    import threading
    calls = [
        ("c1", "search_text", '{"query": "ALD 原理"}'),
        ("c2", "search_image", '{"query": "设备图"}'),
        ("c3", "get_chunk", '{"chunk_id": "c1"}'),
    ]
    active = {"n": 0, "max": 0, "lock": threading.Lock()}

    def slow_dispatch(name, args):
        with active["lock"]:
            active["n"] += 1
            active["max"] = max(active["max"], active["n"])
        time.sleep(0.3)
        with active["lock"]:
            active["n"] -= 1
        return [{"chunk_id": f"ck-{name}", "source_stem": name,
                 "page": "p1", "score": 0.9, "content": "x"}]

    t0 = time.time()
    evs, state = _run_graph(
        {"question": "ALD 相关资料", "started_at": time.time()},
        [_multi_tool_stream(calls), _answer_stream("综合答案")],
        dispatch_fn=slow_dispatch,
    )
    elapsed = time.time() - t0

    # 三个工具都执行了
    tool_results = [e for e in evs if e["type"] == "tool_result"]
    assert len(tool_results) == 3
    # 并发:峰值同时在跑的调用数 >1;且总耗时远小于串行(0.9s)
    assert active["max"] >= 2, f"未并发,峰值并发={active['max']}"
    assert elapsed < 0.75, f"疑似串行,耗时 {elapsed:.2f}s"
    # 结果按 tool_call_id 对齐:三条 ToolMessage 齐全且 id 正确
    tool_msgs = [m for m in state["messages"] if getattr(m, "type", "") == "tool"]
    assert {m.tool_call_id for m in tool_msgs} == {"c1", "c2", "c3"}
    # 最终正常作答
    assert state["final_reason"] == "answer"


def test_parallel_disabled_runs_serial(monkeypatch):
    # TOOL_MAX_PARALLEL=1 时退化为串行(峰值并发=1)
    import threading
    import config as C
    monkeypatch.setattr(C, "TOOL_MAX_PARALLEL", 1)
    active = {"n": 0, "max": 0, "lock": threading.Lock()}

    def slow_dispatch(name, args):
        with active["lock"]:
            active["n"] += 1
            active["max"] = max(active["max"], active["n"])
        time.sleep(0.15)
        with active["lock"]:
            active["n"] -= 1
        return []

    calls = [
        ("c1", "search_text", '{"query": "a"}'),
        ("c2", "search_text", '{"query": "b"}'),
    ]
    _run_graph(
        {"question": "多查几个", "started_at": time.time()},
        [_multi_tool_stream(calls), _answer_stream("答")],
        dispatch_fn=slow_dispatch,
    )
    assert active["max"] == 1


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
    # 两轮工具参数必须不同,否则会被"重复工具调用检测"提前判 error 终止。
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


# ---------------- 短期流水接线 ----------------
def test_persist_event_whitelist(monkeypatch):
    from memories.storage.short import short_term as st
    assert persist_event({"type": "token", "delta": "x"},
                         thread_id="t") is False  # 跳过(token 不在白名单)

    captured = {}

    def _fake_append(thread_id, event_type, payload, **kw):
        captured.update(thread_id=thread_id, event_type=event_type,
                        payload=payload, **kw)
        return 1  # 模拟返回 seq

    # 打桩,不连真实 Redis(短期流水后端的键逻辑由 test_short_term_redis 覆盖)
    monkeypatch.setattr(st, "append_event", _fake_append)
    ok = persist_event(
        {"type": "assistant_message", "content": "hi", "trace": {"big": 1}},
        thread_id="t", user_id="alice")
    assert ok is True  # 白名单内
    assert captured["event_type"] == "assistant_message"
    assert captured["user_id"] == "alice"
    assert "trace" not in captured["payload"]  # trace 已被剔除
