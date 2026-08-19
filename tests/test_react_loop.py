# -*- coding: utf-8 -*-
"""react_loop 独立生成器的契约测试(1.1 抽取)。

mock 掉 LLM 流/检索,验证:
  - 不依赖外层 LangGraph 图即可运行 agent↔tools 循环
  - yield SSE 事件 dict(token/tool_call/tool_result/llm_response 等)
  - 终态通过生成器 return 值(StopIteration.value)返回
  - step_instruction 会作为新 HumanMessage 追加到初始 messages
  - max_steps 有界
"""
import os
import sys
import time
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402

from agent_reasoning.ReAct.core import nodes  # noqa: E402
from agent_reasoning.ReAct.core.loop import react_loop  # noqa: E402
from agent_reasoning.ReAct.trace import TraceRecorder  # noqa: E402


# ---------------- 假 LLM 流(与 test_agent_graph_stream 同构) ----------------
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
    for ch in text:
        yield _Chunk(_Choice(_Delta(content=ch)))
    yield _Chunk(_Choice(_Delta(), finish_reason="stop"), usage=_usage())


def _tool_stream(tc_id="call_1", args='{"query": "ALD"}'):
    yield _Chunk(_Choice(_Delta(tool_calls=[
        _TCDelta(0, id=tc_id, name="search_text", arguments=args),
    ]), finish_reason="tool_calls"))
    yield _Chunk(usage=_usage())


def _make_client(script):
    it = iter(script)

    class _Completions:
        @staticmethod
        def create(**kwargs):
            return next(it)

    class _Chat:
        completions = _Completions()

    return types.SimpleNamespace(chat=_Chat())


def _run(messages, client_script, monkeypatch, *, step_instruction=None,
        max_steps=6, max_total_seconds=60, dispatch_fn=None):
    # 召回/改写 mock 为空,不产生外部调用;dispatch 在各用例按需覆盖。
    monkeypatch.setattr(nodes, "recall_memories", lambda *a, **k: [])
    monkeypatch.setattr(nodes, "rewrite_query", lambda m, h=None: [m])
    fake = _make_client(client_script)
    # get_client 在节点内通过模块绑定调用,必须 patch nodes.get_client;
    # 用同一个 fake client 单例(每步都调 get_client),否则迭代器会从头开始。
    monkeypatch.setattr(nodes, "get_client", lambda: fake)
    if dispatch_fn is not None:
        monkeypatch.setattr(nodes, "dispatch", dispatch_fn)
    recorder = TraceRecorder("tr", time.time(), "q")
    cfg = {"thread_id": "t1", "user_id": "alice", "trace_recorder": recorder}
    gen = react_loop(
        messages,
        step_instruction=step_instruction,
        max_steps=max_steps,
        max_total_seconds=max_total_seconds,
        configurable=cfg,
        question="q",
        trace_id="tr",
    )
    # 单次迭代:既收集事件,又通过 StopIteration.value 拿到 return 值
    evs = []
    final = None
    try:
        while True:
            evs.append(next(gen))
    except StopIteration as e:
        final = e.value
    return evs, final


def test_answer_path_returns_final_state(monkeypatch):
    msgs = [SystemMessage(content="sys", id="p"), HumanMessage(content="你好")]
    evs, final = _run(msgs, [_answer_stream("你好世界")], monkeypatch)
    types = [e["type"] for e in evs]
    assert "step_start" in types
    assert types.count("token") == 4  # 你好世界
    assert "llm_response" in types
    assert final is not None
    assert final["final_reason"] == "answer"
    assert final["full_reply"] == "你好世界"
    assert final["step"] == 1
    assert final["messages"][-1].content == "你好世界"


def test_tool_then_answer(monkeypatch):
    sources = [{"chunk_id": "c1", "source_stem": "doc1", "page": "p1",
                "score": 0.9, "content": "ALD 是..."}]
    msgs = [SystemMessage(content="sys", id="p"), HumanMessage(content="ALD")]
    evs, final = _run(
        msgs,
        [_tool_stream(), _answer_stream("ALD 答案")],
        monkeypatch,
        dispatch_fn=lambda n, a: list(sources),
    )
    types = [e["type"] for e in evs]
    assert "tool_call" in types
    assert "tool_result" in types
    assert "sources" in types
    assert final["final_reason"] == "answer"
    assert final["step"] == 2
    assert "c1" in final["collected_sources"]


def test_step_instruction_appends_human_message(monkeypatch):
    msgs = [SystemMessage(content="sys", id="p")]
    evs, final = _run(msgs, [_answer_stream("ok")], monkeypatch,
                      step_instruction="请回答步骤一:检索 X")
    # 进入子图的 messages 末尾应是步骤指令
    last_human = [m for m in final["messages"] if isinstance(m, HumanMessage)][-1]
    assert last_human.content == "请回答步骤一:检索 X"
    assert final["full_reply"] == "ok"


def test_max_steps_bounds_loop(monkeypatch):
    # 每轮都要工具,两轮参数不同避免重复检测
    msgs = [SystemMessage(content="sys", id="p"), HumanMessage(content="x")]
    evs, final = _run(
        msgs,
        [_tool_stream(tc_id="c1", args='{"query": "ALD"}'),
         _tool_stream(tc_id="c2", args='{"query": "CVD"}')],
        monkeypatch,
        max_steps=2,
    )
    types = [e["type"] for e in evs]
    assert types.count("tool_call") == 2
    assert final["final_reason"] == "max_steps"


def test_bind_tools_false_omits_tools_kwarg(monkeypatch):
    # bind_tools=False(simple 直答):LLM 调用不得带 tools/tool_choice,
    # 且即便 mock 流里没有 tool_calls,也应正常作答、不进入工具节点。
    monkeypatch.setattr(nodes, "recall_memories", lambda *a, **k: [])
    monkeypatch.setattr(nodes, "rewrite_query", lambda m, h=None: [m])

    captured = {}

    def fake_create_with_retry(client, trace_id="", retries=1, **kwargs):
        captured["kwargs"] = kwargs
        return _answer_stream("直接答案"), None

    # nodes.py 通过 `from ..support.llm import llm_create_with_retry` 绑定,
    # 必须 patch nodes 模块上的名字才生效。
    monkeypatch.setattr(nodes, "llm_create_with_retry",
                        fake_create_with_retry)

    msgs = [SystemMessage(content="sys", id="p"), HumanMessage(content="你好")]
    recorder = TraceRecorder("tr", time.time(), "你好")
    cfg = {"thread_id": "t1", "user_id": "alice", "trace_recorder": recorder}
    gen = react_loop(msgs, max_steps=6, max_total_seconds=60,
                     configurable=cfg, question="你好", trace_id="tr",
                     bind_tools=False)
    evs = []
    final = None
    try:
        while True:
            evs.append(next(gen))
    except StopIteration as e:
        final = e.value

    kw = captured["kwargs"]
    assert "tools" not in kw
    assert "tool_choice" not in kw
    assert final["final_reason"] == "answer"
    assert final["full_reply"] == "直接答案"
    assert final["bind_tools"] is False
    assert [e["type"] for e in evs].count("tool_call") == 0
