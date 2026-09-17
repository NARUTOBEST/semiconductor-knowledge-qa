# -*- coding: utf-8 -*-
"""Req4:流式缓冲测试。

- 绑工具的终答:content token 先缓冲,流末确认无 tool_calls 才一次性回放(1 个 token 事件);
- 同一轮出现 tool_calls:缓冲正文丢弃(不把"思考串"当答案吐给前端);
- REACT_STREAM_BUFFER_ENABLED=False:逐 token 即时下发(每字 1 个 token 事件)。
"""
import os
import sys
import time
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402

import config as C  # noqa: E402
from agent_reasoning.ReAct.core import nodes  # noqa: E402
from agent_reasoning.ReAct.core.loop import react_loop  # noqa: E402
from agent_reasoning.ReAct.trace import TraceRecorder  # noqa: E402


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


def _answer_stream(text="最终答案"):
    for ch in text:
        yield _Chunk(_Choice(_Delta(content=ch)))
    yield _Chunk(_Choice(_Delta(), finish_reason="stop"), usage=_usage())


def _tool_stream_with_thought(thought="先想想再查资料"):
    """同一轮里先吐正文(思考串),再吐 tool_calls —— 正文应被缓冲丢弃。"""
    for ch in thought:
        yield _Chunk(_Choice(_Delta(content=ch)))
    yield _Chunk(_Choice(_Delta(tool_calls=[
        _TCDelta(0, id="call_1", name="search_text",
                 arguments='{"query": "ALD"}'),
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


def _drain(monkeypatch, script, dispatch_fn=None):
    fake = _make_client(script)
    monkeypatch.setattr(nodes, "get_client", lambda: fake)
    if dispatch_fn is not None:
        monkeypatch.setattr(nodes, "dispatch", dispatch_fn)
    recorder = TraceRecorder("tr", time.time(), "q")
    cfg = {"thread_id": "t1", "user_id": "alice", "trace_recorder": recorder}
    gen = react_loop(
        [SystemMessage(content="sys", id="p"), HumanMessage(content="你好")],
        max_steps=6, max_total_seconds=60,
        configurable=cfg, question="q", trace_id="tr",
    )
    evs, final = [], None
    try:
        while True:
            evs.append(next(gen))
    except StopIteration as e:
        final = e.value
    return evs, final


def test_final_answer_replayed_as_single_token(monkeypatch):
    monkeypatch.setattr(C, "REACT_STREAM_BUFFER_ENABLED", True, raising=False)
    evs, final = _drain(monkeypatch, [_answer_stream("最终答案")])
    token_evs = [e for e in evs if e["type"] == "token"]
    # 缓冲回放:终答合并为 1 个 token 事件,内容完整
    assert len(token_evs) == 1
    assert token_evs[0]["delta"] == "最终答案"
    assert final["final_reason"] == "answer"


def test_buffered_content_discarded_when_tool_calls(monkeypatch):
    monkeypatch.setattr(C, "REACT_STREAM_BUFFER_ENABLED", True, raising=False)
    evs, final = _drain(
        monkeypatch,
        [_tool_stream_with_thought("先想想再查资料"), _answer_stream("查到了")],
        dispatch_fn=lambda n, a: [],
    )
    deltas = "".join(e.get("delta", "") for e in evs if e["type"] == "token")
    # 思考串绝不能作为答案 token 下发;只有第二轮终答被回放
    assert "先想想再查资料" not in deltas
    assert deltas == "查到了"
    assert final["final_reason"] == "answer"


def test_buffer_disabled_streams_per_char(monkeypatch):
    monkeypatch.setattr(C, "REACT_STREAM_BUFFER_ENABLED", False, raising=False)
    evs, final = _drain(monkeypatch, [_answer_stream("直接答")])
    token_evs = [e for e in evs if e["type"] == "token"]
    # 关闭缓冲:每个 content delta 即时下发(3 个字 -> 3 个 token 事件)
    assert len(token_evs) == 3
    assert "".join(e["delta"] for e in token_evs) == "直接答"
    assert final["final_reason"] == "answer"
