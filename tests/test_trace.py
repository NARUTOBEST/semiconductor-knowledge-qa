# -*- coding: utf-8 -*-
"""ReAct trace / observability tests(经新版 LangGraph 实现)。

经 chat.service.react_stream -> agent_reasoning.ReAct.support.runner 跑内存 checkpointer 的图,
mock 掉 LLM 流、检索、召回、短期落库,验证产出的结构化事件与完整 trace:
  step_start / llm_response / tool_call / tool_result / step_end /
  grounding / error_trace,以及 done 事件携带的完整 trace。
"""
import contextlib
import json
import pytest
from unittest.mock import patch, MagicMock

import chat.service as svc
import agent_reasoning.ReAct.core.nodes as nodes
import agent_reasoning.ReAct.support.runner as runner
from langgraph.checkpoint.memory import InMemorySaver


# ---------- 流式 chunk 构造 ----------

def _chunk(content=None, tool_calls=None, finish_reason=None, usage=None):
    """构造一个 OpenAI 流式 chunk(MagicMock)。"""
    c = MagicMock()
    choice = MagicMock()
    choice.finish_reason = finish_reason
    delta = MagicMock()
    delta.content = content
    # tool_calls: list of {"index","id","name","arguments"}
    delta.tool_calls = None
    if tool_calls:
        delta.tool_calls = []
        for tc in tool_calls:
            m = MagicMock()
            m.index = tc.get("index", 0)
            m.id = tc.get("id")
            fn = MagicMock()
            fn.name = tc.get("name")
            fn.arguments = tc.get("arguments")
            m.function = fn
            delta.tool_calls.append(m)
    choice.delta = delta
    c.choices = [choice]
    if usage is not None:
        u = MagicMock()
        u.prompt_tokens = usage.get("prompt_tokens", 0)
        u.completion_tokens = usage.get("completion_tokens", 0)
        u.total_tokens = usage.get("total_tokens", 0)
        c.usage = u
    else:
        c.usage = None
    return c


def _usage_chunk(prompt=10, completion=20):
    c = MagicMock()
    c.choices = []
    u = MagicMock()
    u.prompt_tokens = prompt
    u.completion_tokens = completion
    u.total_tokens = prompt + completion
    c.usage = u
    return c


def _answer_stream(text="ALD 是一种薄膜沉积技术。", usage=None):
    return iter([
        _chunk(content=text[:5]),
        _chunk(content=text[5:]),
        _chunk(finish_reason="stop"),
        _usage_chunk() if usage is None else _usage_chunk(**usage),
    ])


def _toolcall_stream(tc_id="call_1", name="search_text",
                     arguments='{"query": "ALD principle", "k": 3}'):
    return iter([
        _chunk(tool_calls=[{"index": 0, "id": tc_id, "name": name, "arguments": ""}]),
        _chunk(tool_calls=[{"index": 0, "arguments": arguments}]),
        _chunk(finish_reason="tool_calls"),
        _usage_chunk(prompt=15, completion=25),
    ])


def _collect(events):
    by_type = {}
    order = []
    for ev in events:
        order.append(ev["type"])
        by_type.setdefault(ev["type"], []).append(ev)
    return order, by_type


@pytest.fixture
def _isolated_deps():
    """隔离外部依赖:内存 checkpointer、不连短期/长期库、改写/LLM/消息/dispatch 打桩。"""
    @contextlib.contextmanager
    def _mem_saver():
        yield InMemorySaver()

    # 节点内依赖:在 agent_reasoning.ReAct.core.nodes 命名空间打桩
    node_patches = [
        patch.object(nodes, "rewrite_query", return_value=["ALD principle"]),
        patch.object(nodes, "recall_memories", return_value=[]),
        patch.object(nodes, "get_client", return_value=MagicMock()),
        patch.object(nodes, "build_messages",
                     side_effect=lambda msg, hist, sub_queries=None: [
                         {"role": "user", "content": msg}]),
        patch.object(nodes, "dispatch", side_effect=lambda n, a: []),
    ]
    # runner 依赖:用内存 checkpointer 替换 PostgresSaver,跳过升迁、静默短期落库
    import memories.orchestration.short.events as _se
    from memories.storage.short import short_term as _short
    runner_patches = [
        patch.object(runner, "working_saver", _mem_saver),
        patch.object(runner, "after_stream", lambda *a, **k: None),
        patch.object(_se.short_term, "append_event", lambda *a, **k: None),
        patch.object(_short, "append_event", lambda *a, **k: None),
        # 阶段 3:react_stream 现先做复杂度路由;固定走 medium ReAct,避免真实 LLM 分类
        patch.object(svc, "classify_complexity",
                     return_value={"tier": "medium", "confidence": 1.0,
                                   "source": "rule"}),
    ]
    for p in node_patches + runner_patches:
        p.start()
    yield
    for p in node_patches + runner_patches:
        p.stop()


def _run(message="什么是 ALD?", **kwargs):
    """通过 service.react_stream 跑图(匿名用户 -> 不触发长期升迁)。"""
    return list(svc.react_stream(message, [], thread_id="t-trace", **kwargs))


# ---------- 测试 ----------

class TestAnswerOnlyTrace:
    def test_full_trace_structure_and_order(self, _isolated_deps):
        # 无检索来源的回答会被 grounding 拦截并触发一次反思重生成(缺口1修复),
        # 因此需要两个 answer stream:第一轮被作废、第二轮经 MAX_REFLECT 上限后收尾。
        streams = [_answer_stream(), _answer_stream()]
        with patch.object(nodes, "llm_create_with_retry",
                          side_effect=lambda *a, **k: (streams.pop(0), None)):
            events = _run()

        order, by_type = _collect(events)

        # 关键事件顺序(阶段 3:首个事件为 tier,随后才是 status)
        assert order[0] == "tier"
        assert "status" in order
        assert "step_start" in order
        assert "llm_response" in order
        assert "step_end" in order
        assert "grounding" in order   # 无来源也发 grounding(passed=False)
        assert "reflect" in order    # 触发反思重生成
        assert order[-1] == "done"

        done = by_type["done"][-1]
        trace = done["trace"]
        assert trace["final_reason"] == "answer"
        assert trace["steps_count"] == 2  # 第一轮 answer 被反思,第二轮收尾
        assert trace["steps"][-1]["decision"] == "answer"
        assert trace["steps"][-1]["llm"]["finish_reason"] == "stop"
        assert trace["steps"][-1]["llm"]["has_tool_calls"] is False
        assert trace["steps"][-1]["llm"]["thought_len"] > 0
        assert trace["total_tokens"]["total"] == 60
        assert trace["sub_queries"] == ["ALD principle"]

    def test_token_events_accumulate_to_full_reply(self, _isolated_deps):
        text = "ALD 是一种薄膜沉积技术。"
        # 第一轮先输出被作废的占位答案(触发无来源反思),第二轮输出真正答案。
        streams = [_answer_stream(text="占位"), _answer_stream(text=text)]
        with patch.object(nodes, "llm_create_with_retry",
                          side_effect=lambda *a, **k: (streams.pop(0), None)):
            events = _run()
        # 用户最终看到的 assistant_message 只含第二轮重生成的内容
        # (第一轮已由 reflect 置 full_reply="" 作废,不与新答案串联)
        final = [e for e in events if e["type"] == "assistant_message"]
        assert final and final[-1]["content"] == text

    def test_legacy_events_still_present(self, _isolated_deps):
        """回归:token/status/done/meta 事件不受影响。"""
        with patch.object(nodes, "llm_create_with_retry",
                          return_value=(_answer_stream(), None)):
            events = _run("hi")
        types = {e["type"] for e in events}
        assert {"token", "status", "meta", "done"} <= types
        meta = [e for e in events if e["type"] == "meta"][-1]
        assert "elapsed_ms" in meta and "steps" in meta


class TestToolCallTrace:
    def test_tool_call_and_result_events(self, _isolated_deps):
        tool_result = [{"chunk_id": "d__t1", "source_stem": "manual",
                        "page_start": 12, "score": 0.9, "content": "ALD..."}]
        answer = iter([_chunk(content="根据资料,ALD..."),
                       _chunk(finish_reason="stop"),
                       _usage_chunk(prompt=20, completion=10)])
        streams = [_toolcall_stream(), answer]

        with patch.object(nodes, "llm_create_with_retry",
                          side_effect=lambda *a, **k: (streams.pop(0), None)), \
             patch.object(nodes, "dispatch", return_value=tool_result) as disp, \
             patch.object(nodes, "grounding_check",
                          return_value={"passed": True, "warnings": []}):
            events = _run("ALD 原理?")

        _, by_type = _collect(events)

        tc = by_type["tool_call"][0]
        assert tc["name"] == "search_text"
        assert tc["args"]["query"] == "ALD principle"
        assert tc["args_parse_error"] is None

        tr = by_type["tool_result"][0]
        assert tr["ok"] is True
        assert tr["duration_ms"] >= 0
        assert tr["error"] is None
        assert "ALD" in tr["result_preview"]

        se = by_type["step_end"]
        # 第一轮 decision=tool_calls, 第二轮 decision=answer
        assert se[0]["decision"] == "tool_calls"
        assert se[0]["new_sources_count"] == 1
        assert se[-1]["decision"] == "answer"

        trace = by_type["done"][-1]["trace"]
        assert trace["steps_count"] == 2
        tools = trace["steps"][0]["tools"]
        assert len(tools) == 1
        assert tools[0]["name"] == "search_text"
        assert tools[0]["ok"] is True
        assert tools[0]["duration_ms"] >= 0
        assert trace["steps"][0]["decision"] == "tool_calls"
        disp.assert_called_once_with("search_text", {"query": "ALD principle", "k": 3})

    def test_tool_error_recorded(self, _isolated_deps):
        tool_result = {"error": "检索服务连接失败"}
        answer = iter([_chunk(content="抱歉,检索不可用。"),
                       _chunk(finish_reason="stop"), _usage_chunk()])
        streams = [_toolcall_stream(), answer]
        with patch.object(nodes, "llm_create_with_retry",
                          side_effect=lambda *a, **k: (streams.pop(0), None)), \
             patch.object(nodes, "dispatch", return_value=tool_result):
            events = _run("ALD?")
        _, by_type = _collect(events)
        tr = by_type["tool_result"][0]
        assert tr["ok"] is False
        assert "连接失败" in tr["error"]
        trace = by_type["done"][-1]["trace"]
        assert trace["steps"][0]["tools"][0]["ok"] is False

    def test_invalid_tool_arguments_parse_error(self, _isolated_deps):
        """参数 JSON 非法时不真正执行工具(P0①:避免空参数兜底触发非预期调用、
        假来源混入 collected_sources),把解析错误作为 tool_result 回传给模型自修。"""
        bad_stream = _toolcall_stream(arguments="{not valid json")
        answer = iter([_chunk(content="好的"), _chunk(finish_reason="stop"),
                       _usage_chunk()])
        streams = [bad_stream, answer]
        with patch.object(nodes, "llm_create_with_retry",
                          side_effect=lambda *a, **k: (streams.pop(0), None)), \
             patch.object(nodes, "dispatch", return_value=[]) as disp:
            events = _run("ALD?")
        _, by_type = _collect(events)
        tc = by_type["tool_call"][0]
        assert tc["args_parse_error"] is not None
        # dispatch 不被调用(参数坏了不能带着空参数执行)
        disp.assert_not_called()
        # 仍产生 tool_result,内容是让模型修正参数的错误提示
        assert by_type["tool_result"]
        tr = by_type["tool_result"][0]
        assert tr["ok"] is False
        assert "参数 JSON 解析失败" in tr["result_preview"]


class TestErrorTrace:
    def test_llm_stream_exception_emits_error_trace(self, _isolated_deps):
        def crashing():
            yield _chunk(content="部分内容")
            raise ConnectionError("stream reset by peer")

        with patch.object(nodes, "llm_create_with_retry",
                          return_value=(crashing(), None)):
            events = _run("ALD?")

        _, by_type = _collect(events)
        assert "error_trace" in by_type
        et = by_type["error_trace"][0]
        assert et["phase"] == "llm_stream"
        assert et["error_type"] == "ConnectionError"
        assert "stream reset" in et["message"]
        assert et["traceback_preview"]  # 真实 traceback 被捕获
        # 仍有面向用户的 error 事件 + done
        assert "error" in by_type
        assert by_type["done"][-1]["trace"]["final_reason"] == "error"

    def test_llm_create_error_emits_error_trace(self, _isolated_deps):
        err = RuntimeError("model unavailable")
        with patch.object(nodes, "llm_create_with_retry",
                          return_value=(None, err)):
            events = _run("ALD?")
        _, by_type = _collect(events)
        et = by_type["error_trace"][0]
        assert et["phase"] == "llm_create"
        assert et["error_type"] == "RuntimeError"
        assert by_type["done"][-1]["trace"]["final_reason"] == "error"


class TestOnEventCallback:
    def test_callback_receives_all_events(self, _isolated_deps):
        received = []
        with patch.object(nodes, "llm_create_with_retry",
                          return_value=(_answer_stream(), None)):
            _run(on_event=received.append)
        types = [e["type"] for e in received]
        # 关键事件都经过回调
        assert "step_start" in types
        assert "llm_response" in types
        assert "step_end" in types
        assert "done" in types
        # 回调收到的事件与 yield 的事件是同一对象(数量一致)
        assert len(received) >= 6

    def test_callback_exception_does_not_break_stream(self, _isolated_deps):
        def bad_cb(ev):
            raise RuntimeError("callback boom")

        with patch.object(nodes, "llm_create_with_retry",
                          return_value=(_answer_stream(), None)):
            events = _run(on_event=bad_cb)
        # 主流程不受影响
        assert any(e["type"] == "done" for e in events)
        assert any(e["type"] == "token" for e in events)


class TestGroundingEvent:
    def test_grounding_event_emitted_when_sources_present(self, _isolated_deps):
        tool_result = [{"chunk_id": "d__t1", "source_stem": "manual",
                        "page_start": 12, "score": 0.9, "content": "ALD..."}]
        answer = iter([_chunk(content="根据资料回答"),
                       _chunk(finish_reason="stop"), _usage_chunk()])
        streams = [_toolcall_stream(), answer]
        # grounding_check 通过(在 nodes 命名空间打桩)
        with patch.object(nodes, "llm_create_with_retry",
                          side_effect=lambda *a, **k: (streams.pop(0), None)), \
             patch.object(nodes, "dispatch", return_value=tool_result), \
             patch.object(nodes, "grounding_check",
                          return_value={"passed": True, "warnings": []}):
            events = _run("ALD?")
        _, by_type = _collect(events)
        assert "grounding" in by_type
        g = by_type["grounding"][0]
        assert g["passed"] is True
        trace = by_type["done"][-1]["trace"]
        assert trace["grounding"]["passed"] is True


class TestSSEPassthrough:
    """路由层对新事件类型透明转发(不被过滤)。"""
    def test_new_event_types_forwarded_over_sse(self, isolated_ratelimit):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        import chat.router as chat_router_mod
        from auth.deps import get_current_user

        new_types = [
            {"type": "step_start", "step": 1, "trace_id": "abc"},
            {"type": "llm_response", "step": 1, "finish_reason": "stop"},
            {"type": "tool_call", "step": 1, "name": "search_text"},
            {"type": "tool_result", "step": 1, "ok": True},
            {"type": "step_end", "step": 1, "decision": "answer"},
            {"type": "grounding", "passed": True, "warnings": []},
            {"type": "error_trace", "phase": "llm_stream",
             "error_type": "ValueError", "message": "x", "traceback_preview": ""},
            {"type": "done"},
        ]

        def mock_stream(message, history, **kwargs):
            for ev in new_types:
                yield ev

        app = FastAPI()
        app.include_router(chat_router_mod.router, prefix="/api")
        app.dependency_overrides[get_current_user] = \
            lambda: {"username": "t", "role": "user"}
        with patch.object(chat_router_mod, "react_stream", mock_stream):
            client = TestClient(app)
            resp = client.post("/api/chat", json={"message": "hi", "history": []})

        assert resp.status_code == 200
        parsed = []
        for block in resp.text.split("\n\n"):
            for line in block.strip().split("\n"):
                line = line.strip()
                if line.startswith("data:"):
                    s = line[5:].strip()
                    if s:
                        parsed.append(json.loads(s))
        received_types = [e["type"] for e in parsed]
        for expected in ["step_start", "llm_response", "tool_call",
                         "tool_result", "step_end", "grounding",
                         "error_trace", "done"]:
            assert expected in received_types, f"{expected} 未被 SSE 转发"
