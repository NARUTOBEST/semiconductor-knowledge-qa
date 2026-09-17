# -*- coding: utf-8 -*-
"""工具调用错误处理节点测试(三阶段决策节点 + 轻量 schema 校验)。

覆盖:
  - tool_validate.validate_arguments:类型转换/enum/必填/长度/未知参数;
  - validate_generation:工具名幻觉 -> 不执行、回灌错误;
  - validate_runtime:schema 参数校验失败 -> 不执行 handler;未知参数被剔除。
机械重试/熔断见 test_circuit_breaker.py;图级故障降级见 test_tool_health_adapt.py。
"""
import os
import sys
import time
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402

from tools import registry  # noqa: E402
from agent_reasoning.ReAct.core import nodes  # noqa: E402
from agent_reasoning.ReAct.core.loop import react_loop  # noqa: E402
from agent_reasoning.ReAct.support.tool_validate import validate_arguments  # noqa: E402
from agent_reasoning.ReAct.trace import TraceRecorder  # noqa: E402


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


def _usage():
    return types.SimpleNamespace(prompt_tokens=10, completion_tokens=20, total_tokens=30)


def _answer_stream(text="答案"):
    for ch in text:
        yield _Chunk(_Choice(_Delta(content=ch)))
    yield _Chunk(_Choice(_Delta(), finish_reason="stop"), usage=_usage())


def _tool_stream(tc_id="call_1", name="search_text", args='{"query": "ALD"}'):
    yield _Chunk(_Choice(_Delta(tool_calls=[
        _TCDelta(0, id=tc_id, name=name, arguments=args),
    ]), finish_reason="tool_calls"))
    yield _Chunk(usage=_usage())


def _make_client(script):
    it = iter(script)

    class _Completions:
        @staticmethod
        def create(**kwargs):
            return next(it)

    return types.SimpleNamespace(chat=types.SimpleNamespace(completions=_Completions()))


def _run(monkeypatch, client_script, dispatch_fn):
    fake = _make_client(client_script)
    monkeypatch.setattr(nodes, "get_client", lambda: fake)
    monkeypatch.setattr(nodes, "dispatch", dispatch_fn)
    recorder = TraceRecorder("tr", time.time(), "q")
    cfg = {"thread_id": "t1", "user_id": "alice", "trace_recorder": recorder}
    gen = react_loop(
        [SystemMessage(content="sys", id="p"), HumanMessage(content="你好")],
        max_steps=6, max_total_seconds=60, configurable=cfg,
        question="q", trace_id="tr",
    )
    evs, final = [], None
    try:
        while True:
            evs.append(next(gen))
    except StopIteration as e:
        final = e.value
    return evs, final


# ---------------- 轻量 schema 校验单测 ----------------
def test_validate_arguments_coerces_and_filters():
    spec = registry.get("search_text")
    # k 数字串 -> 转 int;未知参数 bogus 被识别并剔除
    clean, errors, unknown = validate_arguments(spec, {"query": "ALD", "k": "3", "bogus": 1})
    assert errors == []
    assert clean == {"query": "ALD", "k": 3}
    assert unknown == ["bogus"]


def test_validate_arguments_type_error():
    spec = registry.get("search_text")
    clean, errors, unknown = validate_arguments(spec, {"query": "ALD", "k": "abc"})
    assert clean == {"query": "ALD"}
    assert any("k" in e and "整数" in e for e in errors)


def test_validate_arguments_missing_required():
    spec = registry.get("search_text")
    clean, errors, unknown = validate_arguments(spec, {"k": 3})
    assert any("query" in e for e in errors)
    assert "query" not in clean


def test_validate_arguments_too_long():
    spec = registry.get("search_text")
    clean, errors, unknown = validate_arguments(spec, {"query": "x" * 501})
    assert any("长度" in e for e in errors)


# ---------------- 图级:工具名幻觉不执行 ----------------
def test_hallucinated_tool_name_not_executed(monkeypatch):
    calls = []

    def dispatch_fn(name, args):
        calls.append(name)
        return [{"chunk_id": "c1", "score": 0.9}]

    evs, final = _run(
        monkeypatch,
        [_tool_stream(tc_id="c1", name="search_xyz", args='{"query": "x"}'),
         _answer_stream("通用答案")],
        dispatch_fn,
    )
    # 幻觉工具根本没进 dispatch
    assert calls == []
    # 有一条失败的 tool_result:error_type=unknown_tool,name=幻觉名,并附可用清单
    failed = [e for e in evs if e.get("type") == "tool_result" and not e.get("ok")]
    assert failed
    assert failed[0].get("error_type") == "unknown_tool"
    assert failed[0].get("name") == "search_xyz"
    assert "可用工具" in failed[0].get("error", "")
    # 图正常走完出答案
    assert final["final_reason"] == "answer"


# ---------------- 图级:schema 参数校验失败不执行 handler ----------------
def test_schema_violation_not_executed(monkeypatch):
    calls = []

    def dispatch_fn(name, args):
        calls.append(args)
        return [{"chunk_id": "c1", "score": 0.9}]

    evs, final = _run(
        monkeypatch,
        [_tool_stream(tc_id="c1", name="search_text",
                      args='{"query": "ALD", "k": "abc"}'),
         _answer_stream("修正后答案")],
        dispatch_fn,
    )
    # k="abc" 校验失败 -> 不执行
    assert calls == []
    failed = [e for e in evs if e.get("type") == "tool_result" and not e.get("ok")]
    assert failed and "schema" in (failed[0].get("error_type") or "")
    assert final["final_reason"] == "answer"


# ---------------- 图级:合法调用执行,且未知参数被剔除 ----------------
def test_valid_call_executes_with_clean_args(monkeypatch):
    calls = []

    def dispatch_fn(name, args):
        calls.append(args)
        return [{"chunk_id": "c1", "source_stem": "doc", "page": "1",
                 "score": 0.9, "content": "ALD..."}]

    evs, final = _run(
        monkeypatch,
        [_tool_stream(tc_id="c1", name="search_text",
                      args='{"query": "ALD", "bogus": 5}'),
         _answer_stream("ALD 答案")],
        dispatch_fn,
    )
    # 执行了,且 handler 只收到声明过的参数
    assert len(calls) == 1
    assert calls[0] == {"query": "ALD"}
    assert final["final_reason"] == "answer"


# ---------------- 图级:reflect 空结果 -> 换词 hint ----------------
def test_empty_result_triggers_reflect_hint(monkeypatch):
    def dispatch_fn(name, args):
        return []  # 检索成功但无命中

    evs, final = _run(
        monkeypatch,
        [_tool_stream(tc_id="c1", name="search_text", args='{"query": "xyz"}'),
         _answer_stream("无资料答案")],
        dispatch_fn,
    )
    # reflect 注入"未检索到资料/换关键词"提示
    assert any(e.get("type") == "status" and "未检索到资料" in e.get("message", "")
               for e in evs)
    assert final["final_reason"] == "answer"


# ---------------- 图级:持续机械失败 -> reflect 标 down ----------------
def test_persistent_mechanical_failure_marks_down(monkeypatch):
    import httpx

    def dispatch_fn(name, args):
        raise httpx.ReadTimeout("down")  # 中间件重试耗尽后上交 timeout

    evs, final = _run(
        monkeypatch,
        [_tool_stream(tc_id="c1", name="search_text", args='{"query": "ALD"}'),
         _answer_stream("降级答案")],
        dispatch_fn,
    )
    # 失败的 tool_result(timeout)+ reflect 切换备用策略
    failed = [e for e in evs if e.get("type") == "tool_result" and not e.get("ok")]
    assert failed and failed[0].get("error_type") == "timeout"
    assert any("备用策略" in e.get("message", "")
               for e in evs if e.get("type") == "status")
    assert final["final_reason"] == "answer"
