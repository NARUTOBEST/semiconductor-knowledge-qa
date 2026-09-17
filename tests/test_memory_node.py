# -*- coding: utf-8 -*-
"""recall_memory【普通工具化】测试(Req2)。

recall_memory 现注册进 tools registry(Category.MEMORY),与检索三件套同走
generation→runtime→execute_tools(reflect) 管线,不再有专用图节点:
  - 模型 tool_choice=auto 发出 recall_memory,经 dispatch 调 tools.memory_tool.recall_memory;
  - handler 只取回【长期偏好/画像】(低置信按需扩量),合并成一条 ToolMessage(经韧性链回灌);
    近期对话由 build_messages_node 的确定性预取负责,本工具不再拉短期;
  - 与确定性预取(prefetch)按条目 id 去重(去重在 test_memory_prefetch.py 覆盖);
  - 长期依赖失败 -> handler 抛异常 -> 韧性链回灌显式错误 ToolMessage、reflect 标 MEMORY down。
用假 LLM 流 + monkeypatch 记忆函数,不触网、不依赖 Redis/PG。预取在本文件关闭以隔离工具路径。
"""
import os
import sys
import json
import time
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as C  # noqa: E402
from agent_reasoning.ReAct.core import nodes  # noqa: E402
from agent_reasoning.ReAct.core.graph import build_graph  # noqa: E402
from agent_reasoning.ReAct.trace import TraceRecorder  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

# 记忆函数所在模块(handler 在函数内 import,monkeypatch 模块属性即可生效)
import memories.orchestration.short.recall as recall_mod  # noqa: E402
import memories.orchestration.long.inject as inject_mod  # noqa: E402
import tools.memory_tool as mt  # noqa: E402

_MEM_PROFILE = "半导体工程师"
_MEM_FACT = "ALD 工艺工程师,偏好中文回答"


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


def _answer_stream(text="根据你的背景作答"):
    for ch in text:
        yield _Chunk(_Choice(_Delta(content=ch)))
    yield _Chunk(_Choice(_Delta(), finish_reason="stop"), usage=_usage())


def _mem_stream(tc_id="m1", query="结合我的背景推荐"):
    yield _Chunk(_Choice(_Delta(tool_calls=[
        _TCDelta(0, id=tc_id, name="recall_memory",
                 arguments=json.dumps({"query": query})),
    ]), finish_reason="tool_calls"))
    yield _Chunk(usage=_usage())


def _mem_plus_search_stream(mem_id="m1", search_id="s1"):
    yield _Chunk(_Choice(_Delta(tool_calls=[
        _TCDelta(0, id=mem_id, name="recall_memory",
                 arguments=json.dumps({"query": "我的关注领域"})),
        _TCDelta(1, id=search_id, name="search_text",
                 arguments=json.dumps({"query": "ALD"})),
    ]), finish_reason="tool_calls"))
    yield _Chunk(usage=_usage())


def _make_client(script, captured):
    it = iter(script)

    class _Completions:
        @staticmethod
        def create(**kwargs):
            captured.append(kwargs)
            return next(it)

    class _Chat:
        completions = _Completions()

    return types.SimpleNamespace(chat=_Chat())


def _run(script, *, user_id="alice", long_enabled=True, dispatch_fn=None,
         question="结合我的情况推荐工艺", recent_block="", recall_ret=None):
    recorder = TraceRecorder("tr", time.time(), question)
    cfg = {"configurable": {"thread_id": "t1", "trace_recorder": recorder}}
    if user_id is not None:
        cfg["configurable"]["user_id"] = user_id

    g = build_graph(checkpointer=InMemorySaver())
    captured: list = []
    fake = _make_client(script, captured)
    old_c, old_d = nodes.get_client, nodes.dispatch
    old_flag, old_pf = C.LONG_MEM_ENABLED, C.RECALL_PREFETCH_ENABLED
    nodes.get_client = lambda: fake
    # recall_memory 委托给真实 handler(走 stub 记忆函数 + contextvar 身份);检索走桩。
    def _default_dispatch(name, args, **kw):
        if name == mt.MEMORY_TOOL_NAME:
            return mt.recall_memory(**(args or {}))
        return []
    nodes.dispatch = dispatch_fn or _default_dispatch
    C.LONG_MEM_ENABLED = long_enabled
    C.RECALL_PREFETCH_ENABLED = False  # 隔离:本文件只测工具路径,预取在 test_memory_prefetch 覆盖

    # 记忆函数桩
    old_recent = recall_mod.recent_dialogue_block
    old_recall = inject_mod.recall_memories
    recall_mod.recent_dialogue_block = lambda *a, **k: recent_block
    if recall_ret is None:
        recall_ret = ([{"id": 101, "content": _MEM_FACT, "distance": 0.1}],
                      {"summary": _MEM_PROFILE})
    inject_mod.recall_memories = recall_ret if callable(recall_ret) else (lambda u, q: recall_ret)
    try:
        evs = list(g.stream({
            "question": question, "history": [],
            "started_at": time.time(),
            "max_steps": 6, "max_total_seconds": 60,
        }, config=cfg, stream_mode="custom"))
        state = g.get_state(cfg).values
    finally:
        nodes.get_client = old_c
        nodes.dispatch = old_d
        C.LONG_MEM_ENABLED = old_flag
        C.RECALL_PREFETCH_ENABLED = old_pf
        recall_mod.recent_dialogue_block = old_recent
        inject_mod.recall_memories = old_recall
    return evs, state, captured


def _tool_messages(state):
    return [m for m in (state.get("messages") or [])
            if getattr(m, "type", "") == "tool"]


def _bound_tool_names(captured):
    names = set()
    for kw in captured:
        for sch in (kw.get("tools") or []):
            fn = sch.get("function", {})
            if fn.get("name"):
                names.add(fn["name"])
    return names


# ---------------- 场景 ----------------

def test_memory_recall_runs_via_dispatch_as_normal_tool():
    # recall_memory 走通用 dispatch 管线:回灌记忆 ToolMessage,产生 tool_call/tool_result 事件。
    evs, state, captured = _run([_mem_stream(), _answer_stream()])

    assert any(e.get("type") == "tool_call" and e.get("name") == "recall_memory"
               for e in evs)
    assert any(e.get("type") == "tool_result" and e.get("name") == "recall_memory"
               and e.get("ok") for e in evs)
    tool_msgs = _tool_messages(state)
    body = "\n".join(m.content or "" for m in tool_msgs)
    assert _MEM_PROFILE in body and _MEM_FACT in body   # 长期偏好
    assert not state.get("collected_sources")           # 记忆不产生来源
    assert state["final_reason"] == "answer"
    # gating:已登录且启用 -> schema 下发
    assert "recall_memory" in _bound_tool_names(captured)


# 近期对话由预取通道负责;recall_memory 工具只回长期,任何情况下都不取短期。
_SHORT_TEXT = "【会话历史对话】用户:ALD 是什么?\n助手:ALD 是原子层沉积。"


def test_recall_returns_long_only():
    # recall_memory 只返回长期偏好/画像;即使短期有对话也不经工具重复注入。
    evs, state, _ = _run([_mem_stream(), _answer_stream("综合作答")],
                         recent_block=_SHORT_TEXT)
    tool_msgs = _tool_messages(state)
    assert len(tool_msgs) == 1
    body = tool_msgs[0].content or ""
    assert _MEM_PROFILE in body       # 长期画像
    assert _MEM_FACT in body          # 长期偏好
    assert _SHORT_TEXT not in body    # 短期不经工具返回


def test_recall_long_empty_returns_no_info():
    # 长期无偏好:工具回"无更多长期背景"兜底文案,且不取短期(短期归预取)。
    evs, state, _ = _run([_mem_stream(), _answer_stream("好的")],
                         recent_block=_SHORT_TEXT, recall_ret=([], None))
    body = _tool_messages(state)[0].content or ""
    assert _SHORT_TEXT not in body
    assert "长期背景" in body or "一般情况" in body


def test_memory_plus_search_same_turn():
    dispatched = []

    def fake_dispatch(name, args, **kw):
        if name == mt.MEMORY_TOOL_NAME:
            return mt.recall_memory(**(args or {}))
        dispatched.append((name, args))
        return [{"chunk_id": "c1", "source_stem": "doc", "page": "1",
                 "score": 0.9, "content": "ALD 资料"}]

    evs, state, captured = _run(
        [_mem_plus_search_stream(), _answer_stream("综合作答")],
        dispatch_fn=fake_dispatch)

    # 检索工具经过通用 dispatch(一次 search_text);记忆与检索各一条 ToolMessage
    assert [n for n, _ in dispatched] == ["search_text"]
    tool_msgs = _tool_messages(state)
    body = "\n".join(m.content or "" for m in tool_msgs)
    assert _MEM_PROFILE in body
    assert len(tool_msgs) == 2
    assert state["final_reason"] == "answer"


def test_memory_recall_failure_is_explicit_and_answering_continues():
    # 长期记忆依赖异常:handler 抛错 -> 韧性链回灌显式错误 ToolMessage(ok=false),照常作答。
    def _boom(u, q):
        raise RuntimeError("pg down")

    evs, state, _ = _run([_mem_stream(), _answer_stream("好的")], recall_ret=_boom)
    # 有一条 recall_memory 的失败 tool_result
    assert any(e.get("type") == "tool_result" and e.get("name") == "recall_memory"
               and not e.get("ok") for e in evs)
    # 主流程不炸,仍出答案
    assert state["final_reason"] == "answer"


def test_memory_schema_gating():
    # 1) 已登录 + 启用 -> 下发
    _, _, cap_on = _run([_answer_stream("hi")], user_id="alice", long_enabled=True)
    assert "recall_memory" in _bound_tool_names(cap_on)
    # 2) 匿名 -> 不下发
    _, _, cap_anon = _run([_answer_stream("hi")], user_id=None, long_enabled=True)
    assert "recall_memory" not in _bound_tool_names(cap_anon)
    # 3) 已登录但 LONG_MEM_ENABLED=0 -> 不下发
    _, _, cap_off = _run([_answer_stream("hi")], user_id="alice", long_enabled=False)
    assert "recall_memory" not in _bound_tool_names(cap_off)
