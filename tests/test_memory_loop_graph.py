# -*- coding: utf-8 -*-
"""memory-loop(记忆工作闭环)测试。

两部分:
  1) 节点一沉淀逻辑(consolidate):门控 skip、成功落事实+升迁、LLM 失败 retry→
     degrade、熔断后直接裸写、Redis 不可用静默跳过。
  2) 图级:门控(匿名 skip→done)、沉淀成功(done 在最后、事实落库)、done 携带
     retrieval 字段。用假 LLM 流 + fakeredis + InMemorySaver,不触网/不依赖 PG。
"""
import os
import sys
import time
import types
import importlib
import itertools

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fakeredis  # noqa: E402
import pytest  # noqa: E402
import config as C  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402

SS = importlib.import_module("memories.orchestration.memory_loop.session_summary")
from memories.storage.working import session_file as SF  # noqa: E402

from agent_reasoning.ReAct.core import nodes  # noqa: E402
from agent_reasoning.ReAct.core.graph import build_graph  # noqa: E402
from agent_reasoning.ReAct.trace import TraceRecorder  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

import importlib  # noqa: E402
facts_module = importlib.import_module("memories.storage.short.facts")
CSL = importlib.import_module("memories.orchestration.memory_loop.consolidate")
extract_module = importlib.import_module("memories.orchestration.long.extract")

TID = "alice|t1"


@pytest.fixture(autouse=True)
def _reset_breaker():
    CSL.reset_breaker()
    yield
    CSL.reset_breaker()


def _fake_redis(monkeypatch):
    fake = fakeredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(facts_module, "get_redis", lambda: fake)
    monkeypatch.setattr(CSL, "redis_ready_fast", lambda: True)
    return fake


# ---------------- 节点一:沉淀逻辑 ----------------
class TestConsolidation:
    def test_gate_anonymous_or_disabled_skips(self, monkeypatch):
        _fake_redis(monkeypatch)
        promoted = []
        monkeypatch.setattr(extract_module, "consolidate_turn",
                            lambda *a, **k: promoted.append(1) or {"promoted": 1})
        # 匿名
        r = CSL.run_consolidation(None, TID, "q", "a", "answer")
        assert r["route"] == CSL.ROUTE_SKIP
        # 关闭
        monkeypatch.setattr(C, "MEM_LOOP_ENABLED", False, raising=False)
        r = CSL.run_consolidation("alice", TID, "q", "a", "answer")
        assert r["route"] == CSL.ROUTE_SKIP
        monkeypatch.setattr(C, "MEM_LOOP_ENABLED", True, raising=False)
        # 非 answer
        r = CSL.run_consolidation("alice", TID, "q", "a", "max_steps")
        assert r["route"] == CSL.ROUTE_SKIP
        assert promoted == []  # 门控不通过不调升迁门

    def test_redis_down_skips_silently(self, monkeypatch):
        monkeypatch.setattr(CSL, "redis_ready_fast", lambda: False)
        r = CSL.run_consolidation("alice", TID, "q", "a", "answer")
        # Redis 不可用:不写事实、不阻塞,返回 OK 以继续节点二
        assert r["route"] == CSL.ROUTE_OK and r["fid"] is None

    def test_success_writes_fact_and_promotes(self, monkeypatch):
        fake = _fake_redis(monkeypatch)
        monkeypatch.setattr(extract_module, "consolidate_turn",
                            lambda *a, **k: {"items": 1, "promoted": 1})
        r = CSL.run_consolidation("alice", TID, "ALD 是什么", "ALD 是原子层沉积", "answer")
        assert r["route"] == CSL.ROUTE_OK and r["fid"] == "f1"
        assert fake.zcard("memf:facts:alice|t1") == 1
        h = fake.hgetall("memf:fact:alice|t1:f1")
        assert h["promoted"] == "1"  # 升迁成功已标记

    def test_duplicate_skips_gate(self, monkeypatch):
        _fake_redis(monkeypatch)
        calls = []
        monkeypatch.setattr(extract_module, "consolidate_turn",
                            lambda *a, **k: calls.append(1) or {"promoted": 0})
        CSL.run_consolidation("alice", TID, "重复问题", "重复答案", "answer")
        CSL.run_consolidation("alice", TID, "重复问题", "重复答案", "answer")
        # 第二次命中指纹去重:只 1 次升迁门,事实仍 1 条
        assert len(calls) == 1

    def test_llm_fail_then_retry_then_degrade(self, monkeypatch):
        fake = _fake_redis(monkeypatch)
        monkeypatch.setattr(C, "MEM_RETRY_MAX", 2, raising=False)
        monkeypatch.setattr(CSL.time, "sleep", lambda s: None)  # 退避不真睡

        def _always_fail(*a, **k):
            raise RuntimeError("gate LLM down")
        monkeypatch.setattr(extract_module, "consolidate_turn", _always_fail)

        r1 = CSL.run_consolidation("alice", TID, "q", "a", "answer", fail_count=0)
        assert r1["route"] == CSL.ROUTE_RETRY and r1["fail_count"] == 1
        r2 = CSL.run_retry("alice", TID, "q", "a", "answer", fail_count=1)
        assert r2["route"] == CSL.ROUTE_RETRY and r2["fail_count"] == 2
        r3 = CSL.run_retry("alice", TID, "q", "a", "answer", fail_count=2)
        # 第 3 次仍败(且熔断打开)→ 裸写降级
        assert r3["route"] == CSL.ROUTE_DEGRADE and r3["degraded"] is True
        h = fake.hgetall(f"memf:fact:alice|t1:{r3['fid']}")
        assert h["degraded"] == "1"

    def test_breaker_open_degrades_without_llm(self, monkeypatch):
        _fake_redis(monkeypatch)
        monkeypatch.setattr(C, "MEM_BREAKER_FAIL_THRESHOLD", 1, raising=False)
        calls = []

        def _always_fail(*a, **k):
            calls.append(1)
            raise RuntimeError("gate LLM down")
        monkeypatch.setattr(extract_module, "consolidate_turn", _always_fail)
        # 第一次失败即触发熔断(阈值 1)并降级
        CSL.run_consolidation("alice", TID, "q1", "a1", "answer")
        assert CSL.breaker_is_open()
        calls.clear()
        # 熔断期:新会话直接降级,不再调 LLM
        r = CSL.run_consolidation("alice", "alice|t2", "q2", "a2", "answer")
        assert r["route"] == CSL.ROUTE_DEGRADE
        assert calls == []


# ---------------- 节点二:会话摘要与 Auto-Compact ----------------
def _turn(user: str, assistant: str):
    # 逻辑级调用不经 add_messages(不会自动赋 id),显式给 id 以测 RemoveMessage/游标
    import itertools
    n = next(_turn._n)
    return [HumanMessage(content=user, id=f"h{n}"),
            AIMessage(content=assistant, id=f"a{n}")]
_turn._n = itertools.count(1)


class TestSessionSummary:
    def test_below_first_threshold_no_file(self, monkeypatch, tmp_path):
        monkeypatch.setattr(C, "MEM_SESSION_DIR", str(tmp_path), raising=False)
        msgs = _turn("短问题", "短答案")
        r = SS.run_session_maintenance("alice", "alice|t1", msgs)
        assert r["summarized"] is False and r["compacted"] is False
        assert not SF.SessionFile("alice", "alice|t1").exists()

    def test_first_summary_uses_subagent_llm(self, monkeypatch, tmp_path):
        monkeypatch.setattr(C, "MEM_SESSION_DIR", str(tmp_path), raising=False)
        called = {"n": 0}

        def _sub(transcript, prev, deadline=None):
            called["n"] += 1
            return "## 1. 会话目标\n首摘目标\n## 2. 已达成结论 / 决策\n无"

        monkeypatch.setattr(SS, "_llm_subagent", _sub)
        monkeypatch.setattr(SS, "_llm_direct",
                            lambda t, p, deadline=None: (_ for _ in ()).throw(AssertionError("不应降级到 direct")))
        # 构造 > 1万 tokens 的多轮对话(每条足够长)
        big = "工艺" * 400  # ~800 tokens/条
        msgs = []
        for i in range(8):
            msgs += _turn(f"问题{i}:{big}", f"答案{i}:{big}")
        r = SS.run_session_maintenance("alice", "alice|t1", msgs)
        assert r["summarized"] is True and called["n"] == 1
        f = SF.SessionFile("alice", "alice|t1")
        assert f.exists()
        meta, body = f.read()
        assert "首摘目标" in body and meta["cursor_msg_id"]

    def test_three_level_degradation_to_rule(self, monkeypatch, tmp_path):
        monkeypatch.setattr(C, "MEM_SESSION_DIR", str(tmp_path), raising=False)
        monkeypatch.setattr(SS, "_llm_subagent", lambda t, p, deadline=None: None)   # 级别1 失败
        monkeypatch.setattr(SS, "_llm_direct", lambda t, p, deadline=None: None)     # 级别2 失败
        big = "内容" * 400
        msgs = []
        for i in range(8):
            msgs += _turn(f"q{i}:{big}", f"a{i}:{big}")
        r = SS.run_session_maintenance("alice", "alice|t1", msgs)
        assert r["summarized"] is True and r["level"] == "rule"
        _, body = SF.SessionFile("alice", "alice|t1").read()
        assert "规则降级" in body  # 规则截断附录
        # 规则级别不 compact(不删消息)
        assert r["remove"] == [] and r["compacted"] is False

    def test_compact_removes_old_and_sets_summary(self, monkeypatch, tmp_path):
        monkeypatch.setattr(C, "MEM_SESSION_DIR", str(tmp_path), raising=False)
        monkeypatch.setattr(SS, "_llm_subagent",
                            lambda t, p, deadline=None: "## 1. 会话目标\n压缩摘要SUMMARY")
        # 把 compact 触发线压低,保留窗口收窄,使少量消息即可触发
        monkeypatch.setattr(C, "MEM_MODEL_CONTEXT_TOKENS", 4000, raising=False)
        monkeypatch.setattr(C, "MEM_COMPACT_RESERVE_MIN", 200, raising=False)
        monkeypatch.setattr(C, "MEM_COMPACT_RESERVE_RATIO", 0.0, raising=False)
        monkeypatch.setattr(C, "MEM_KEEP_RECENT_DEFAULT", 1500, raising=False)
        monkeypatch.setattr(C, "MEM_KEEP_RECENT_MIN", 800, raising=False)
        monkeypatch.setattr(C, "MEM_MIN_TEXT_MESSAGES", 2, raising=False)
        monkeypatch.setattr(C, "MEM_SUMMARY_FIRST_TOKENS", 500, raising=False)
        big = "资料" * 300  # ~600 tokens/条,一轮 ~1200
        msgs = []
        for i in range(8):
            msgs += _turn(f"问题{i}{big}", f"答案{i}{big}")
        ids_before = [m.id for m in msgs]
        r = SS.run_session_maintenance("alice", "alice|t1", msgs)
        assert r["compacted"] is True and r["remove"], "应触发 compact"
        removed_ids = {rm.id for rm in r["remove"]}
        # 删掉的是旧轮,保留区最后一轮(最新)不被删
        assert ids_before[-1] not in removed_ids
        assert removed_ids.issubset(set(ids_before))
        assert "压缩摘要SUMMARY" in r["summary"]
        # 边界落在 HumanMessage:被删集合的最后一条之后紧跟保留区起点是 Human
        kept = [m for m in msgs if m.id not in removed_ids]
        assert isinstance(kept[0], HumanMessage)


# ---------------- 图级 ----------------
class _Delta:
    def __init__(self, content=None):
        self.content = content
        self.tool_calls = None


class _Choice:
    def __init__(self, delta, finish_reason=None):
        self.delta = delta
        self.finish_reason = finish_reason


class _Chunk:
    def __init__(self, choice=None, usage=None):
        self.choices = [choice] if choice is not None else []
        self.usage = usage


def _answer_stream(text="你好世界"):
    for ch in text:
        yield _Chunk(_Choice(_Delta(content=ch)))
    yield _Chunk(_Choice(_Delta(), finish_reason="stop"),
                 usage=types.SimpleNamespace(prompt_tokens=10, completion_tokens=20,
                                             total_tokens=30))


def _make_client(script):
    it = iter(script)

    class _Completions:
        @staticmethod
        def create(**kwargs):
            return next(it)

    return types.SimpleNamespace(chat=types.SimpleNamespace(completions=_Completions()))


def _run_graph(question="你好", *, user_id="alice", thread_id="t1"):
    # 生产 runner 用 scoped_thread_id(username|tid) 作 configurable.thread_id
    store_tid = f"{user_id}|{thread_id}" if user_id else thread_id
    recorder = TraceRecorder("tr", time.time(), question)
    cfg = {"configurable": {"thread_id": store_tid, "trace_recorder": recorder}}
    if user_id is not None:
        cfg["configurable"]["user_id"] = user_id
    g = build_graph(checkpointer=InMemorySaver())
    fake_client = _make_client([_answer_stream()])
    old_c, old_d = nodes.get_client, nodes.dispatch
    nodes.get_client = lambda: fake_client
    nodes.dispatch = lambda n, a: []
    try:
        evs = list(g.stream({
            "question": question, "history": [],
            "started_at": time.time(), "max_steps": 6, "max_total_seconds": 60,
        }, config=cfg, stream_mode="custom"))
        state = g.get_state(cfg).values
    finally:
        nodes.get_client = old_c
        nodes.dispatch = old_d
    return evs, state, g, cfg


class TestGraphMemoryLoop:
    def test_anonymous_skips_to_done(self, monkeypatch):
        # 匿名:不进 memory-loop,直接 emit_done;done 仍是最后一帧
        evs, state, _, _ = _run_graph(user_id=None)
        types = [e["type"] for e in evs]
        assert types[-1] == "done"
        assert state["final_reason"] == "answer"

    def test_consolidation_runs_and_done_last(self, monkeypatch, tmp_path):
        fake = _fake_redis(monkeypatch)
        monkeypatch.setattr(C, "MEM_SESSION_DIR", str(tmp_path), raising=False)
        monkeypatch.setattr(extract_module, "consolidate_turn",
                            lambda *a, **k: {"items": 1, "promoted": 1})
        evs, state, _, _ = _run_graph(question="ALD 是什么")
        types = [e["type"] for e in evs]
        # done 在最后且携带检索字段
        assert types[-1] == "done"
        done = evs[-1]
        assert "retrieval_max_score" in done and "search_count" in done
        # 记忆已迁出主图:图内不再写事实表(由后台管道负责,见 test_memory_pipeline.py)
        assert fake.zcard("memf:facts:alice|t1") == 0
        # usage 不被子图/节点链翻倍
        assert state["usage"]["total_tokens"] == 30

    def test_next_turn_sees_state(self, monkeypatch, tmp_path):
        # 同一 thread 连续两轮:第二轮图正常续跑(消息累积),done 各自在最后
        _fake_redis(monkeypatch)
        monkeypatch.setattr(C, "MEM_SESSION_DIR", str(tmp_path), raising=False)
        monkeypatch.setattr(extract_module, "consolidate_turn",
                            lambda *a, **k: {"promoted": 0})
        recorder = TraceRecorder("tr", time.time(), "q1")
        cfg = {"configurable": {"thread_id": "alice|t1", "user_id": "alice",
                                "trace_recorder": recorder}}
        g = build_graph(checkpointer=InMemorySaver())
        old_c, old_d = nodes.get_client, nodes.dispatch
        nodes.dispatch = lambda n, a: []
        try:
            for q in ["第一个问题", "第二个问题"]:
                nodes.get_client = lambda: _make_client([_answer_stream("答")])
                evs = list(g.stream({
                    "question": q, "history": [],
                    "started_at": time.time(), "max_steps": 6, "max_total_seconds": 60,
                }, config=cfg, stream_mode="custom"))
                assert evs[-1]["type"] == "done"
            state = g.get_state(cfg).values
        finally:
            nodes.get_client = old_c
            nodes.dispatch = old_d
        # state 里有两轮 human 消息(短期事实写入由后台管道负责,不在图内)
        humans = [m for m in state["messages"] if getattr(m, "type", "") == "human"]
        assert len(humans) == 2
