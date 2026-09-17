# -*- coding: utf-8 -*-
"""独立记忆图 + 后台管道测试。

覆盖:
  - 独立记忆图:门控 skip(匿名/非 answer)只走 resilience;正常轮跑沉淀+摘要;
  - 管道:submit 非阻塞、同会话 FIFO 串行、wait_idle 等待/超时放行;
  - compact 剪除消息经 compact_applier 落地(模拟 checkpoint 更新);
  - 主图:finalize → emit_done,done 最后一帧,记忆不再在图内执行。

不触网:fakeredis + LLM/摘要桩 + resilience 桩。
"""
import importlib
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fakeredis  # noqa: E402
import pytest  # noqa: E402
import config as C  # noqa: E402
from langchain_core.messages import RemoveMessage  # noqa: E402

MG = importlib.import_module("memories.orchestration.memory_loop.graph")
PL = importlib.import_module("memories.orchestration.memory_loop.pipeline")
ML = importlib.import_module("memories.orchestration.memory_loop")
CSL = importlib.import_module("memories.orchestration.memory_loop.consolidate")
SS = importlib.import_module("memories.orchestration.memory_loop.session_summary")
RS = importlib.import_module("memories.orchestration.memory_loop.resilience")
extract_module = importlib.import_module("memories.orchestration.long.extract")
facts_module = importlib.import_module("memories.storage.short.facts")


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    CSL.reset_breaker()
    PL.reset_pipeline()
    yield
    CSL.reset_breaker()
    PL.reset_pipeline()


def _fake_redis(monkeypatch):
    fake = fakeredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(facts_module, "get_redis", lambda: fake)
    monkeypatch.setattr(CSL, "redis_ready_fast", lambda: True)
    return fake


def _stub_llm(monkeypatch, promoted=0):
    monkeypatch.setattr(extract_module, "consolidate_turn",
                        lambda *a, **k: {"items": 1, "promoted": promoted})
    # 摘要阈值不满足时 SS 内部零成本早退,不会调 _llm_*;为防意外触发,也打桩
    monkeypatch.setattr(SS, "_llm_subagent", lambda t, p, deadline=None: None)
    monkeypatch.setattr(SS, "_llm_direct", lambda t, p, deadline=None: None)


def _job(username="alice", thread_id="alice|t1", **kw):
    return ML.MemoryJob(username=username, thread_id=thread_id,
                        question="ALD 是什么", answer="ALD 是原子层沉积",
                        final_reason="answer", **kw)


# ---------------- 独立记忆图 ----------------
class TestMemoryGraph:
    def test_work_route_writes_facts(self, monkeypatch):
        fake = _fake_redis(monkeypatch)
        _stub_llm(monkeypatch, promoted=1)
        g = MG.build_memory_graph()
        final = g.invoke({"username": "alice", "thread_id": "alice|t1",
                          "question": "ALD 是什么", "full_reply": "ALD 是原子层沉积",
                          "final_reason": "answer"})
        assert fake.zcard("memf:facts:alice|t1") == 1
        h = fake.hgetall("memf:fact:alice|t1:" + (final.get("mem_fact_id") or "f1"))
        assert h.get("promoted") == "1"

    def test_skip_route_anonymous_goes_resilience_only(self, monkeypatch):
        fake = _fake_redis(monkeypatch)
        _stub_llm(monkeypatch)
        called = {"n": 0}
        monkeypatch.setattr(RS, "run_resilience_maintenance",
                            lambda *a, **k: called.__setitem__("n", called["n"] + 1))
        g = MG.build_memory_graph()
        g.invoke({"username": "", "thread_id": "anon",
                  "question": "q", "full_reply": "a",
                  "final_reason": "answer"})
        # 门控 skip:没有沉淀(无事实写入),只过兜底维护
        assert fake.zcard("memf:facts:anon") == 0
        assert called["n"] == 1

    def test_summary_node_passes_remove_to_state(self, monkeypatch):
        _fake_redis(monkeypatch)
        _stub_llm(monkeypatch)
        monkeypatch.setattr(SS, "run_session_maintenance",
                            lambda *a, **k: {"route": "done", "summarized": True,
                                             "compacted": True, "level": "subagent",
                                             "remove": [RemoveMessage(id="m1")],
                                             "summary": "s"})
        g = MG.build_memory_graph()
        final = g.invoke({"username": "alice", "thread_id": "alice|t1",
                          "question": "q", "full_reply": "a",
                          "final_reason": "answer"})
        assert [m.id for m in final.get("remove_messages") or []] == ["m1"]


# ---------------- 后台管道 ----------------
class TestPipeline:
    def test_submit_async_and_wait_idle(self, monkeypatch):
        fake = _fake_redis(monkeypatch)
        _stub_llm(monkeypatch)
        p = PL.get_pipeline()
        p.submit(_job())
        assert p.pending_count("alice", "alice|t1") >= 1
        assert p.wait_idle("alice", "alice|t1", timeout=5.0) is True
        assert fake.zcard("memf:facts:alice|t1") == 1
        assert p.pending_count("alice", "alice|t1") == 0

    def test_fifo_serial_order_same_conversation(self, monkeypatch):
        _fake_redis(monkeypatch)
        _stub_llm(monkeypatch)
        order = []
        real_exec = PL._execute

        def _spy(job):
            order.append(job.question)
            return real_exec(job)
        monkeypatch.setattr(PL, "_execute", _spy)
        p = PL.get_pipeline()
        for q in ["q1", "q2", "q3"]:
            p.submit(ML.MemoryJob(username="alice", thread_id="t1",
                                  question=q, answer="a", final_reason="answer"))
        assert p.wait_idle("alice", "t1", timeout=5.0) is True
        assert order == ["q1", "q2", "q3"]

    def test_wait_idle_timeout_releases(self, monkeypatch):
        _fake_redis(monkeypatch)
        _stub_llm(monkeypatch)
        p = PL.get_pipeline()
        blocker = __import__("threading").Event()
        # 执行期挂起:替换 _execute 为阻塞版,制造"上一轮还没跑完"的场景
        monkeypatch.setattr(PL, "_execute", lambda job: blocker.wait(timeout=2.0))
        p.submit(_job())
        time.sleep(0.2)  # 让 worker 取走任务进入执行
        assert p.wait_idle("alice", "alice|t1", timeout=0.3) is False  # 超时放行
        blocker.set()
        assert p.wait_idle("alice", "alice|t1", timeout=2.0) is True   # 完成后空闲

    def test_compact_applier_receives_removes(self, monkeypatch):
        _fake_redis(monkeypatch)
        _stub_llm(monkeypatch)
        monkeypatch.setattr(SS, "run_session_maintenance",
                            lambda *a, **k: {"route": "done", "summarized": True,
                                             "compacted": True, "level": "subagent",
                                             "remove": [RemoveMessage(id="m1"),
                                                        RemoveMessage(id="m2")],
                                             "summary": "s"})
        got = []
        p = PL.get_pipeline()
        p.submit(_job(compact_applier=lambda tid, rms: got.append((tid, [m.id for m in rms]))))
        assert p.wait_idle("alice", "alice|t1", timeout=5.0) is True
        assert got == [("alice|t1", ["m1", "m2"])]

    def test_anonymous_job_runs_resilience_only(self, monkeypatch):
        _fake_redis(monkeypatch)
        _stub_llm(monkeypatch)
        called = {"n": 0}
        monkeypatch.setattr(RS, "run_resilience_maintenance",
                            lambda *a, **k: called.__setitem__("n", called["n"] + 1))
        p = PL.get_pipeline()
        p.submit(_job(username="", thread_id="anon"))
        assert p.wait_idle("", "anon", timeout=5.0) is True
        assert called["n"] == 1


# ---------------- 主图:done 即完即关,记忆不在图内 ----------------
class TestMainGraphSlimmed:
    def test_main_graph_has_no_memory_nodes(self):
        from agent_reasoning.ReAct.core.graph import build_graph
        g = build_graph()
        for node in ("mem_consolidate", "mem_summary", "mem_resilience",
                     "mem_consolidate_retry", "mem_consolidate_degrade"):
            assert node not in g.nodes
        assert "emit_done" in g.nodes

    def test_compaction_graph_applies_remove(self):
        """压缩图:update_state 把 RemoveMessage 应用到 InMemorySaver checkpoint。"""
        from agent_reasoning.ReAct.core.graph import build_compaction_graph
        from langchain_core.messages import AIMessage, HumanMessage
        from langgraph.checkpoint.memory import InMemorySaver
        cp = InMemorySaver()
        cfg = {"configurable": {"thread_id": "t1"}}
        seed = build_compaction_graph(checkpointer=cp)
        seed.update_state(cfg, {"messages": [
            HumanMessage(content="q", id="h1"),
            AIMessage(content="a", id="a1")]})
        state = seed.get_state(cfg).values
        assert {m.id for m in state["messages"]} == {"h1", "a1"}
        # 删除一条
        seed.update_state(cfg, {"messages": [RemoveMessage(id="a1")]})
        state = seed.get_state(cfg).values
        assert {m.id for m in state["messages"]} == {"h1"}

    def test_reset_react_memory_wipes_checkpoint_messages(self, monkeypatch):
        """质检重做前置:reset_react_memory 清空原会话 checkpoint 的 messages
        (会话键不动,重做轮冷启动由短期流水重建上下文)。"""
        import contextlib
        import types
        from agent_reasoning.ReAct.support import memory_background as MB
        import memories.storage.working as WK
        from agent_reasoning.ReAct.core.graph import build_compaction_graph
        from langchain_core.messages import AIMessage, HumanMessage
        from langgraph.checkpoint.memory import InMemorySaver
        cp = InMemorySaver()
        store_tid = "alice|t1"
        cfg = {"configurable": {"thread_id": store_tid}}
        seed = build_compaction_graph(checkpointer=cp)
        seed.update_state(cfg, {"messages": [
            HumanMessage(content="q", id="h1"),
            AIMessage(content="a", id="a1")]})

        @contextlib.contextmanager
        def _fake_saver():
            yield cp
        monkeypatch.setattr(WK, "working_saver", _fake_saver)

        MB.reset_react_memory("alice", "t1")
        assert seed.get_state(cfg).values.get("messages") == []
