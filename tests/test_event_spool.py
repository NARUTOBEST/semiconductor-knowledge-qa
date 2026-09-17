# -*- coding: utf-8 -*-
"""短期对话流水本地 WAL + 幂等回填 + 兜底编排 单测(fakeredis,不触网)。

覆盖:
  - 在线直写成功不落 WAL(行为与原先一致);
  - Redis 故障期 append 落本地 WAL,恢复后 drain 幂等推回、WAL 清空;
  - 同一 event_id 重放两次只落一条(processing->done 幂等标记);
  - 坏行跳过、条数/时间盒有界、回填失败保留;
  - resilience 编排聚合与开关、Redis 快探不通过时 0ms 跳过。
"""
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fakeredis  # noqa: E402
import pytest  # noqa: E402

st_module = importlib.import_module("memories.storage.short.short_term")
spool_module = importlib.import_module("memories.storage.short.event_spool")
import memories.storage.connections as CONN  # noqa: E402
RS = importlib.import_module("memories.orchestration.memory_loop.resilience")
from memories.storage.short.short_term import ShortTermMemory  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_spool(monkeypatch, tmp_path):
    """每个测试独立 WAL 目录,关闭时间盒干扰。"""
    monkeypatch.setenv("MEM_SPOOL_DIR", str(tmp_path))
    monkeypatch.setenv("MEM_SPOOL_DRAIN_LIMIT", "1000")
    monkeypatch.setenv("MEM_SPOOL_DRAIN_BUDGET", "10")
    yield


class _Holder:
    """可控的 get_redis:down=True 抛错(模拟 Redis 挂),False 返回 fakeredis。"""
    def __init__(self, fake):
        self.fake = fake
        self.down = False

    def __call__(self):
        if self.down:
            raise RuntimeError("redis down (simulated)")
        return self.fake


def _memory(monkeypatch):
    fake = fakeredis.FakeRedis(decode_responses=True)
    holder = _Holder(fake)
    monkeypatch.setattr(st_module, "get_redis", holder)
    return ShortTermMemory(), fake, holder


class TestLiveAndSpool:
    def test_live_success_no_wal(self, monkeypatch):
        m, _fake, holder = _memory(monkeypatch)
        holder.down = False
        assert m.append_event("alice|t1", "user_message", {"q": "hi"}) == 1
        assert spool_module.pending_count() == 0

    def test_down_spools_then_drain_replays(self, monkeypatch):
        m, fake, holder = _memory(monkeypatch)
        # 1) Redis 挂:直写失败 -> 落本地 WAL,返回 -1,不抛
        holder.down = True
        seq = m.append_event("alice|t1", "user_message", {"content": "问题"},
                             user_id="alice")
        assert seq == -1
        assert spool_module.pending_count() == 1  # 已落本地 WAL,未丢

        # 2) Redis 恢复:drain 幂等回填,WAL 清空,事件可读
        holder.down = False
        r = m.drain_spooled()
        assert r["sent"] == 1 and r["remain"] == 0
        assert spool_module.pending_count() == 0
        rows = m.recent_dialogue("alice|t1")
        assert len(rows) == 1 and rows[0]["content"] == "问题"
        # 计数器从 1 开始(回填也走 INCR)
        assert fake.get("mem:cnt:alice|t1") == "1"

    def test_replay_same_event_idempotent(self, monkeypatch):
        m, fake, holder = _memory(monkeypatch)
        holder.down = False
        rec = {"eid": "fixed-eid-1", "tid": "alice|t2", "etype": "user_message",
               "payload": {"content": "只应有一条"}, "uid": "alice", "sid": "",
               "ts": 1.0}
        assert m.replay_record(rec) is True
        assert m.replay_record(rec) is True  # 第二次:done 幂等跳过
        rows = m.recent_dialogue("alice|t2")
        assert len(rows) == 1 and rows[0]["content"] == "只应有一条"
        assert fake.get("mem:eid:fixed-eid-1") == "done"


class TestSpoolGuards:
    def test_corrupt_line_skipped(self, monkeypatch):
        m, _fake, holder = _memory(monkeypatch)
        path = spool_module._wal_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("{not a json\n")
            f.write('{"eid":"ok1","tid":"alice|t3","etype":"user_message",'
                    '"payload":{"content":"好"},"uid":"alice","sid":"","ts":1}\n')
        holder.down = False
        r = m.drain_spooled()
        assert r["bad"] == 1 and r["sent"] == 1 and r["remain"] == 0
        assert len(m.recent_dialogue("alice|t3")) == 1

    def test_drain_respects_item_limit(self, monkeypatch):
        m, _fake, holder = _memory(monkeypatch)
        holder.down = True
        for i in range(5):
            m.append_event("alice|t4", "user_message", {"content": f"q{i}"})
        assert spool_module.pending_count() == 5
        holder.down = False
        r1 = m.drain_spooled(max_items=2)
        assert r1["processed"] == 2 and r1["sent"] == 2 and r1["remain"] == 3
        r2 = m.drain_spooled()
        assert r2["sent"] == 3 and r2["remain"] == 0
        assert len(m.recent_dialogue("alice|t4")) == 5

    def test_failed_replay_kept_in_wal(self, monkeypatch):
        m, _fake, holder = _memory(monkeypatch)
        holder.down = True
        m.append_event("alice|t5", "user_message", {"content": "待恢复"})
        # 仍然没恢复:replay 抛错 -> 条目保留
        r = m.drain_spooled()
        assert r["sent"] == 0 and r["remain"] == 1
        assert spool_module.pending_count() == 1

    def test_disabled_spool_returns_false(self, monkeypatch):
        monkeypatch.setenv("MEM_SPOOL_ENABLED", "0")
        ok = spool_module.append_record({"eid": "x"})
        assert ok is False


class TestResilienceOrchestration:
    def test_redis_down_skips_wal_without_raise(self, monkeypatch):
        monkeypatch.setattr(CONN, "redis_ready_fast", lambda: False)
        out = RS.run_resilience_maintenance()
        assert out["wal"] == {"skipped": "redis_down"}

    def test_redis_up_drains(self, monkeypatch):
        monkeypatch.setattr(CONN, "redis_ready_fast", lambda: True)
        m, _fake, holder = _memory(monkeypatch)
        holder.down = True
        m.append_event("alice|t6", "user_message", {"content": "回填我"})
        holder.down = False
        out = RS.run_resilience_maintenance()
        assert out["wal"]["sent"] == 1
        assert len(m.recent_dialogue("alice|t6")) == 1

    def test_master_switch_disables(self, monkeypatch):
        monkeypatch.setenv("MEM_RESILIENCE_ENABLED", "0")
        out = RS.run_resilience_maintenance()
        assert out["wal"] == {"skipped": "disabled"}
        assert out["backfill"] == {"skipped": "disabled"}

    def test_resilience_runs_via_memory_pipeline(self, monkeypatch):
        # 兜底维护收拢在 mem_consolidate 节点内(无独立 mem_resilience 节点):
        # 每轮记忆任务都先过兜底(不挑门控),best-effort 不外抛。
        from memories.orchestration.memory_loop.graph import build_memory_graph
        g = build_memory_graph()
        assert "mem_resilience" not in g.nodes
        assert "mem_consolidate" in g.nodes
        called = {"n": 0}
        monkeypatch.setattr(RS, "run_resilience_maintenance",
                            lambda *a, **k: called.__setitem__("n", called["n"] + 1))
        monkeypatch.setattr(CONN, "redis_ready_fast", lambda: False)
        g.invoke({"username": "", "thread_id": "anon",
                  "question": "q", "full_reply": "a", "final_reason": "answer"})
        assert called["n"] == 1  # 匿名(门控 skip)也经过兜底维护
