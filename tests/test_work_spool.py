# -*- coding: utf-8 -*-
"""记忆欠账补做 spool(work_spool)单测(fakeredis,不触网)。

覆盖:
  - Redis 不可达:run_consolidation 静默跳过 + Q+A 落欠账 spool;恢复后重放补跑
    完整沉淀(升迁门 -> 事实落库);
  - 升迁门 LLM 失败降级:裸写 degraded 事实 + 判定欠账落 spool;LLM 恢复后重放
    经指纹去重原地升级 promoted(不产生重复事实);
  - replay_turn:门控不再通过(丢弃)、熔断期保留、升迁门仍挂保留;
  - resilience 编排:redis_down 跳过 / 恢复后 drain 计数。
"""
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fakeredis  # noqa: E402
import pytest  # noqa: E402
import config as C  # noqa: E402

import memories.storage.connections as CONN  # noqa: E402
WS = importlib.import_module("memories.storage.work_spool")
CSL = importlib.import_module("memories.orchestration.memory_loop.consolidate")
RS = importlib.import_module("memories.orchestration.memory_loop.resilience")
extract_module = importlib.import_module("memories.orchestration.long.extract")
facts_module = importlib.import_module("memories.storage.short.facts")

TID = "alice|t1"


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    """独立 spool 目录 + 复位熔断;时间盒放宽避免干扰。"""
    monkeypatch.setenv("MEM_SPOOL_DIR", str(tmp_path))
    monkeypatch.setenv("MEM_WORK_SPOOL_DRAIN_LIMIT", "50")
    monkeypatch.setenv("MEM_WORK_SPOOL_DRAIN_BUDGET", "10")
    monkeypatch.setenv("MEM_SPOOL_DRAIN_LIMIT", "1000")
    monkeypatch.setenv("MEM_SPOOL_DRAIN_BUDGET", "10")
    CSL.reset_breaker()
    yield
    CSL.reset_breaker()


def _fake_redis(monkeypatch):
    fake = fakeredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(facts_module, "get_redis", lambda: fake)
    return fake


def _stub_gate(monkeypatch, promoted=1):
    monkeypatch.setattr(extract_module, "consolidate_turn",
                        lambda *a, **k: {"items": 1, "promoted": promoted})


class TestRedisDownDebt:
    def test_redis_down_spools_then_replays(self, monkeypatch):
        fake = _fake_redis(monkeypatch)
        # 1) Redis 挂:沉淀跳过(ROUTE_OK 不阻塞),Q+A 落欠账 spool
        monkeypatch.setattr(CSL, "redis_ready_fast", lambda: False)
        r = CSL.run_consolidation("alice", TID, "ALD 是什么", "ALD 是原子层沉积", "answer")
        assert r["route"] == CSL.ROUTE_OK and r["fid"] is None
        assert WS.pending_count() == 1
        assert fake.zcard(f"memf:facts:{TID}") == 0

        # 2) Redis 恢复:兜底维护重放 -> 升迁门 + 事实落库,spool 清空
        monkeypatch.setattr(CSL, "redis_ready_fast", lambda: True)
        monkeypatch.setattr(CONN, "redis_ready_fast", lambda: True)
        _stub_gate(monkeypatch, promoted=1)
        stats = RS.run_resilience_maintenance()
        assert stats["work_spool"]["sent"] == 1
        assert WS.pending_count() == 0
        assert fake.zcard(f"memf:facts:{TID}") == 1

    def test_redis_still_down_skips_replay(self, monkeypatch):
        """Redis 未恢复时补做必须跳过(否则重放会被误记成已完成而丢账)。"""
        _fake_redis(monkeypatch)
        monkeypatch.setattr(CSL, "redis_ready_fast", lambda: False)
        CSL.run_consolidation("alice", TID, "q1", "a1", "answer")
        assert WS.pending_count() == 1
        stats = RS.run_resilience_maintenance()
        assert stats["work_spool"] == {"skipped": "redis_down"}
        assert WS.pending_count() == 1  # 欠账保留


class TestGateFailDebt:
    def test_degrade_bare_writes_and_spools_then_replay_upgrades(self, monkeypatch):
        fake = _fake_redis(monkeypatch)
        monkeypatch.setattr(CSL, "redis_ready_fast", lambda: True)
        monkeypatch.setattr(C, "MEM_RETRY_MAX", 0, raising=False)  # 首败即降级

        def _always_fail(*a, **k):
            raise RuntimeError("gate LLM down")
        monkeypatch.setattr(extract_module, "consolidate_turn", _always_fail)
        r = CSL.run_consolidation("alice", TID, "ALD 是什么", "ALD 是原子层沉积", "answer")
        assert r["route"] == CSL.ROUTE_DEGRADE
        # 裸写 degraded 事实 + 判定欠账已落 spool
        assert fake.zcard(f"memf:facts:{TID}") == 1
        assert WS.pending_count() == 1

        # LLM 恢复:重放补判 -> 命中指纹去重,同一事实原地升级 promoted,不重复
        _stub_gate(monkeypatch, promoted=1)
        assert CSL.replay_turn({"username": "alice", "thread_id": TID,
                                "q": "ALD 是什么", "a": "ALD 是原子层沉积"}) is True
        assert fake.zcard(f"memf:facts:{TID}") == 1  # 无重复条目
        fid = fake.zrange(f"memf:facts:{TID}", 0, -1)[0]
        h = fake.hgetall(f"memf:fact:{TID}:{fid}")
        assert h["promoted"] == "1"

    def test_replay_gate_still_fail_keeps_record(self, monkeypatch):
        _fake_redis(monkeypatch)
        monkeypatch.setattr(CSL, "redis_ready_fast", lambda: True)

        def _always_fail(*a, **k):
            raise RuntimeError("gate LLM down")
        monkeypatch.setattr(extract_module, "consolidate_turn", _always_fail)
        assert CSL.replay_turn({"username": "alice", "thread_id": TID,
                                "q": "q", "a": "a"}) is False

    def test_replay_breaker_open_keeps_record(self, monkeypatch):
        _fake_redis(monkeypatch)
        monkeypatch.setattr(CSL, "redis_ready_fast", lambda: True)
        monkeypatch.setattr(CSL, "breaker_is_open", lambda: True)
        called = {"n": 0}

        def _no_call(*a, **k):
            called["n"] += 1
            return {"items": 0, "promoted": 0}
        monkeypatch.setattr(extract_module, "consolidate_turn", _no_call)
        assert CSL.replay_turn({"username": "alice", "thread_id": TID,
                                "q": "q", "a": "a"}) is False
        assert called["n"] == 0  # 熔断期不调 LLM


class TestReplayGating:
    def test_gated_off_record_dropped(self, monkeypatch):
        _fake_redis(monkeypatch)
        monkeypatch.setattr(CSL, "redis_ready_fast", lambda: True)
        _stub_gate(monkeypatch)
        # 匿名 / 非记忆开关关闭:视为处理完,丢弃不保留
        assert CSL.replay_turn({"username": "", "thread_id": "t", "q": "q", "a": "a"}) is True

    def test_promote_disabled_writes_fact_without_llm(self, monkeypatch):
        fake = _fake_redis(monkeypatch)
        monkeypatch.setattr(CSL, "redis_ready_fast", lambda: True)
        monkeypatch.setattr(C, "MEM_PROMOTE_ENABLED", False, raising=False)
        called = {"n": 0}

        def _no_call(*a, **k):
            called["n"] += 1
            return {"items": 0, "promoted": 0}
        monkeypatch.setattr(extract_module, "consolidate_turn", _no_call)
        assert CSL.replay_turn({"username": "alice", "thread_id": TID,
                                "q": "q", "a": "a"}) is True
        assert called["n"] == 0
        assert fake.zcard(f"memf:facts:{TID}") == 1
