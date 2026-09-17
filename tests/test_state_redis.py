# -*- coding: utf-8 -*-
"""运行态外置 Redis 的专项测试(fakeredis 注入,不连真实 Redis)。

覆盖 state_store 门控下的三条链路:
  - ratelimit:Redis 模式 per-user 锁(SET NX)与全局 ZSET 槽位、token 比对删除
  - metrics:Redis 模式计数/延迟列表/哈希写入与 get_stats 输出形状
  - admin.pipeline:任务态 Redis 键 CRUD(get/_update/list 排序)

注入方式:monkeypatch state_store._client_fn 返回 fakeredis 客户端,
并让 redis_mode() 恒为 True(绕过真实实例快探)。
"""
import json
import time

import fakeredis
import pytest

from support import state_store


@pytest.fixture()
def fake_redis(monkeypatch):
    """backend=redis + fakeredis 注入:get_state_redis() 恒返回同一 fake 实例。"""
    fake = fakeredis.FakeRedis(decode_responses=True)
    monkeypatch.setenv("RUNTIME_STATE_BACKEND", "redis")
    monkeypatch.setattr(state_store, "_client_fn", lambda: fake)
    monkeypatch.setattr(state_store, "redis_mode", lambda: True)
    # 隔离:清掉可能残留的熔断状态
    monkeypatch.setattr(state_store, "_open_until", 0.0)
    monkeypatch.setattr(state_store, "_retry_until", 0.0)
    monkeypatch.setattr(state_store, "_fail_streak", 0)
    yield fake


# ---------------- 熔断状态机(closed 重试 → open 降级 → half-open 试探) ----------------
@pytest.fixture()
def breaker(monkeypatch):
    """backend=redis + 加速熔断参数 + 可编程探测结果。"""
    monkeypatch.setenv("RUNTIME_STATE_BACKEND", "redis")
    monkeypatch.setattr(state_store, "_TRIP_THRESHOLD", 3)
    monkeypatch.setattr(state_store, "_RETRY_COOLDOWN", 0.05)
    monkeypatch.setattr(state_store, "_OPEN_COOLDOWN", 0.1)
    monkeypatch.setattr(state_store, "_OK_CACHE_S", 0.05)
    for attr in ("_open_until", "_retry_until", "_fail_streak",
                 "_ok_until", "_last_fail"):
        monkeypatch.setattr(state_store, attr, 0)
    calls = {"n": 0}
    result = {"ok": True}

    def _fake_probe(cooldown=None):
        calls["n"] += 1
        return result["ok"]

    monkeypatch.setattr("memories.storage.connections.redis_ready_fast",
                        _fake_probe)
    yield {"calls": calls, "result": result}


class TestCircuitBreaker:
    def test_single_failure_quick_retry_then_heal(self, breaker):
        """CLOSED 内单次失败:进快速重试窗口走内存,窗口一过自动重试并恢复。"""
        state_store.note_fail()
        assert state_store.redis_mode() is False      # 重试窗口内走内存
        time.sleep(0.07)
        assert state_store.redis_mode() is True       # 窗口过后重试成功
        assert breaker["calls"]["n"] == 1             # 只真实探测一次
        state_store.note_fail()                       # 已自愈:计数重新开始
        time.sleep(0.07)
        assert state_store.redis_mode() is True       # 未达阈值,不跳闸

    def test_trip_open_then_half_open_heal(self, breaker):
        """连续失败达阈值 → OPEN 降级;冷却结束 HALF_OPEN 试探成功 → 自愈 CLOSED。"""
        for _ in range(3):
            state_store.note_fail()
        time.sleep(0.07)                              # 出快速重试窗口
        assert state_store.redis_mode() is False      # OPEN:探测都不发起
        assert breaker["calls"]["n"] == 0
        time.sleep(0.11)                              # OPEN 冷却结束 → HALF_OPEN
        assert state_store.redis_mode() is True       # 试探成功 → 自愈
        assert breaker["calls"]["n"] == 1
        assert state_store.redis_mode() is True       # 成功短缓存内不再探测
        assert breaker["calls"]["n"] == 1

    def test_half_open_failure_reopens(self, breaker):
        """HALF_OPEN 试探失败 → 回到 OPEN 再冷却,不恢复。"""
        for _ in range(3):
            state_store.note_fail()
        time.sleep(0.11)
        breaker["result"]["ok"] = False
        assert state_store.redis_mode() is False      # 试探失败
        assert breaker["calls"]["n"] == 1
        assert state_store.redis_mode() is False      # 重新 OPEN:直接降级
        assert breaker["calls"]["n"] == 1             # 冷却内不再发起探测

    def test_closed_probe_failure_counts_toward_trip(self, breaker):
        """CLOSED 下快探失败也计入熔断(不依赖调用方 note_fail)。"""
        breaker["result"]["ok"] = False
        for _ in range(3):
            time.sleep(0.07)                          # 跨过重试窗口再试
            assert state_store.redis_mode() is False
        time.sleep(0.07)
        # 3 次连续失败已跳闸:这次直接 OPEN 降级,不再发起探测
        assert state_store.redis_mode() is False
        assert breaker["calls"]["n"] == 3


# ---------------- ratelimit ----------------
class TestRatelimitRedis:
    def test_user_lock_nx_and_release(self, fake_redis):
        from support import ratelimit
        from fastapi import HTTPException

        tok = ratelimit.acquire_user_slot("u1")
        assert tok  # Redis 模式返回 token
        with pytest.raises(HTTPException) as e:
            ratelimit.acquire_user_slot("u1")
        assert e.value.status_code == 429

        ratelimit.release_user_slot("u1", tok)
        assert ratelimit.acquire_user_slot("u1")  # 释放后可再获取

    def test_release_wrong_token_keeps_lock(self, fake_redis):
        from support import ratelimit as rl
        from fastapi import HTTPException

        tok = rl.acquire_user_slot("u2")
        # 用错误 token 释放:锁必须仍在(防误删他人新锁)
        rl.release_user_slot("u2", "wrong-token")
        with pytest.raises(HTTPException):
            rl.acquire_user_slot("u2")
        rl.release_user_slot("u2", tok)  # 正确 token 释放成功

    def test_global_zset_slots(self, fake_redis, monkeypatch):
        import config as C
        from support import ratelimit as rl
        from fastapi import HTTPException

        monkeypatch.setattr(C, "RATE_LIMIT_GLOBAL_CONCURRENT", 2)
        monkeypatch.setattr(C, "RATE_LIMIT_QUEUE_TIMEOUT", 0.2)

        t1 = rl.acquire_global_slot("u")
        t2 = rl.acquire_global_slot("u")
        assert t1 and t2
        with pytest.raises(HTTPException) as e:
            rl.acquire_global_slot("u")
        assert e.value.status_code == 429

        rl.release_global_slot(t1)
        assert rl.acquire_global_slot("u")  # 释放一个槽位后可再进

    def test_memory_fallback_when_redis_down(self, fake_redis, monkeypatch):
        """Redis 命令失败 → note_fail 冷却 → 后续走内存回退(同用户 429 语义不变)。"""
        import threading
        import config as C
        from support import ratelimit as rl
        from fastapi import HTTPException

        class Boom:
            def set(self, *a, **k):
                raise ConnectionError("redis down")

        monkeypatch.setattr(state_store, "_client_fn", lambda: Boom())
        monkeypatch.setattr(C, "RATE_LIMIT_QUEUE_TIMEOUT", 0.1)
        monkeypatch.setattr(rl, "_global_sem", threading.Semaphore(2))
        monkeypatch.setattr(rl, "_user_locks", {})

        assert rl.acquire_user_slot("u3") is None  # 内存模式:token 为 None
        with pytest.raises(HTTPException):
            rl.acquire_user_slot("u3")
        rl.release_user_slot("u3", None)
        assert rl.acquire_user_slot("u3") is None


# ---------------- metrics ----------------
class TestMetricsRedis:
    def test_record_and_stats_shape(self, fake_redis):
        from support.metrics import metrics

        metrics.record_request("alice", 120)
        metrics.record_request("bob", 300, error=True)
        metrics.record_tool_call("search_docs", success=True,
                                 category="retrieval", duration_ms=50,
                                 cache_hit=True)
        metrics.record_tool_call("search_docs", success=False,
                                 category="retrieval", duration_ms=90,
                                 error_type="timeout")
        metrics.record_search(hit=True)
        metrics.record_search(hit=False)
        metrics.record_tokens(10, 5)
        metrics.record_internal_tokens(3, 2)
        metrics.record_escalation("simple", "react")
        metrics.record_tier_result("react", 500, tokens={"prompt": 10,
                                                         "completion": 5,
                                                         "total": 15})

        s = metrics.get_stats()
        # 形状:与内存版逐字段一致
        for k in ("uptime_seconds", "requests", "latency_ms", "tools",
                  "tool_categories", "search", "tokens", "internal_tokens",
                  "per_user", "escalations_total", "escalation_rate", "by_tier"):
            assert k in s
        assert s["requests"]["total"] == 2
        assert s["requests"]["errors"] == 1
        assert s["per_user"] == {"alice": 1, "bob": 1}
        assert s["search"] == {"hits": 1, "misses": 1, "hit_rate": 0.5}
        assert s["tokens"] == {"prompt": 10, "completion": 5, "total": 15}
        assert s["internal_tokens"] == {"prompt": 3, "completion": 2, "total": 5}
        assert s["escalations_total"] == 1

        tool = s["tools"]["search_docs"]
        assert tool["calls"] == 2
        assert tool["failures"] == 1
        assert tool["category"] == "retrieval"
        assert tool["cache_hits"] == 1
        assert tool["p95_duration_ms"] >= tool["avg_duration_ms"] >= 0

        tier = s["by_tier"]["react"]
        assert tier["requests"] == 1
        assert tier["tokens"]["total"] == 15
        # 升级记在 from_tier=simple(无请求量,不进 by_tier,与内存版口径一致)
        assert fake_redis.hget("metrics:tier:escalations", "simple") == "1"

        # 延迟列表入 Redis(限长 1000)
        assert fake_redis.llen("metrics:req:lat") == 2

    def test_fallback_on_redis_failure(self, fake_redis, monkeypatch):
        """写入时 Redis 抛错 → 冷却 + 落内存,get_stats 仍返回完整形状。"""
        from support import metrics as mmod
        from support.metrics import Metrics

        class Boom:
            def pipeline(self, *a, **k):
                raise ConnectionError("redis down")

        monkeypatch.setattr(state_store, "_client_fn", lambda: Boom())
        m = Metrics()
        m.record_request("alice", 100)
        s = m.get_stats()
        assert s["requests"]["total"] == 1
        assert s["per_user"] == {"alice": 1}


# ---------------- admin.pipeline ----------------
class TestPipelineRedis:
    def test_task_crud_and_ordering(self, fake_redis):
        from admin import pipeline as pl

        task_id = "t-1"
        task = {"task_id": task_id, "filename": "a.pdf", "status": "pending",
                "message": "排队中...", "started_at": time.time()}
        pl._save_redis(fake_redis, task)

        assert pl.get_task(task_id)["filename"] == "a.pdf"
        assert pl.get_task("missing") is None

        pl._update(task_id, "done", "完成")
        updated = pl.get_task(task_id)
        assert updated["status"] == "done"
        assert updated["message"] == "完成"

        # _update 不存在的任务:静默忽略(与内存版一致)
        pl._update("nope", "done", "x")

        # 第二个任务(started_at 更晚)→ 列表按插入序
        task2 = dict(task, task_id="t-2", started_at=time.time() + 1)
        pl._save_redis(fake_redis, task2)
        ids = [t["task_id"] for t in pl.list_tasks()]
        assert ids == ["t-1", "t-2"]

    def test_json_roundtrip_chinese(self, fake_redis):
        from admin import pipeline as pl

        task = {"task_id": "t-cn", "filename": "中文.pdf", "status": "pending",
                "message": "清洗中(可能需要几分钟)...", "started_at": time.time()}
        pl._save_redis(fake_redis, task)
        got = pl.get_task("t-cn")
        assert got == task  # ensure_ascii=False 往返无损
