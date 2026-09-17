# -*- coding: utf-8 -*-
"""prune 守护线程值班锁测试:多 worker 下仅一个进程执行清理(Redis SET NX 选主)。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fakeredis  # noqa: E402

from memories.orchestration.working import lifecycle  # noqa: E402


def test_leader_lock_acquire_renew_and_takeover(monkeypatch):
    fake = fakeredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr("memories.storage.connections.get_redis", lambda: fake)

    # A 抢到值班锁
    assert lifecycle._i_lead_prune(60_000, "token-A") is True
    # B 抢不到
    assert lifecycle._i_lead_prune(60_000, "token-B") is False
    # A 续期(GET==token 后 PEXPIRE)
    assert lifecycle._i_lead_prune(60_000, "token-A") is True
    # A 的锁过期(模拟 A 崩溃)→ B 自动接管
    fake.delete(lifecycle._LEADER_KEY)
    assert lifecycle._i_lead_prune(60_000, "token-B") is True


def test_leader_lock_redis_down_skips(monkeypatch):
    """Redis 不可用 → 不值班(无 Redis checkpoint 可清,跳过是正确语义)。"""
    def _boom():
        raise ConnectionError("redis down")

    monkeypatch.setattr("memories.storage.connections.get_redis", _boom)
    assert lifecycle._i_lead_prune(60_000, "t") is False
