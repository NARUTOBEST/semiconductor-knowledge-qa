# -*- coding: utf-8 -*-
"""工作记忆 checkpointer 三级降级测试:
RedisSaver 失败 → SqliteSaver(本地文件,跨轮续跑保留)→ InMemorySaver(保底)。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memories.storage.working import working as working_mod  # noqa: E402


class _BoomCM:
    """模拟 RedisSaver.from_conn_string 返回的上下文管理器,__enter__ 即连接失败。"""
    def __enter__(self):
        raise ConnectionError("redis 连接失败(模拟)")

    def __exit__(self, *a):
        return False


def _force_redis_failure(monkeypatch):
    import langgraph.checkpoint.redis as redis_mod
    monkeypatch.setattr(redis_mod.RedisSaver, "from_conn_string",
                        staticmethod(lambda url: _BoomCM()))


def test_working_saver_falls_back_to_sqlite_on_redis_failure(monkeypatch, tmp_path):
    """RedisSaver 失败 → 二级降级 SqliteSaver(本地文件),不抛异常。"""
    _force_redis_failure(monkeypatch)
    monkeypatch.setenv("WORKING_DEGRADE_DB", str(tmp_path / "degrade.sqlite3"))
    with working_mod.working_saver() as cp:
        assert cp.__class__.__name__ == "SqliteSaver"
    # 文件确实落盘(降级窗口的 checkpoint 持久化)
    assert (tmp_path / "degrade.sqlite3").exists()


def test_working_saver_falls_back_to_memory_when_sqlite_unavailable(monkeypatch):
    """SQLite 也不可用(依赖缺失/文件打不开)→ 保底 InMemorySaver,主流程不崩。"""
    _force_redis_failure(monkeypatch)
    import sqlite3
    monkeypatch.setattr(sqlite3, "connect",
                        lambda *a, **k: (_ for _ in ()).throw(
                            ConnectionError("sqlite 不可用(模拟)")))
    with working_mod.working_saver() as cp:
        assert "MemorySaver" in cp.__class__.__name__ \
            or "memory" in cp.__class__.__module__.lower()


def test_working_saver_uses_redis_when_available(monkeypatch):
    """配置正常时应走 RedisSaver(通过 mock from_conn_string 验证,不连真实 Redis)。"""
    entered = {}

    class _FakeSaver:
        def setup(self):
            entered["setup"] = True

        def __enter__(self):
            entered["enter"] = True
            return self

        def __exit__(self, *a):
            entered["exit"] = True

    fake_cm = _FakeSaver()

    # RedisSaver 在函数内部 import,需 patch 到它的源模块
    import langgraph.checkpoint.redis as redis_mod
    monkeypatch.setattr(redis_mod.RedisSaver, "from_conn_string",
                        staticmethod(lambda url: fake_cm))

    with working_mod.working_saver() as cp:
        assert cp is fake_cm
    assert entered.get("setup") and entered.get("exit")
