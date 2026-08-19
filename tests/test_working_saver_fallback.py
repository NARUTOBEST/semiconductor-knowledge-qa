# -*- coding: utf-8 -*-
"""工作记忆 checkpointer 降级测试:PG 不可用时回退 InMemorySaver,主流程不崩。"""
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memories.storage.working import working as working_mod  # noqa: E402


def test_working_saver_falls_back_to_memory_on_pg_failure(caplog):
    """PostgresSaver.from_conn_string 抛异常时,应降级到 InMemorySaver。"""
    import config as C

    # 给一个会连接失败的 URI(不可路由地址 + 极短超时),触发降级
    bad_uri = "postgresql://nobody:wrong@127.0.0.1:1/nodb?connect_timeout=1"
    caplog.set_level("WARNING", logger="agent")

    with patch.object(C, "WORKING_PG_URI", bad_uri):
        with working_mod.working_saver() as cp:
            # 降级得到的是内存 checkpointer,支持基础图编译接口
            assert hasattr(cp, "put") or hasattr(cp, "aget") \
                or cp.__class__.__name__ == "InMemorySaver"
            assert "MemorySaver" in cp.__class__.__name__ \
                or "memory" in cp.__class__.__module__.lower()

    # 降级路径有 warning 日志(记到了 agent logger 或 warnings)
    # 关键是不抛异常即通过


def test_working_saver_uses_postgres_when_available(monkeypatch):
    """配置正常时应走 PostgresSaver(通过 mock from_conn_string 验证,不连真实 PG)。"""
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

    def _fake_from_conn_string(uri):
        return fake_cm

    # PostgresSaver 在函数内部 import,需 patch 到它的源模块
    import langgraph.checkpoint.postgres as pgmod
    monkeypatch.setattr(pgmod.PostgresSaver, "from_conn_string",
                        staticmethod(lambda uri: fake_cm))

    with working_mod.working_saver() as cp:
        assert cp is fake_cm
    assert entered.get("setup") and entered.get("exit")
