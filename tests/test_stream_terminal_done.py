# -*- coding: utf-8 -*-
"""SSE 终态契约测试:done 是唯一终端事件,任何路径失败都必须补发。

背景(审计项5):simple 路径 LLM 失败只发 error 不发 done,前端加载圈可能
永久挂起。修复分两层:
  - service.react_stream:held_done is None 时合成终态 done(final_reason=error);
  - router.gen:流走完仍未发过 done 时兜底补一条(防编排层自身异常漏发)。

本文件验证:
  - 服务层:simple 路径 LLM 创建失败 / 流读取中断 → error 后必跟 done 且唯一;
  - 服务层:error+done 的路径(path_errored)不被二次补发;
  - 路由层:react_stream 漏发 done 时,SSE 响应仍以 done 收尾。
不跑真实图 / 不调真实 LLM。
"""
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import chat.service as svc


def _tier(tier):
    return lambda m, h: {"tier": tier, "confidence": 1.0, "source": "rule"}


def _path_gen(*events):
    """返回可作 run_simple/run_agent_graph 的生成器工厂:原样吐给定事件。"""
    def _gen(message, history, **k):
        yield from events
    return _gen


def _dones(events):
    return [e for e in events if e["type"] == "done"]


class TestServiceTerminalDone:
    def test_llm_create_failure_emits_terminal_done(self):
        """simple 路径 LLM 创建失败(只发 error)→ 服务层补发终态 done。"""
        events_in = [
            {"type": "status", "message": "思考中…"},
            {"type": "error", "trace_id": "x", "step": 1,
             "message": "模型请求失败: boom"},
        ]
        gate = MagicMock(side_effect=AssertionError("路径失败不应触发质检"))
        with patch.object(svc, "classify_complexity", _tier("simple")), \
             patch.object(svc, "quality_check", gate), \
             patch.object(svc, "run_simple", _path_gen(*events_in)):
            events = list(svc.react_stream("你好", [], thread_id="t"))

        types_ = [e["type"] for e in events]
        assert "error" in types_
        assert types_[-1] == "done"
        dones = _dones(events)
        assert len(dones) == 1
        assert dones[0]["final_reason"] == "error"
        gate.assert_not_called()

    def test_stream_interrupt_emits_terminal_done(self):
        """流读取中断(token 已上屏后 error)→ 同样补发终态 done。"""
        events_in = [
            {"type": "status", "message": "思考中…"},
            {"type": "token", "delta": "你好"},
            {"type": "error", "trace_id": "x", "step": 1,
             "message": "流读取中断: boom"},
        ]
        with patch.object(svc, "classify_complexity", _tier("simple")), \
             patch.object(svc, "quality_check", MagicMock(side_effect=AssertionError("不应触发质检"))), \
             patch.object(svc, "run_simple", _path_gen(*events_in)):
            events = list(svc.react_stream("你好", [], thread_id="t"))

        assert [e["type"] for e in events][-1] == "done"
        assert len(_dones(events)) == 1

    def test_path_errored_with_done_not_duplicated(self):
        """路径 error 后自带 done(final_reason=error):放行,不二次补发。"""
        events_in = [
            {"type": "error", "trace_id": "x", "step": 1, "message": "e"},
            {"type": "done", "trace": {}, "final_reason": "error"},
        ]
        with patch.object(svc, "classify_complexity", _tier("react")), \
             patch.object(svc, "run_agent_graph", _path_gen(*events_in)):
            events = list(svc.react_stream("ALD 是什么", [], thread_id="t"))

        dones = _dones(events)
        assert len(dones) == 1
        assert [e["type"] for e in events][-1] == "done"


# ---------------- 路由层兜底 ----------------
class TestRouterFallbackDone:
    @pytest.fixture()
    def client(self):
        import chat.router as cr
        from auth.deps import get_current_user
        app = FastAPI()
        app.include_router(cr.router, prefix="/api")
        app.dependency_overrides[get_current_user] = lambda: {
            "username": "tester", "role": "user"}
        yield TestClient(app), cr
        app.dependency_overrides.clear()

    def test_missing_done_is_appended(self, client):
        """react_stream 整条流没发 done(极端漏发)→ 路由补一条,SSE 仍以 done 收尾。"""
        tc, cr = client
        # 打桩 react_stream:只发 error,不发 done(模拟编排层自身异常前的漏发)
        def _leaky(message, history, **k):
            yield {"type": "error", "message": "boom"}
        with patch.object(cr, "react_stream", _leaky), \
             patch.object(cr, "acquire_user_slot"), \
             patch.object(cr, "acquire_global_slot"), \
             patch.object(cr, "release_user_slot"), \
             patch.object(cr, "release_all"):
            r = tc.post("/api/chat", json={
                "message": "你好",
                "history": [{"role": "user", "content": "你好"}],
                "thread_id": "t1",
            })
        assert r.status_code == 200
        types_ = []
        for line in r.text.splitlines():
            if line.startswith("data:"):
                import json
                types_.append(json.loads(line[5:]).get("type"))
        assert types_[-1] == "done"
        assert types_.count("done") == 1

    def test_done_not_duplicated_when_present(self, client):
        """正常 done 已发:路由不重复补发。"""
        tc, cr = client
        def _ok(message, history, **k):
            yield {"type": "token", "delta": "你好"}
            yield {"type": "done", "trace": {}}
        with patch.object(cr, "react_stream", _ok), \
             patch.object(cr, "acquire_user_slot"), \
             patch.object(cr, "acquire_global_slot"), \
             patch.object(cr, "release_user_slot"), \
             patch.object(cr, "release_all"):
            r = tc.post("/api/chat", json={
                "message": "你好",
                "history": [{"role": "user", "content": "你好"}],
                "thread_id": "t2",
            })
        assert r.status_code == 200
        import json
        types_ = [json.loads(l[5:]).get("type")
                  for l in r.text.splitlines() if l.startswith("data:")]
        assert types_.count("done") == 1
        assert types_[-1] == "done"
