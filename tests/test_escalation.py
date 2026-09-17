# -*- coding: utf-8 -*-
"""升级链 / 重做编排(服务层 react_stream)。

打桩 run_simple / run_agent_graph 与 quality_check(三态由我们直接控制),
验证(两级 simple/react):
  - simple 领域题 needs_escalation -> 升级 react,发 escalation 事件
  - react 已是最高 tier,needs_escalation 无处可升 -> 带警示放行
  - 最多升级一次
  - failed -> 同 tier 重做一次,仍失败则放行
  - escalation 事件格式正确,done 始终在最后
不跑真实图 / 不连任何外部服务。
"""
import pytest
from unittest.mock import patch, MagicMock

import chat.service as svc

# 编排类测试:走旧两段式入口(clarify + classify_complexity 分别打桩),
# ROUTER_FUSED 融合入口的接线行为在 test_fused_entry.py 单独覆盖。
@pytest.fixture(autouse=True)
def _legacy_router_entry(monkeypatch):
    import config as C
    monkeypatch.setattr(C, "ROUTER_FUSED", 0)


_DONE = {"type": "done", "trace": {}}


def _path_stub(answer="这是一条回答内容。", calls=None):
    """返回一个可作为 run_simple/run_agent_graph 的生成器工厂;记录调用次数。"""
    def _gen(message, history, **k):
        if calls is not None:
            calls.append({"message": message, "history": history, **k})
        yield {"type": "token", "delta": answer[0]}
        yield {"type": "assistant_message", "content": answer}
        yield _DONE
    return _gen


def _gate(verdicts):
    """quality_check 桩:按调用顺序返回给定 verdict 列表(最后一个循环复用)。"""
    state = {"i": 0}

    def _check(answer, context, tier=None):
        i = state["i"]
        state["i"] += 1
        v = verdicts[min(i, len(verdicts) - 1)]
        if isinstance(v, str):
            return {"verdict": v, "feedback": f"fb-{v}", "warnings": []}
        return v
    return _check


def _types(events):
    return [e["type"] for e in events]


class TestEscalation:
    def test_simple_domain_escalates_to_react(self):
        calls_s, calls_r = [], []
        with patch.object(svc, "classify_complexity",
                          return_value={"tier": "simple", "confidence": 0.9,
                                        "source": "rule"}), \
             patch.object(svc, "quality_check",
                          _gate(["needs_escalation", "passed"])), \
             patch.object(svc, "run_simple", _path_stub(calls=calls_s)), \
             patch.object(svc, "run_agent_graph", _path_stub(calls=calls_r)):
            events = list(svc.react_stream("ALD 是什么", [], thread_id="t"))

        assert len(calls_s) == 1
        assert len(calls_r) == 1   # 升级后跑 react
        esc = [e for e in events if e["type"] == "escalation"]
        assert len(esc) == 1
        assert esc[0]["from_tier"] == "simple"
        assert esc[0]["to_tier"] == "react"
        assert esc[0].get("reason")
        assert _types(events)[-1] == "done"

    def test_react_top_tier_escalation_releases_with_warning(self):
        # react 已是最高 tier:质检判 needs_escalation 也无处可升,
        # 不重跑,带警示放行。
        calls = []
        gate = MagicMock(return_value={
            "verdict": "needs_escalation", "feedback": "still bad",
            "warnings": ["答案质量存疑"]})
        with patch.object(svc, "classify_complexity",
                          return_value={"tier": "react", "confidence": 0.8,
                                        "source": "llm"}), \
             patch.object(svc, "quality_check", gate), \
             patch.object(svc, "run_agent_graph", _path_stub(calls=calls)):
            events = list(svc.react_stream("固晶机保养分步流程", [],
                                           thread_id="t"))
        assert len(calls) == 1                       # 只跑 react 一次,不升级
        assert not any(e["type"] == "escalation" for e in events)
        assert any("答案质量存疑" in e.get("message", "")
                   for e in events if e["type"] == "status")
        assert _types(events)[-1] == "done"

    def test_escalates_at_most_once(self):
        # 整条请求最多升级一次:simple->react 已用掉唯一一次升级;
        # 即便 react 仍判 needs_escalation,也无处再升,直接放行。
        calls = []
        with patch.object(svc, "classify_complexity",
                          return_value={"tier": "simple", "confidence": 0.9,
                                        "source": "rule"}), \
             patch.object(svc, "quality_check",
                          _gate(["needs_escalation", "needs_escalation",
                                 "needs_escalation"])), \
             patch.object(svc, "run_simple", _path_stub(calls=calls)), \
             patch.object(svc, "run_agent_graph", _path_stub(calls=calls)):
            events = list(svc.react_stream("ALD 是什么", [], thread_id="t"))
        # simple 1 次 + react 1 次 = 2;升级额度已用完
        assert len(calls) == 2
        assert len([e for e in events if e["type"] == "escalation"]) == 1
        assert _types(events)[-1] == "done"


class TestRedo:
    def test_failed_redo_once_then_pass(self):
        calls = []
        with patch.object(svc, "classify_complexity",
                          return_value={"tier": "react", "confidence": 0.9,
                                        "source": "rule"}), \
             patch.object(svc, "quality_check",
                          _gate(["failed", "passed"])), \
             patch.object(svc, "run_agent_graph", _path_stub(calls=calls)):
            events = list(svc.react_stream("ALD 原理", [], thread_id="t"))
        assert len(calls) == 2   # 同 tier 重做一次
        assert not any(e["type"] == "escalation" for e in events)
        # redo 先发 reflect(前端清屏)再发重做 status
        assert any(e["type"] == "reflect" for e in events)
        assert any("重新" in e.get("message", "")
                   for e in events if e["type"] == "status")
        assert _types(events)[-1] == "done"

    def test_redo_keeps_thread_id_and_clears_working_memory(self):
        """质检 failed 重做:不换 thread_id(修复下一轮召回失忆),改为清空 ReAct
        工作记忆,并显式携带 qc_feedback。"""
        calls = []
        resets = []
        with patch.object(svc, "classify_complexity",
                          return_value={"tier": "react", "confidence": 0.9,
                                        "source": "rule"}), \
             patch.object(svc, "quality_check",
                          _gate(["failed", "passed"])), \
             patch.object(svc, "run_agent_graph", _path_stub(calls=calls)), \
             patch("agent_reasoning.ReAct.support.memory_background.reset_react_memory",
                   side_effect=lambda u, t: resets.append((u, t))):
            list(svc.react_stream("ALD 原理", [], thread_id="orig-t", username="alice"))
        assert len(calls) == 2
        # 会话键不变:短期流水/事实/摘要/等待门连续,下一轮召回看得到重做轮
        assert calls[0]["thread_id"] == "orig-t"
        assert calls[1]["thread_id"] == "orig-t"
        # 重做前清空了工作记忆(按 username+thread_id 定位原 checkpoint)
        assert resets == [("alice", "orig-t")]
        # 质检反馈显式随重做轮下发(不再依赖 history 夹带)
        assert calls[1]["qc_feedback"]
        assert not calls[0].get("qc_feedback")

    def test_failed_redo_exhausted_releases(self):
        calls = []
        with patch.object(svc, "classify_complexity",
                          return_value={"tier": "react", "confidence": 0.9,
                                        "source": "rule"}), \
             patch.object(svc, "quality_check",
                          _gate(["failed", "failed", "failed"])), \
             patch.object(svc, "run_agent_graph", _path_stub(calls=calls)):
            events = list(svc.react_stream("ALD 原理", [], thread_id="t"))
        assert len(calls) == 2   # 原始 + 1 次重做,不再第三次
        assert _types(events)[-1] == "done"

    def test_error_path_skips_gate(self):
        # 路径发 error 事件(但仍发 done):不升级不重做,直接放行
        def _err_path(message, history, **k):
            yield {"type": "error", "message": "模型失败"}
            yield _DONE

        gate = MagicMock(return_value={"verdict": "passed", "warnings": []})
        with patch.object(svc, "classify_complexity",
                          return_value={"tier": "react", "confidence": 0.9,
                                        "source": "rule"}), \
             patch.object(svc, "quality_check", gate), \
             patch.object(svc, "run_agent_graph", _err_path):
            events = list(svc.react_stream("ALD", [], thread_id="t"))
        gate.assert_not_called()
        assert any(e["type"] == "error" for e in events)
        assert _types(events)[-1] == "done"
