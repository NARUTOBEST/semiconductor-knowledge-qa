# -*- coding: utf-8 -*-
"""阶段 4.6:升级链 / 重做编排(服务层 react_stream)。

打桩 run_simple / run_agent_graph 与 quality_check(三态由我们直接控制),
验证:
  - simple 领域题 needs_escalation -> 升级 medium,发 escalation 事件
  - medium 多子问题 needs_escalation -> 升级 complex
  - 最多升级一次;complex 无法再升,带警示放行
  - failed -> 同 tier 重做一次,仍失败则放行
  - escalation 事件格式正确,done 始终在最后
不跑真实图 / 不连任何外部服务。
"""
from unittest.mock import patch, MagicMock

import chat.service as svc

_DONE = {"type": "done", "trace": {"grounding": None, "coverage": None}}


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
    def test_simple_domain_escalates_to_medium(self):
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
        assert len(calls_r) == 1   # 升级后跑 medium
        esc = [e for e in events if e["type"] == "escalation"]
        assert len(esc) == 1
        assert esc[0]["from_tier"] == "simple"
        assert esc[0]["to_tier"] == "medium"
        assert esc[0].get("reason")
        assert _types(events)[-1] == "done"

    def test_medium_multiquestion_escalates_to_complex(self):
        calls = []
        with patch.object(svc, "classify_complexity",
                          return_value={"tier": "medium", "confidence": 0.8,
                                        "source": "llm"}), \
             patch.object(svc, "quality_check",
                          _gate(["needs_escalation", "passed"])), \
             patch.object(svc, "run_agent_graph", _path_stub(calls=calls)), \
             patch.object(svc, "run_simple", _path_stub()):
            events = list(svc.react_stream("对比 ALD 和 CVD 的优缺点", [],
                                           thread_id="t"))
        assert len(calls) == 2   # medium 一次 + complex 一次
        esc = [e for e in events if e["type"] == "escalation"]
        assert esc[0]["from_tier"] == "medium"
        assert esc[0]["to_tier"] == "complex"

    def test_escalates_at_most_once(self):
        # 整条请求最多升级一次:simple->medium 已用掉唯一一次升级;
        # 即便 medium 仍判 needs_escalation,也不能再升到 complex,直接放行。
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
        # simple 1 次 + medium 1 次 = 2;不会升到 complex
        assert len(calls) == 2
        assert len([e for e in events if e["type"] == "escalation"]) == 1
        assert _types(events)[-1] == "done"

    def test_complex_cannot_escalate_releases_with_warning(self):
        calls = []
        gate = MagicMock(return_value={
            "verdict": "needs_escalation", "feedback": "still bad",
            "warnings": ["complex 质检未过"]})
        with patch.object(svc, "classify_complexity",
                          return_value={"tier": "complex", "confidence": 0.9,
                                        "source": "rule"}), \
             patch.object(svc, "quality_check", gate), \
             patch.object(svc, "run_agent_graph", _path_stub(calls=calls)):
            events = list(svc.react_stream("综合分析 ALD 工艺", [],
                                           thread_id="t"))
        assert len(calls) == 1   # 没有升级,也没有重做(verdict 不是 failed)
        assert not any(e["type"] == "escalation" for e in events)
        # 警示以 status 事件可见
        assert any("complex 质检未过" in e.get("message", "")
                   for e in events if e["type"] == "status")
        assert _types(events)[-1] == "done"


class TestRedo:
    def test_failed_redo_once_then_pass(self):
        calls = []
        with patch.object(svc, "classify_complexity",
                          return_value={"tier": "medium", "confidence": 0.9,
                                        "source": "rule"}), \
             patch.object(svc, "quality_check",
                          _gate(["failed", "passed"])), \
             patch.object(svc, "run_agent_graph", _path_stub(calls=calls)):
            events = list(svc.react_stream("ALD 原理", [], thread_id="t"))
        assert len(calls) == 2   # 同 tier 重做一次
        assert not any(e["type"] == "escalation" for e in events)
        assert any("重新生成" in e.get("message", "")
                   for e in events if e["type"] == "status")
        assert _types(events)[-1] == "done"

    def test_failed_redo_exhausted_releases(self):
        calls = []
        with patch.object(svc, "classify_complexity",
                          return_value={"tier": "medium", "confidence": 0.9,
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
                          return_value={"tier": "medium", "confidence": 0.9,
                                        "source": "rule"}), \
             patch.object(svc, "quality_check", gate), \
             patch.object(svc, "run_agent_graph", _err_path):
            events = list(svc.react_stream("ALD", [], thread_id="t"))
        gate.assert_not_called()
        assert any(e["type"] == "error" for e in events)
        assert _types(events)[-1] == "done"
