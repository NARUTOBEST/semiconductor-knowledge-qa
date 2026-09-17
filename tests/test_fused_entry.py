# -*- coding: utf-8 -*-
"""ROUTER_FUSED=1 融合入口在服务层的接线测试。

覆盖:
  - classify_and_clarify 判 raglite -> run_raglite 执行,tier 事件对外仍报
    "react"(path=raglite),投机检索 future 只递给 raglite;
  - 带领域信号的问题先发投机检索;
  - raglite 质检 needs_escalation -> 升级 react(_NEXT_TIER 含 raglite)。
不连真实 LLM / MCP(全打桩)。
"""
import pytest
from unittest.mock import patch, MagicMock

import chat.service as svc

_DONE = {"type": "done", "trace": {},
         "retrieval_max_score": 0.8, "search_count": 1}


def _raglite_stub(calls):
    def _gen(message, history, **k):
        calls.append({"message": message, **k})
        yield {"type": "assistant_message", "content": "单点事实回答。"}
        yield _DONE
    return _gen


def _react_stub(calls):
    def _gen(message, history, **k):
        calls.append({"message": message, **k})
        yield {"type": "assistant_message", "content": "react 回答。"}
        yield _DONE
    return _gen


class TestFusedEntry:
    def test_raglite_tier_reports_react_with_path(self):
        calls = []
        entry = {"need_clarify": False, "question": "", "options": [],
                 "tier": "raglite", "confidence": 0.9, "source": "rule"}
        with patch.object(svc, "classify_and_clarify", return_value=entry), \
             patch.object(svc, "_speculative_search",
                          return_value=MagicMock(name="future")), \
             patch.object(svc, "quality_check",
                          return_value={"verdict": "passed", "warnings": []}), \
             patch.object(svc, "run_raglite", _raglite_stub(calls)):
            events = list(svc.react_stream("什么是 ALD", [], thread_id="t",
                                           username="alice"))
        assert len(calls) == 1
        # 投机检索 future 只递给 raglite 消费
        assert calls[0]["search_future"] is not None
        tier_ev = next(e for e in events if e["type"] == "tier")
        assert tier_ev["tier"] == "react"      # 对外口径不变(eval/前端零改动)
        assert tier_ev["path"] == "raglite"    # 真实路径放 path 观测
        assert events[-1]["type"] == "done"

    def test_no_domain_signal_skips_speculative_search(self):
        calls = []
        entry = {"need_clarify": False, "question": "", "options": [],
                 "tier": "simple", "confidence": 0.9, "source": "rule"}
        with patch.object(svc, "classify_and_clarify", return_value=entry), \
             patch.object(svc, "_speculative_search") as spec, \
             patch.object(svc, "quality_check",
                          return_value={"verdict": "passed", "warnings": []}), \
             patch.object(svc, "run_simple", _react_stub(calls)):
            list(svc.react_stream("今天天气不错", [], thread_id="t"))
        spec.assert_not_called()
        assert not any(k == "search_future" and v is not None
                       for c in calls for k, v in c.items())

    def test_raglite_escalates_to_react(self):
        calls_r, calls_l = [], []
        entry = {"need_clarify": False, "question": "", "options": [],
                 "tier": "raglite", "confidence": 0.9, "source": "rule"}
        gate = MagicMock(return_value={"verdict": "needs_escalation",
                                       "feedback": "需要多步检索",
                                       "warnings": []})
        with patch.object(svc, "classify_and_clarify", return_value=entry), \
             patch.object(svc, "_speculative_search",
                          return_value=MagicMock(name="future")), \
             patch.object(svc, "quality_check", gate), \
             patch.object(svc, "run_raglite", _raglite_stub(calls_l)), \
             patch.object(svc, "run_agent_graph", _react_stub(calls_r)):
            events = list(svc.react_stream("什么是 ALD", [], thread_id="t"))
        assert len(calls_l) == 1 and len(calls_r) == 1
        esc = [e for e in events if e["type"] == "escalation"]
        assert esc and esc[0]["from_tier"] == "raglite"
        assert esc[0]["to_tier"] == "react"
        assert events[-1]["type"] == "done"


class TestReactTierKwargs:
    def test_react_tier_passes_search_future(self, monkeypatch):
        """react tier 把投机检索 future 透传给 run_agent_graph(首轮预检索注入,
        见 nodes.agent_node);simple 不透传。签名回归:runner 必须含该参数
        (曾因签名不含 search_future 引发 TypeError -> 47% 请求 '内部错误')。"""
        import inspect
        from agent_reasoning.ReAct.support import runner as react_runner
        assert "search_future" in inspect.signature(
            react_runner.run_agent_graph).parameters
        import server.chat.service as svc
        captured = {}
        def fake_graph(message, history, **kw):
            captured.update(kw)
            yield {"type": "done", "trace": {}}
        fut = object()  # 哨兵 future,验证透传
        monkeypatch.setattr(svc, "run_agent_graph", fake_graph)
        list(svc._run_tier("react", "q", [], thread_id="t",
                           max_steps=2, max_total_seconds=25,
                           hard_deadline=0.0, qc_feedback=None,
                           on_event=None, search_future=fut))
        assert captured.get("search_future") is fut

    def test_simple_tier_strips_search_future(self, monkeypatch):
        """simple tier 不消费投机检索 future(_NON_SIMPLE_KW 剔除)。"""
        import server.chat.service as svc
        captured = {}
        def fake_simple(message, history, **kw):
            captured.update(kw)
            yield {"type": "done", "trace": {}}
        monkeypatch.setattr(svc, "run_simple", fake_simple)
        list(svc._run_tier("simple", "q", [], thread_id="t",
                           max_steps=1, max_total_seconds=25,
                           hard_deadline=0.0, qc_feedback=None,
                           on_event=None, search_future="fut"))
        assert "search_future" not in captured
