# -*- coding: utf-8 -*-
"""ROUTER_FUSED=1 融合入口在服务层的接线测试。

覆盖:
  - classify_and_clarify 判 raglite -> run_raglite 执行,tier 事件对外仍报
    "react"(path=raglite),投机检索 future 只递给 raglite;
  - 带领域信号的问题先发投机检索;
  - raglite 质检 needs_escalation -> 升级 react(_NEXT_TIER 含 raglite)。
不连真实 LLM / MCP(全打桩)。
"""
import sys

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


class TestClarifyPendingRegistration:
    """入口判定器反问必须登记 _SCOPE_PENDING:否则应答轮 _scope_resolve
    匹配不到,clarified_q 为空,检索退化成选项文本(对象名),turn-1 命中的
    金块全丢(cascade_v2 dev200 五题硬失败根因)。"""

    ENTRY = {"need_clarify": True, "question": "请问您想了解哪个设备?",
             "options": ["WAFER HANDLER Mi30伺服电机", "Model 7AF"],
             "tier": "react", "confidence": 0.8, "source": "llm"}
    Q = "Overspeed 这个报警的判定条件是什么?"

    @pytest.fixture(autouse=True)
    def _clean_scope_state(self, monkeypatch):
        """_SCOPE_PENDING/_SCOPE_LOCKS 是模块级全局:用例间互清,防污染。
        chat.service 与 server.chat.service 是两个独立模块实例,都要清。
        本类测的是选项口径(CLARIFY_OFFER_OPTIONS=1 回滚路径)的登记与转正。"""
        import server.chat.service as svc
        monkeypatch.setattr(svc.C, "CLARIFY_OFFER_OPTIONS", True)
        for mod in (svc, sys.modules.get("chat.service")):
            if mod is not None:
                mod._SCOPE_PENDING.clear()
                mod._SCOPE_LOCKS.clear()
        yield

    def _run(self, message, history, entry, cap_search):
        import server.chat.service as svc
        with patch.object(svc, "classify_and_clarify", return_value=entry), \
             patch.object(svc, "_speculative_search",
                          side_effect=lambda q: cap_search.append(q)), \
             patch.object(svc, "quality_check",
                          return_value={"verdict": "passed", "warnings": []}), \
             patch.object(svc, "run_agent_graph", _react_stub([])):
            return list(svc.react_stream(message, history, thread_id="t",
                                         username="alice"))

    def test_fused_clarify_registers_pending(self):
        # 轮1:反问 -> 事件含 clarify;pending 已登记
        ev = self._run(self.Q, [], self.ENTRY, [])
        assert any(e["type"] == "clarify" for e in ev)
        import server.chat.service as svc
        assert len(svc._SCOPE_PENDING) == 1
        (pend,) = svc._SCOPE_PENDING.values()
        assert pend["message"] == self.Q            # 原问题存进去了
        assert pend["choices"][0][0] == "WAFER HANDLER Mi30伺服电机"

        # 轮2:用户回选项文本 -> _scope_resolve 转正,clarified_q=原问题,
        # 投机检索用原问题而非选项文本
        history = [{"role": "user", "content": self.Q},
                   {"role": "assistant", "content": "请问您想了解哪个设备?"}]
        picked = []
        entry2 = {"need_clarify": False, "tier": "react",
                  "confidence": 0.8, "source": "rule"}
        import server.chat.service as svc
        with patch.object(svc, "classify_and_clarify",
                          return_value=entry2), \
             patch.object(svc, "_speculative_search",
                          side_effect=lambda q, scope_stem="":
                              picked.append((q, scope_stem))), \
             patch.object(svc, "quality_check",
                          return_value={"verdict": "passed", "warnings": []}), \
             patch.object(svc, "run_agent_graph", _react_stub([])):
            ev2 = list(svc.react_stream("WAFER HANDLER Mi30伺服电机",
                                        history, thread_id="t",
                                        username="alice"))
        assert picked and picked[0][0] == self.Q   # 检索用原问题非选项文本
        assert svc._SCOPE_PENDING == {}             # pending 已消费

    def test_clarify_clarifies_too(self):
        """非融合路径(check_clarify)同样要登记 pending。"""
        import server.chat.service as svc
        clarify = {"need_clarify": True, "question": "哪台设备?",
                   "options": ["Mi30"], "source": "llm"}
        cap = []
        entry = {"need_clarify": False, "tier": "react",
                 "confidence": 0.8, "source": "rule"}
        with patch.object(svc.C, "ROUTER_FUSED", False), \
             patch.object(svc, "check_clarify", return_value=clarify), \
             patch.object(svc, "classify_complexity",
                          return_value={"tier": "react", "confidence": 0.8,
                                        "source": "rule"}), \
             patch.object(svc, "_speculative_search",
                          side_effect=lambda q: cap.append(q)), \
             patch.object(svc, "run_agent_graph", _react_stub([])), \
             patch.object(svc, "quality_check",
                          return_value={"verdict": "passed", "warnings": []}):
            ev = list(svc.react_stream(self.Q, [], thread_id="t",
                                       username="bob"))
        assert any(e["type"] == "clarify" for e in ev)
        assert len(svc._SCOPE_PENDING) == 1
        (pend,) = svc._SCOPE_PENDING.values()
        assert pend["message"] == self.Q

    def test_scope_store_shared_redis(self, monkeypatch):
        """Redis 可用时 pending/lock 必须走共享存储——多 worker 下进程内字典
        不共享,是 cascade_v2 五题 clarify 应答轮 clarified_q 丢失的根因。"""
        import server.chat.service as svc

        class FakeR:
            def __init__(self):
                self.d = {}
            def set(self, k, v, ex=None):
                self.d[k] = v
            def getdel(self, k):
                return self.d.pop(k, None)
            def get(self, k):
                return self.d.get(k)

        fake = FakeR()
        monkeypatch.setattr(svc, "_scope_redis", lambda: fake)
        # 清掉其他用例的内存残留(chat.service 与 server.chat.service 是两个
        # 模块实例,各自持有独立 _SCOPE_PENDING)
        for mod in (svc, sys.modules.get("chat.service")):
            if mod is not None:
                mod._SCOPE_PENDING.clear()
        svc._register_clarify_pending("u", [], "原始问题?", ["选项A", "选项B"])
        assert fake.d and svc._SCOPE_PENDING == {}   # 写进共享存储而非进程内
        stem, q = svc._scope_resolve(
            "u", "选项A", [{"role": "user", "content": "原始问题?"}])
        assert q == "原始问题?" and stem == ""
        assert svc._SCOPE_LOCKS == {}                # 锁也在共享存储


class TestClarifyFreeText:
    """反问无选项化(轮次20):指代不明一律自由文本反问,不给候选项;
    应答轮回复经引擎词面推断消解对象 → scope 锁 + 检索域收窄;
    推断失败 fail-open 用原问题检索。"""

    Q = "这台机器人的晶圆放置重复精度标称多少?"
    REPLY = "双臂取晶圆机器人magnatran"
    STEM = "双臂取晶圆机器人magnatran"

    @pytest.fixture(autouse=True)
    def _clean_and_default_off(self, monkeypatch):
        import server.chat.service as svc
        monkeypatch.setattr(svc.C, "CLARIFY_OFFER_OPTIONS", False)
        # 注意:不能动 RETRIEVAL_CONFLICT_CLARIFY——_scope_resolve 只在该开关
        # 开启时才被调用(service.react_stream 入口),关掉它应答轮整体旁路。
        for mod in (svc, sys.modules.get("chat.service")):
            if mod is not None:
                mod._SCOPE_PENDING.clear()
                mod._SCOPE_LOCKS.clear()
        yield

    def test_no_options_stripped_at_emission(self):
        """LLM 仍产出候选时,无选项口径在发射端强制剥成空数组。"""
        import server.chat.service as svc
        entry = {"need_clarify": True,
                 "question": "请问您指的是哪台设备?",
                 "options": ["Brooks ABM 407B 晶圆机器人", "IC测试机机器人"],
                 "tier": "react", "confidence": 0.8, "source": "llm"}
        with patch.object(svc, "classify_and_clarify", return_value=entry), \
             patch.object(svc, "_speculative_search"), \
             patch.object(svc, "quality_check",
                          return_value={"verdict": "passed", "warnings": []}):
            ev = list(svc.react_stream(self.Q, [], thread_id="t",
                                       username="carol"))
        cl = next(e for e in ev if e["type"] == "clarify")
        assert cl["question"] == "请问您指的是哪台设备?"
        assert cl["options"] == []                    # 候选被剥掉
        (pend,) = svc._SCOPE_PENDING.values()
        assert pend["choices"] == []                  # 登记同样无选项
        assert pend["message"] == self.Q

    def test_free_text_reply_resolves_stem_and_scope(self, monkeypatch):
        """应答轮自由文本 → 引擎推断出 stem:锁 scope + 检索域收窄生效,
        检索 query = 原问题 + 回复(设备名进 query 触发 DEVICE_FILTER)。"""
        import server.chat.service as svc
        entry = {"need_clarify": True, "question": "哪台设备?",
                 "options": [], "tier": "react", "confidence": 0.8,
                 "source": "llm"}
        cap = []
        # 轮1 反问
        with patch.object(svc, "classify_and_clarify", return_value=entry), \
             patch.object(svc, "_speculative_search"), \
             patch.object(svc, "quality_check",
                          return_value={"verdict": "passed", "warnings": []}):
            list(svc.react_stream(self.Q, [], thread_id="t", username="carol"))
        # 轮2:自由文本回复,_resolve_free_reply 打桩为推断成功
        history = [{"role": "user", "content": self.Q},
                   {"role": "assistant", "content": "哪台设备?"}]
        entry2 = {"need_clarify": False, "tier": "react", "confidence": 0.8,
                  "source": "rule"}
        with patch.object(svc, "_resolve_free_reply",
                          return_value=(self.STEM, self.REPLY)), \
             patch.object(svc, "classify_and_clarify",
                          return_value=entry2), \
             patch.object(svc, "_speculative_search",
                          side_effect=lambda q, scope_stem="":
                              cap.append((q, scope_stem))), \
             patch.object(svc, "quality_check",
                          return_value={"verdict": "passed", "warnings": []}), \
             patch.object(svc, "run_agent_graph", _react_stub([])):
            list(svc.react_stream(self.REPLY, history, thread_id="t",
                                  username="carol"))
        assert cap and cap[0] == (self.Q + " " + self.REPLY, self.STEM)
        # 锁已登记(粘性轮不再反问)
        assert any(v["stem"] == self.STEM for v in svc._SCOPE_LOCKS.values())

    def test_free_text_reply_failopen_uses_original_q(self, monkeypatch):
        """推断失败(纯语气词回复):fail-open——不加权不锁定,检索仍用原问题。"""
        import server.chat.service as svc
        entry = {"need_clarify": True, "question": "哪台设备?",
                 "options": [], "tier": "react", "confidence": 0.8,
                 "source": "llm"}
        with patch.object(svc, "classify_and_clarify", return_value=entry), \
             patch.object(svc, "_speculative_search"), \
             patch.object(svc, "quality_check",
                          return_value={"verdict": "passed", "warnings": []}):
            list(svc.react_stream(self.Q, [], thread_id="t", username="dave"))
        history = [{"role": "user", "content": self.Q},
                   {"role": "assistant", "content": "哪台设备?"}]
        entry2 = {"need_clarify": False, "tier": "react", "confidence": 0.8,
                  "source": "rule"}
        cap = []
        with patch.object(svc, "_resolve_free_reply",
                          return_value=("", "")), \
             patch.object(svc, "classify_and_clarify",
                          return_value=entry2), \
             patch.object(svc, "_speculative_search",
                          side_effect=lambda q, scope_stem="":
                              cap.append((q, scope_stem))), \
             patch.object(svc, "quality_check",
                          return_value={"verdict": "passed", "warnings": []}), \
             patch.object(svc, "run_agent_graph", _react_stub([])):
            list(svc.react_stream("嗯", history, thread_id="t",
                                  username="dave"))
        assert cap and cap[0] == (self.Q, "")         # 原问题原样检索
        assert svc._SCOPE_LOCKS == {}                 # 不误锁

    def test_resolve_free_reply_token_fallback(self, monkeypatch):
        """引擎推断不出但回复含设备 token → aux=回复原文(拼 query 交
        DEVICE_FILTER);无 token 回复不拼(噪声)。"""
        import server.chat.service as svc
        monkeypatch.setattr(svc, "_resolve_free_reply",
                            lambda r: ("", "") if r == "嗯" else ("", r))
        svc._register_clarify_pending("u", [], self.Q, [])
        stem, q = svc._scope_resolve(
            "u", "嗯", [{"role": "user", "content": self.Q}])
        assert (stem, q) == ("", self.Q)
        svc._register_clarify_pending("v", [], self.Q, [])
        stem, q = svc._scope_resolve(
            "v", "WAFER HANDLER Mi30",
            [{"role": "user", "content": self.Q}])
        assert stem == "" and q == self.Q + " WAFER HANDLER Mi30"
