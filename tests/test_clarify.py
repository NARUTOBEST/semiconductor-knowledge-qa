# -*- coding: utf-8 -*-
"""澄清反问(ambiguity -> ask back)测试。

覆盖:
  - 规则预筛:无历史的纯指代/省略句 -> 必澄清;含型号/问候/超短 -> 不澄清;
    有历史时规则不下定论(交给 LLM)。
  - LLM 判定:clarify=true/false 解析、异常 fail-open。
  - 开关关闭 -> 直接跳过。
  - 服务层 react_stream:命中澄清时发 clarify+done 并短路,不调用任何推理路径。
不连真实 LLM(全部打桩)。
"""
import pytest
from unittest.mock import patch, MagicMock

import config as C
import agent_reasoning.clarify as clarify
import chat.service as svc


# 编排类测试:走旧两段式入口(clarify + classify_complexity 分别打桩),
# ROUTER_FUSED 融合入口的接线行为在 test_fused_entry.py 单独覆盖。
@pytest.fixture(autouse=True)
def _legacy_router_entry(monkeypatch):
    monkeypatch.setattr(C, "ROUTER_FUSED", 0)


def _llm_resp(content: str):
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp


class TestRulePrescreen:
    def test_standalone_pronoun_no_history_clarifies(self):
        for q in ["它怎么保养?", "这个呢", "然后呢？", "那台设备呢", "什么意思"]:
            r = clarify.check_clarify(q, [])
            assert r["need_clarify"] is True, q
            assert r["source"] == "rule"
            assert r["question"]

    def test_explicit_model_no_clarify(self):
        for q in ["TMA 的存储温度是多少", "ALD 原理是什么", "wafer chuck 温度控制"]:
            r = clarify.check_clarify(q, [])
            assert r["need_clarify"] is False, q

    def test_greeting_no_clarify(self):
        assert clarify.check_clarify("你好", [])["need_clarify"] is False
        assert clarify.check_clarify("hi", [])["source"] == "rule"

    def test_tiny_message_no_clarify(self):
        assert clarify.check_clarify("q", [])["need_clarify"] is False

    def test_ambiguous_with_history_defers_to_llm(self):
        # 有历史时代词可能可消解,规则不应下定论 -> _rule_prescreen 返回 None
        history = [{"role": "user", "content": "介绍一下 TMA 前驱体"},
                   {"role": "assistant", "content": "TMA 是三甲基铝..."}]
        assert clarify._rule_prescreen("它怎么保养?", history) is None
        # 但含型号的仍规则放行
        assert clarify._rule_prescreen("TMA 温度", history) is not None


class TestLlmClarify:
    def test_llm_says_clarify(self):
        content = '{"clarify": true, "question": "请问您指的是哪台设备?", "options": ["ALD 设备", "CVD 设备"]}'
        with patch.object(clarify, "llm_create_with_retry",
                          return_value=(_llm_resp(content), None)):
            r = clarify.check_clarify("温度多少", [{"role": "user", "content": "x"}])
        assert r["need_clarify"] is True
        assert "哪台设备" in r["question"]
        assert r["options"] == ["ALD 设备", "CVD 设备"]
        assert r["source"] == "llm"

    def test_llm_says_no_clarify(self):
        content = '{"clarify": false, "question": "", "options": []}'
        with patch.object(clarify, "llm_create_with_retry",
                          return_value=(_llm_resp(content), None)):
            r = clarify.check_clarify("温度多少", [{"role": "user", "content": "x"}])
        assert r["need_clarify"] is False

    def test_llm_unparseable_fail_open(self):
        with patch.object(clarify, "llm_create_with_retry",
                          return_value=(_llm_resp("我觉得不用问"), None)):
            r = clarify.check_clarify("温度多少", [{"role": "user", "content": "x"}])
        assert r["need_clarify"] is False

    def test_llm_error_fail_open(self):
        with patch.object(clarify, "llm_create_with_retry",
                          return_value=(None, MagicMock(name="err"))):
            r = clarify.check_clarify("温度多少", [{"role": "user", "content": "x"}])
        assert r["need_clarify"] is False

    def test_disabled_skips(self):
        with patch.object(C, "CLARIFY_ENABLED", False):
            r = clarify.check_clarify("它怎么保养?", [])
        assert r["need_clarify"] is False
        assert r["source"] == "off"


class TestServiceShortCircuit:
    def test_clarify_short_circuits_paths(self):
        # 命中澄清:只应产出 clarify + done,且不调用复杂度路由/推理路径
        classify = MagicMock()
        paths = MagicMock()
        with patch.object(svc, "check_clarify",
                          return_value={"need_clarify": True,
                                        "question": "请问指哪台设备?",
                                        "options": ["A", "B"], "source": "llm"}), \
             patch.object(svc, "classify_complexity", classify), \
             patch.object(svc, "run_simple", paths), \
             patch.object(svc, "run_agent_graph", paths):
            events = list(svc.react_stream("它呢", [], thread_id="t"))

        types = [e["type"] for e in events]
        assert types == ["clarify", "done"]
        c = events[0]
        assert c["question"] == "请问指哪台设备?"
        assert c["options"] == ["A", "B"]
        classify.assert_not_called()
        paths.assert_not_called()

    def test_no_clarify_proceeds_normally(self):
        def _path(message, history, **k):
            yield {"type": "assistant_message", "content": "答案"}
            yield {"type": "done", "trace": {}}

        with patch.object(svc, "check_clarify",
                          return_value={"need_clarify": False}), \
             patch.object(svc, "classify_complexity",
                          return_value={"tier": "react", "confidence": 0.9,
                                        "source": "rule"}), \
             patch.object(svc, "quality_check",
                          return_value={"verdict": "passed", "warnings": [],
                                        "feedback": ""}), \
             patch.object(svc, "run_agent_graph", _path):
            events = list(svc.react_stream("ALD 原理", [], thread_id="t"))

        types = [e["type"] for e in events]
        assert "tier" in types
        assert "clarify" not in types
        assert types[-1] == "done"
