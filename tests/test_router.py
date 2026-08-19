# -*- coding: utf-8 -*-
"""阶段 3.6:复杂度路由器单元测试。

覆盖:
  - 规则预筛(问候/元问题 -> simple;复杂特征词 -> complex;空串 -> medium)
  - 领域信号的短问题不被规则误判 simple(交 LLM)
  - LLM 分类正常解析
  - LLM 输出不可解析 / 低置信度 -> 兜底 medium
  - LLM 异常 -> fail-open medium
不调用真实 LLM(get_client / llm_create_with_retry 全部打桩)。
"""
from unittest.mock import patch, MagicMock

import pytest

import agent_reasoning.router as router_mod
from agent_reasoning.router import (
    classify_complexity,
    _rule_prescreen,
    _parse_router_output,
)


# ---------- 规则预筛 ----------

class TestRulePrescreen:
    def test_empty_question_medium(self):
        assert _rule_prescreen("") == ("medium", 1.0)
        assert _rule_prescreen("   ") == ("medium", 1.0)

    def test_greeting_simple(self):
        for q in ["你好", "hi", "hello", "拜拜", "谢谢!"]:
            tier, conf = _rule_prescreen(q)
            assert tier == "simple", q
            assert conf > 0

    def test_meta_question_simple_without_domain(self):
        tier, _ = _rule_prescreen("你是谁?")
        assert tier == "simple"

    def test_meta_question_with_domain_signal_not_simple(self):
        # 含领域信号的"元问题"不应被规则直接判 simple(交给 LLM 或判更高级)
        tier, _ = _rule_prescreen("你能解释一下 ALD 是什么吗")
        assert tier != "simple"

    def test_complex_markers_complex(self):
        for q in ["对比 ALD 和 CVD 的优缺点", "分别说明光刻和刻蚀流程",
                  "总结两者的区别"]:
            tier, _ = _rule_prescreen(q)
            assert tier == "complex", q

    def test_short_no_domain_simple(self):
        tier, conf = _rule_prescreen("今天天气")
        assert tier == "simple"
        assert conf == pytest.approx(0.7)

    def test_short_domain_signal_not_prescreened(self):
        # "什么是 ALD" 很短但含领域信号 -> 规则不判定,交 LLM
        tier, conf = _rule_prescreen("什么是 ALD")
        assert tier is None and conf == 0.0

    def test_domain_fact_goes_to_llm(self):
        tier, _ = _rule_prescreen("解释一下晶圆制造中的薄膜沉积工艺")
        assert tier is None  # 无复杂特征词,但明显是领域题 -> 走 LLM


# ---------- 输出解析 ----------

class TestParseRouterOutput:
    def test_plain_json(self):
        assert _parse_router_output('{"tier":"medium","confidence":0.9}') == ("medium", 0.9)

    def test_markdown_fenced(self):
        text = '```json\n{"tier": "complex", "confidence": 0.8}\n```'
        assert _parse_router_output(text) == ("complex", 0.8)

    def test_text_with_surrounding_noise(self):
        text = '好的,结果是 {"tier": "simple", "confidence": 0.7} 以上。'
        assert _parse_router_output(text) == ("simple", 0.7)

    def test_invalid_tier(self):
        assert _parse_router_output('{"tier":"hard","confidence":0.9}')[0] is None

    def test_unparseable(self):
        assert _parse_router_output("no json here") == (None, 0.0)
        assert _parse_router_output("") == (None, 0.0)


# ---------- classify_complexity 集成(LLM 打桩) ----------

def _llm_return(content):
    """构造 llm_create_with_retry 的 (resp, None) 返回。"""
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp, None


class TestClassifyComplexity:
    def test_rule_short_circuits_llm(self):
        # 问候命中规则,不应调用 LLM
        with patch.object(router_mod, "llm_create_with_retry") as llm:
            d = classify_complexity("你好")
        assert d["tier"] == "simple"
        assert d["source"] == "rule"
        llm.assert_not_called()

    def test_llm_classifies_domain_question(self):
        with patch.object(router_mod, "get_client", return_value=MagicMock()), \
             patch.object(router_mod, "llm_create_with_retry",
                          return_value=_llm_return('{"tier":"medium","confidence":0.9}')):
            d = classify_complexity("什么是 ALD")
        assert d["tier"] == "medium"
        assert d["source"] == "llm"
        assert d["confidence"] == pytest.approx(0.9)

    def test_llm_uses_lite_model(self):
        """分类器应使用 lite 模型(TIER_MODEL_SIMPLE),温度 0。"""
        captured = {}

        def _fake(client, **kwargs):
            captured.update(kwargs)
            return _llm_return('{"tier":"medium","confidence":0.95}')

        with patch.object(router_mod, "get_client", return_value=MagicMock()), \
             patch.object(router_mod, "llm_create_with_retry", side_effect=_fake):
            classify_complexity("ALD 工艺的温度窗口是多少")
        import config as C
        assert captured.get("model") == C.TIER_MODEL_SIMPLE
        assert captured.get("temperature") == 0

    def test_unparseable_llm_output_fallback_medium(self):
        with patch.object(router_mod, "get_client", return_value=MagicMock()), \
             patch.object(router_mod, "llm_create_with_retry",
                          return_value=_llm_return("我无法分类")):
            d = classify_complexity("请帮我仔细分析一下这个说法的细节问题")
        assert d["tier"] == "medium"
        assert d["source"] == "fallback"

    def test_low_confidence_fallback_medium(self):
        with patch.object(router_mod, "get_client", return_value=MagicMock()), \
             patch.object(router_mod, "llm_create_with_retry",
                          return_value=_llm_return('{"tier":"complex","confidence":0.3}')):
            d = classify_complexity("请帮我仔细分析一下这个说法的细节问题")
        assert d["tier"] == "medium"
        assert d["source"] == "fallback"

    def test_llm_error_fallback_medium(self):
        with patch.object(router_mod, "get_client", return_value=MagicMock()), \
             patch.object(router_mod, "llm_create_with_retry",
                          return_value=(None, RuntimeError("timeout"))):
            d = classify_complexity("请帮我仔细分析一下这个说法的细节问题")
        assert d["tier"] == "medium"
        assert d["source"] == "fallback"

    def test_llm_exception_fallback_medium(self):
        with patch.object(router_mod, "get_client", return_value=MagicMock()), \
             patch.object(router_mod, "llm_create_with_retry",
                          side_effect=ConnectionError("boom")):
            d = classify_complexity("请帮我仔细分析一下这个说法的细节问题")
        assert d["tier"] == "medium"
        assert d["source"] == "fallback"
