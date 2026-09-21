# -*- coding: utf-8 -*-
"""复杂度路由器单元测试(两级 simple/react)。

覆盖:
  - 规则预筛(问候/元问题 -> simple;空串 -> react;超短无领域 -> simple)
  - 领域信号(含对比/流程/综合)的问题不被规则误判 simple(交 LLM,多为 react)
  - LLM 分类正常解析
  - LLM 输出不可解析 / 低置信度 / 非法 tier -> 兜底 react
  - LLM 异常 -> fail-open react
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
    def test_empty_question_react(self):
        assert _rule_prescreen("") == ("react", 1.0)
        assert _rule_prescreen("   ") == ("react", 1.0)

    def test_greeting_simple(self):
        for q in ["你好", "hi", "hello", "拜拜", "谢谢!"]:
            tier, conf = _rule_prescreen(q)
            assert tier == "simple", q
            assert conf > 0

    def test_meta_question_simple_without_domain(self):
        tier, _ = _rule_prescreen("你是谁?")
        assert tier == "simple"

    def test_meta_question_with_domain_signal_not_simple(self):
        # 含领域信号的"元问题"不应被规则直接判 simple(交给 LLM)
        tier, _ = _rule_prescreen("你能解释一下 ALD 是什么吗")
        assert tier != "simple"

    def test_complex_questions_go_to_llm(self):
        # 非对比/汇总类的多步复杂问题不被规则预判,交 LLM(多判 react)
        for q in ["ALD 设备的日常维护流程是怎样的?请分步骤说明",
                  "划片工艺涉及哪些工具",
                  "请评估当前工艺参数下膜厚均匀性的表现并给出调整思路"]:
            tier, _ = _rule_prescreen(q)
            assert tier is None, q

    def test_agg_questions_react_fast_rule(self):
        # ROUTER_REACT_FAST:对比/区别类 + 领域词 = 规则直判 react(省一次串行 LLM)
        for q in ["对比 ALD 和 CVD 的优缺点",
                  "ALD、CVD 以及 PVD 三者的区别",
                  "ALD 和 CVD 有什么区别?"]:
            tier, conf = _rule_prescreen(q)
            assert tier == "react" and conf >= 0.8, q

    def test_single_object_cause_question_raglite_rule(self):
        # 路由收紧:单一对象的原因/排查类问法不再因"原因/排查"字样上抛 LLM,
        # 领域信号 + 非复杂标记即规则直判 raglite(实测 3/6 react 题属此类误路由)
        for q in ["排查键合机断线的可能原因",
                  "1416 报警是什么原因",
                  "SIPLACE 贴片机 X 轴原点丢失怎么修"]:
            tier, _ = _rule_prescreen(q)
            assert tier == "raglite", q

    def test_single_fact_question_raglite_rule(self):
        # 三级范式:单一事实点(领域信号 + 非复杂标记 + 短问题)规则直判 raglite
        for q in ["什么是 ALD",
                  "解释一下晶圆制造中的薄膜沉积工艺",
                  "ALD 工艺的温度窗口是多少"]:
            tier, conf = _rule_prescreen(q)
            assert tier == "raglite", q
            assert conf > 0, q

    def test_short_no_domain_simple(self):
        tier, conf = _rule_prescreen("今天天气")
        assert tier == "simple"
        assert conf == pytest.approx(0.7)

    def test_short_domain_signal_routed_raglite(self):
        # "什么是 ALD" 很短但含领域信号 -> 规则直判 raglite(零 LLM)
        tier, conf = _rule_prescreen("什么是 ALD")
        assert tier == "raglite" and conf > 0



# ---------- 输出解析 ----------

class TestParseRouterOutput:
    def test_plain_json(self):
        assert _parse_router_output('{"tier":"react","confidence":0.9}') == ("react", 0.9)

    def test_markdown_fenced(self):
        text = '```json\n{"tier": "simple", "confidence": 0.8}\n```'
        assert _parse_router_output(text) == ("simple", 0.8)

    def test_text_with_surrounding_noise(self):
        text = '好的,结果是 {"tier": "simple", "confidence": 0.7} 以上。'
        assert _parse_router_output(text) == ("simple", 0.7)

    def test_invalid_tier(self):
        # 已删除的 tier 名(plan/sequential)视为非法输出
        assert _parse_router_output('{"tier":"plan","confidence":0.9}')[0] is None
        assert _parse_router_output('{"tier":"sequential","confidence":0.9}')[0] is None
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
                          return_value=_llm_return('{"tier":"react","confidence":0.9}')):
            d = classify_complexity("请评估当前工艺参数下膜厚均匀性的表现并给出调整思路")
        assert d["tier"] == "react"
        assert d["source"] == "llm"
        assert d["confidence"] == pytest.approx(0.9)

    def test_llm_uses_lite_model(self):
        """分类器应使用 lite 模型(TIER_MODEL_SIMPLE),温度 0。"""
        captured = {}

        def _fake(client, **kwargs):
            captured.update(kwargs)
            return _llm_return('{"tier":"react","confidence":0.95}')

        with patch.object(router_mod, "get_client", return_value=MagicMock()), \
             patch.object(router_mod, "llm_create_with_retry", side_effect=_fake):
            classify_complexity("请评估当前工艺参数下膜厚均匀性的表现并给出调整思路")
        import config as C
        assert captured.get("model") == C.TIER_MODEL_SIMPLE
        assert captured.get("temperature") == 0

    def test_unparseable_llm_output_fallback_react(self):
        with patch.object(router_mod, "get_client", return_value=MagicMock()), \
             patch.object(router_mod, "llm_create_with_retry",
                          return_value=_llm_return("我无法分类")):
            d = classify_complexity("请帮我仔细分析一下这个说法的细节问题")
        assert d["tier"] == "react"
        assert d["source"] == "fallback"

    def test_low_confidence_fallback_react(self):
        with patch.object(router_mod, "get_client", return_value=MagicMock()), \
             patch.object(router_mod, "llm_create_with_retry",
                          return_value=_llm_return('{"tier":"simple","confidence":0.3}')):
            d = classify_complexity("请帮我仔细分析一下这个说法的细节问题")
        assert d["tier"] == "react"
        assert d["source"] == "fallback"

    def test_llm_error_fallback_react(self):
        with patch.object(router_mod, "get_client", return_value=MagicMock()), \
             patch.object(router_mod, "llm_create_with_retry",
                          return_value=(None, RuntimeError("timeout"))):
            d = classify_complexity("请帮我仔细分析一下这个说法的细节问题")
        assert d["tier"] == "react"
        assert d["source"] == "fallback"

    def test_llm_exception_fallback_react(self):
        with patch.object(router_mod, "get_client", return_value=MagicMock()), \
             patch.object(router_mod, "llm_create_with_retry",
                          side_effect=ConnectionError("boom")):
            d = classify_complexity("请帮我仔细分析一下这个说法的细节问题")
        assert d["tier"] == "react"
        assert d["source"] == "fallback"


# ---------- classify_and_clarify 融合入口(服务层 ROUTER_FUSED=1 走这条) ----------

class TestClassifyAndClarify:
    def test_rule_short_circuits_zero_llm(self):
        # 问候规则命中:不澄清、simple、不调 LLM
        with patch.object(router_mod, "llm_create_with_retry") as llm:
            d = router_mod.classify_and_clarify("你好", [])
        assert d["need_clarify"] is False
        assert d["tier"] == "simple"
        assert d["source"] == "rule"
        llm.assert_not_called()

    def test_clarify_rule_priority(self):
        # 澄清规则命中优先于路由(需要澄清时不作答,tier 为 None)
        d = router_mod.classify_and_clarify("它呢", [])
        assert d["need_clarify"] is True
        assert d["tier"] is None

    def test_raglite_rule_short_circuit(self):
        with patch.object(router_mod, "llm_create_with_retry") as llm:
            d = router_mod.classify_and_clarify("什么是 ALD", [])
        assert d["tier"] == "raglite"
        assert d["source"] == "rule"
        llm.assert_not_called()

    def test_llm_fused_output_parsed(self):
        content = ('{"need_clarify": false, "question": "", "options": [],'
                   ' "tier": "react", "confidence": 0.9}')
        with patch.object(router_mod, "get_client", return_value=MagicMock()),              patch.object(router_mod, "llm_create_with_retry",
                          return_value=_llm_return(content)):
            d = router_mod.classify_and_clarify("请评估当前工艺参数下膜厚均匀性的表现并给出调整思路", [])
        assert d["tier"] == "react"
        assert d["source"] == "llm"
        assert d["confidence"] == pytest.approx(0.9)

    def test_llm_react_downgraded_to_raglite_when_eligible(self):
        # 防提示漂移:LLM 判 react 但问题符合 raglite 规则准入 -> 降级 raglite
        content = ('{"need_clarify": false, "question": "", "options": [],'
                   ' "tier": "react", "confidence": 0.9}')
        with patch.object(router_mod, "get_client", return_value=MagicMock()),              patch.object(router_mod, "llm_create_with_retry",
                          return_value=_llm_return(content)):
            d = router_mod.classify_and_clarify("什么是 ALD", [])
        assert d["tier"] == "raglite"

    def test_llm_error_fallback_react_no_clarify(self):
        with patch.object(router_mod, "get_client", return_value=MagicMock()),              patch.object(router_mod, "llm_create_with_retry",
                          return_value=(None, RuntimeError("timeout"))):
            d = router_mod.classify_and_clarify("请评估当前工艺参数下膜厚均匀性的表现并给出调整思路", [])
        assert d["need_clarify"] is False
        assert d["tier"] == "react"
        assert d["source"] == "fallback"
