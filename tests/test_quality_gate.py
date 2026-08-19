# -*- coding: utf-8 -*-
"""阶段 4.6:共享质检门 quality_gate.check 三态判定 + 各 tier 深度。

不调真实 LLM:grounding_check 全部打桩。
"""
from unittest.mock import patch, MagicMock

import pytest

from agent_reasoning.quality_gate import check


_G_PASSED = {"passed": True, "warnings": []}
_G_FAILED = {"passed": False, "warnings": ["回答的部分内容未能从检索资料中验证"]}


# ---------- simple ----------

class TestSimpleGate:
    def test_chitchat_passes(self):
        r = check("你好,我是助手,很高兴和你聊天。", {}, tier="simple")
        assert r["verdict"] == "passed"

    def test_empty_answer_failed(self):
        r = check("", {}, tier="simple")
        assert r["verdict"] == "failed"
        r2 = check("  ", {}, tier="simple")
        assert r2["verdict"] == "failed"

    def test_domain_content_escalates(self):
        # 答案冒出半导体领域术语 -> 需升级检索
        r = check("ALD 是一种原子层沉积薄膜工艺。", {}, tier="simple")
        assert r["verdict"] == "needs_escalation"
        assert "medium" in r["feedback"]

    def test_does_not_call_grounding(self):
        # simple 只做启发式,不应调 grounding LLM
        with patch("agent_reasoning.quality_gate.grounding_check") as gc:
            check("你好呀", {}, tier="simple")
        gc.assert_not_called()


# ---------- medium ----------

class TestMediumGate:
    def test_grounding_passed(self):
        with patch("agent_reasoning.quality_gate.grounding_check",
                   return_value=_G_PASSED):
            r = check("根据资料,这是一个原理解释。",
                      {"question": "ALD 原理是什么?", "sources": [{"s": 1}]},
                      tier="medium")
        assert r["verdict"] == "passed"

    def test_grounding_failed_single_topic_failed(self):
        # 单一意图问题 grounding 失败、且无预计算结果 -> 退回重做
        with patch("agent_reasoning.quality_gate.grounding_check",
                   return_value=_G_FAILED):
            r = check("一段没有来源支撑的回答内容。",
                      {"question": "ALD 原理是什么?", "sources": [{"s": 1}]},
                      tier="medium")
        assert r["verdict"] == "failed"

    def test_grounding_failed_complex_question_escalates(self):
        # 问题本身含复杂特征 -> grounding 失败时升级 complex
        with patch("agent_reasoning.quality_gate.grounding_check",
                   return_value=_G_FAILED):
            r = check("一段没有来源支撑的回答内容。",
                      {"question": "对比 ALD 和 CVD 的优缺点和流程",
                       "sources": [{"s": 1}]},
                      tier="medium")
        assert r["verdict"] == "needs_escalation"

    def test_precomputed_grounding_failed_failopen_no_redo(self):
        # trace 已带 grounding 失败(图内部已 reflect 过一次) -> 不重复跑图,带警示放行
        with patch("agent_reasoning.quality_gate.grounding_check") as gc:
            r = check("一段回答。",
                      {"question": "ALD 原理?",
                       "grounding": _G_FAILED},
                      tier="medium")
        gc.assert_not_called()
        assert r["verdict"] == "passed"   # fail-open
        assert r["warnings"]              # 但带可见警示

    def test_empty_answer_failed(self):
        r = check("", {"question": "x"}, tier="medium")
        assert r["verdict"] == "failed"


# ---------- complex ----------

class TestComplexGate:
    def test_all_good_passes(self):
        with patch("agent_reasoning.quality_gate.grounding_check",
                   return_value=_G_PASSED):
            r = check("综合回答…",
                      {"question": "对比 ALD 和 CVD",
                       "coverage": {"uncovered_steps": []}},
                      tier="complex")
        assert r["verdict"] == "passed"

    def test_grounding_failed_is_failed_not_escalation(self):
        # complex 已是最高级,失败只返回 failed(不再升级)
        with patch("agent_reasoning.quality_gate.grounding_check",
                   return_value=_G_FAILED):
            r = check("回答…",
                      {"question": "对比", "coverage": {"uncovered_steps": []}},
                      tier="complex")
        assert r["verdict"] == "failed"

    def test_uncovered_steps_failed(self):
        with patch("agent_reasoning.quality_gate.grounding_check",
                   return_value=_G_PASSED):
            r = check("回答…",
                      {"question": "对比",
                       "coverage": {"uncovered_steps": ["刻蚀参数"]}},
                      tier="complex")
        assert r["verdict"] == "failed"
        assert any("刻蚀参数" in w for w in r["warnings"])

    def test_coverage_none_failopen(self):
        # 未启用 coverage tracker(None) -> 不因覆盖度阻断
        with patch("agent_reasoning.quality_gate.grounding_check",
                   return_value=_G_PASSED):
            r = check("回答…", {"question": "对比", "coverage": None},
                      tier="complex")
        assert r["verdict"] == "passed"


# ---------- fail-open ----------

class TestFailOpen:
    def test_grounding_exception_passes_with_warning(self):
        with patch("agent_reasoning.quality_gate.grounding_check",
                   side_effect=RuntimeError("llm down")):
            r = check("回答内容。",
                      {"question": "ALD 原理?", "sources": [{"s": 1}]},
                      tier="medium")
        # 异常被捕获 -> fail-open
        assert r["verdict"] == "passed"
        assert any("暂不可用" in w for w in r["warnings"])
