# -*- coding: utf-8 -*-
"""两级范式共享质检门 quality_gate.check 三态判定。

不调任何 LLM(两级都无旁路 LLM):
  - simple(L1):启发式,空答案 -> failed;领域内容 -> needs_escalation;闲聊 -> passed。
  - react(L2):只判空答案,空 -> failed;非空即 passed。
"""
from agent_reasoning.quality_gate import check


# ---------- simple(L1) ----------

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
        # 答案冒出半导体领域术语 -> 需升级到 react 检索
        r = check("ALD 是一种原子层沉积薄膜工艺。", {}, tier="simple")
        assert r["verdict"] == "needs_escalation"
        assert "react" in r["feedback"]


# ---------- react(L2):只判空答案,无旁路 LLM ----------

class TestReactGate:
    def test_non_empty_passes(self):
        r = check("根据检索资料,ALD 是一种原子层沉积工艺。",
                  {"question": "ALD 原理?", "history": []},
                  tier="react")
        assert r["verdict"] == "passed"

    def test_empty_answer_failed(self):
        r = check("", {"question": "x"}, tier="react")
        assert r["verdict"] == "failed"

    def test_unknown_tier_treated_as_react(self):
        r = check("一段回答。", {"question": "q"}, tier="nonsense")
        assert r["verdict"] == "passed"

    def test_low_confidence_redo_when_enabled(self, monkeypatch):
        """QC_LOW_CONF_REDO=1 时:低置信 -> failed(触发换关键词重进)。"""
        import config as C
        monkeypatch.setattr(C, "QC_LOW_CONF_REDO", True)
        r = check("一段非空回答。",
                  {"question": "q", "retrieval_max_score": 0.12, "search_count": 2},
                  tier="react")
        assert r["verdict"] == "failed"
        assert "关键词" in r["feedback"]

    def test_low_confidence_default_passes_with_warning(self):
        """默认(QC_LOW_CONF_REDO=0):低置信不再整轮 redo,放行+可见警示
        (react 的 reflect_node 循环内 requery 已覆盖重查,redo 纯属 +12s 重复)。"""
        import config as C
        assert not getattr(C, "QC_LOW_CONF_REDO", False)
        r = check("一段非空回答。",
                  {"question": "q", "retrieval_max_score": 0.12, "search_count": 2},
                  tier="react")
        assert r["verdict"] == "passed"
        assert r["warnings"], "低置信放行必须带可见警示"

    def test_high_confidence_passes(self):
        r = check("一段非空回答。",
                  {"question": "q", "retrieval_max_score": 0.9, "search_count": 1},
                  tier="react")
        assert r["verdict"] == "passed"

    def test_zero_score_without_search_passes(self):
        """没实际检索(search_count=0,如检索服务降级/模型未调工具)分数 0 不误触发低置信重做。"""
        r = check("一段非空回答。",
                  {"question": "q", "retrieval_max_score": 0.0, "search_count": 0},
                  tier="react")
        assert r["verdict"] == "passed"
