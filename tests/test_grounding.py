# -*- coding: utf-8 -*-
"""grounding.check tests: citation verification + faithfulness check."""
import pytest
from unittest.mock import patch, MagicMock
from chat.service import verify_citations, check_faithfulness, grounding_check

class TestVerifyCitations:
    def test_no_citations(self):
        ok, invalid = verify_citations("ALD is a technique.", [])
        assert ok is True and invalid == []

    def test_valid_citation(self):
        sources = [{"source_stem": "Oxford_Manual", "page": "12"}]
        ok, invalid = verify_citations("ALD uses TMA [Oxford_Manual p12].", sources)
        assert ok is True and invalid == []

    def test_invalid_citation_wrong_page(self):
        sources = [{"source_stem": "Oxford_Manual", "page": "12"}]
        ok, invalid = verify_citations("See [Oxford_Manual p999].", sources)
        assert ok is False and len(invalid) == 1 and "999" in invalid[0]

    def test_invalid_citation_wrong_doc(self):
        sources = [{"source_stem": "Oxford_Manual", "page": "12"}]
        ok, invalid = verify_citations("See [Fake_Doc p12].", sources)
        assert ok is False and len(invalid) == 1

    def test_multiple_citations_mixed(self):
        sources = [{"source_stem": "doc_a", "page": "1"}, {"source_stem": "doc_b", "page": "5"}]
        ok, invalid = verify_citations("A [doc_a p1] B [doc_b p5] C [doc_c p9].", sources)
        assert ok is False and len(invalid) == 1 and "doc_c" in invalid[0]

    def test_page_range_citation(self):
        sources = [{"source_stem": "manual", "page": "1"}]
        ok, invalid = verify_citations("See [manual p1-5].", sources)
        assert ok is True and invalid == []

    def test_citation_without_p_prefix(self):
        sources = [{"source_stem": "manual", "page": "12"}]
        ok, invalid = verify_citations("See [manual 12].", sources)
        assert ok is True and invalid == []

    def test_fuzzy_match_substring(self):
        sources = [{"source_stem": "Oxford_ALD_Operation_Manual", "page": "3"}]
        ok, invalid = verify_citations("See [Oxford p3].", sources)
        assert ok is True and invalid == []

    def test_fuzzy_match_reverse(self):
        sources = [{"source_stem": "Oxford", "page": "3"}]
        ok, invalid = verify_citations("See [Oxford_Manual p3].", sources)
        assert ok is True and invalid == []

    def test_all_valid_multiple_sources(self):
        sources = [{"source_stem": "d1", "page": "1"}, {"source_stem": "d2", "page": "2"}]
        ok, invalid = verify_citations("A [d1 p1] B [d2 p2].", sources)
        assert ok is True and invalid == []

class TestCheckFaithfulness:
    @patch("agent_reasoning.ReAct.support.answer_grounding.get_client")
    def test_faithful_high_score(self, mock_gc):
        from tests.conftest import make_llm_response
        mc = MagicMock()
        mc.chat.completions.create.return_value = make_llm_response('{"score": 0.95, "issues": []}')
        mock_gc.return_value = mc
        score, issues = check_faithfulness("ALD is a technique.", [{"source_stem": "d", "page": "1", "content": "ALD"}])
        assert score >= 0.9 and issues == []

    @patch("agent_reasoning.ReAct.support.answer_grounding.get_client")
    def test_unfaithful_low_score(self, mock_gc):
        from tests.conftest import make_llm_response
        mc = MagicMock()
        mc.chat.completions.create.return_value = make_llm_response('{"score": 0.2, "issues": ["unsupported"]}')
        mock_gc.return_value = mc
        score, issues = check_faithfulness("CVD is better.", [{"source_stem": "d", "page": "1", "content": "ALD"}])
        assert score < 0.5 and len(issues) >= 1

    def test_llm_failure_returns_none(self):
        with patch("agent_reasoning.ReAct.support.answer_grounding.get_client", side_effect=Exception("down")):
            result = check_faithfulness("ans.", [{"source_stem": "d", "page": "1", "content": "t"}])
            assert result is None

class TestGroundingCheck:
    @patch("agent_reasoning.ReAct.support.answer_grounding.check_faithfulness")
    def test_all_pass(self, mock_f):
        """高分 + 有效引用 + 忠实 -> 通过."""
        mock_f.return_value = (0.9, [])
        sources = [{"source_stem": "doc", "page": "1", "content": "x", "score": 0.9}]
        r = grounding_check("ALD [doc p1].", sources)
        assert r["passed"] is True and r["warnings"] == []

    @patch("agent_reasoning.ReAct.support.answer_grounding.check_faithfulness")
    def test_invalid_citation_warning(self, mock_f):
        """无效引用 -> 不通过."""
        mock_f.return_value = (0.9, [])
        sources = [{"source_stem": "doc", "page": "1", "content": "x", "score": 0.9}]
        r = grounding_check("ALD [fake p99].", sources)
        assert r["passed"] is False and len(r["warnings"]) >= 1

    @patch("agent_reasoning.ReAct.support.answer_grounding.check_faithfulness")
    def test_low_faithfulness_warning(self, mock_f):
        """高分 chunk 但答案不忠实 -> 不通过."""
        mock_f.return_value = (0.2, ["unsupported"])
        sources = [{"source_stem": "doc", "page": "1", "content": "x", "score": 0.9}]
        r = grounding_check("ALD [doc p1].", sources)
        assert r["passed"] is False

    def test_no_sources_passes(self):
        """无来源 -> 通过."""
        r = grounding_check("general answer.", [])
        assert r["passed"] is True

    def test_empty_answer_passes(self):
        """空回答 -> 通过."""
        r = grounding_check("", [{"source_stem": "d", "page": "1", "content": "x", "score": 0.9}])
        assert r["passed"] is True

    @patch("agent_reasoning.ReAct.support.answer_grounding.check_faithfulness")
    def test_low_score_skips_faithfulness(self, mock_f):
        """低分 chunk(<=0.5)不触发忠实度检测."""
        sources = [{"source_stem": "doc", "page": "1", "content": "x", "score": 0.3}]
        r = grounding_check("ALD [doc p1].", sources)
        mock_f.assert_not_called()
        assert r["passed"] is True

    @patch("agent_reasoning.ReAct.support.answer_grounding.check_faithfulness")
    def test_high_score_triggers_faithfulness(self, mock_f):
        """高分 chunk(>0.5)触发忠实度检测."""
        mock_f.return_value = (0.9, [])
        sources = [{"source_stem": "doc", "page": "1", "content": "x", "score": 0.8}]
        r = grounding_check("ALD [doc p1].", sources)
        mock_f.assert_called_once()

    @patch("agent_reasoning.ReAct.support.answer_grounding.check_faithfulness")
    def test_faithfulness_failure_warns_user(self, mock_f):
        """忠实度检测执行失败 -> 提示用户自行核实."""
        mock_f.return_value = None
        sources = [{"source_stem": "doc", "page": "1", "content": "x", "score": 0.9}]
        r = grounding_check("ALD [doc p1].", sources)
        assert r["passed"] is False
        assert any("暂不可用" in w for w in r["warnings"])
