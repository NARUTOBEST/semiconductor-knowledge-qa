# -*- coding: utf-8 -*-
"""query.rewrite tests: LLM-driven query rewriting with mocked client."""
import pytest
from unittest.mock import patch, MagicMock
from query_rewrite import rewrite_query
from tests.conftest import make_llm_response

class TestSuccessfulRewrite:
    @patch("agent_reasoning.ReAct.support.llm.get_client")
    def test_single_query(self, mock_gc):
        mc = MagicMock()
        mc.chat.completions.create.return_value = make_llm_response('["TMA storage"]')
        mock_gc.return_value = mc
        assert rewrite_query("TMA storage", []) == ["TMA storage"]

    @patch("agent_reasoning.ReAct.support.llm.get_client")
    def test_multiple_queries(self, mock_gc):
        mc = MagicMock()
        mc.chat.completions.create.return_value = make_llm_response('["ALD principle", "CVD principle", "ALD CVD comparison"]')
        mock_gc.return_value = mc
        r = rewrite_query("ALD vs CVD", [])
        assert len(r) == 3 and "ALD principle" in r

    @patch("agent_reasoning.ReAct.support.llm.get_client")
    def test_markdown_code_block_stripped(self, mock_gc):
        mc = MagicMock()
        mc.chat.completions.create.return_value = make_llm_response('```json\n["q1", "q2"]\n```')
        mock_gc.return_value = mc
        assert rewrite_query("test", []) == ["q1", "q2"]

    @patch("agent_reasoning.ReAct.support.llm.get_client")
    def test_max_three_queries(self, mock_gc):
        mc = MagicMock()
        mc.chat.completions.create.return_value = make_llm_response('["q1","q2","q3","q4","q5"]')
        mock_gc.return_value = mc
        assert len(rewrite_query("test", [])) == 3

    @patch("agent_reasoning.ReAct.support.llm.get_client")
    def test_coreference_resolution(self, mock_gc):
        mc = MagicMock()
        mc.chat.completions.create.return_value = make_llm_response('["TMA storage temperature"]')
        mock_gc.return_value = mc
        history = [{"role": "user", "content": "TMA safety?"}, {"role": "assistant", "content": "TMA is reactive..."}]
        r = rewrite_query("its storage temp?", history)
        assert len(r) >= 1
        prompt = mc.chat.completions.create.call_args.kwargs["messages"][0]["content"]
        assert "TMA" in prompt and "reactive" in prompt

class TestFallback:
    @patch("agent_reasoning.ReAct.support.llm.get_client")
    def test_llm_exception_fallback(self, mock_gc):
        mock_gc.side_effect = Exception("LLM down")
        assert rewrite_query("original query", []) == ["original query"]

    @patch("agent_reasoning.ReAct.support.llm.get_client")
    def test_invalid_json_fallback(self, mock_gc):
        mc = MagicMock()
        mc.chat.completions.create.return_value = make_llm_response("not json")
        mock_gc.return_value = mc
        assert rewrite_query("my query", []) == ["my query"]

    @patch("agent_reasoning.ReAct.support.llm.get_client")
    def test_empty_array_fallback(self, mock_gc):
        mc = MagicMock()
        mc.chat.completions.create.return_value = make_llm_response("[]")
        mock_gc.return_value = mc
        assert rewrite_query("my query", []) == ["my query"]

    @patch("agent_reasoning.ReAct.support.llm.get_client")
    def test_non_array_json_fallback(self, mock_gc):
        mc = MagicMock()
        mc.chat.completions.create.return_value = make_llm_response('{"key": "val"}')
        mock_gc.return_value = mc
        assert rewrite_query("my query", []) == ["my query"]

    @patch("agent_reasoning.ReAct.support.llm.get_client")
    def test_empty_strings_filtered(self, mock_gc):
        mc = MagicMock()
        mc.chat.completions.create.return_value = make_llm_response('["q1", "", "q3"]')
        mock_gc.return_value = mc
        r = rewrite_query("test", [])
        assert "q1" in r and "q3" in r and "" not in r

class TestHistoryHandling:
    @patch("agent_reasoning.ReAct.support.llm.get_client")
    def test_empty_history(self, mock_gc):
        mc = MagicMock()
        mc.chat.completions.create.return_value = make_llm_response('["q"]')
        mock_gc.return_value = mc
        rewrite_query("test", [])
        prompt = mc.chat.completions.create.call_args.kwargs["messages"][0]["content"]
        assert "(\u65e0)" in prompt

    @patch("agent_reasoning.ReAct.support.llm.get_client")
    def test_none_history(self, mock_gc):
        mc = MagicMock()
        mc.chat.completions.create.return_value = make_llm_response('["q"]')
        mock_gc.return_value = mc
        assert len(rewrite_query("test", None)) >= 1

    @patch("agent_reasoning.ReAct.support.llm.get_client")
    def test_long_history_truncated(self, mock_gc):
        mc = MagicMock()
        mc.chat.completions.create.return_value = make_llm_response('["q"]')
        mock_gc.return_value = mc
        history = [{"role": "user", "content": f"msg{i}"} for i in range(20)]
        rewrite_query("test", history)
        prompt = mc.chat.completions.create.call_args.kwargs["messages"][0]["content"]
        assert "msg19" in prompt and "msg0" not in prompt
