# -*- coding: utf-8 -*-
"""eval 离线模块的确定性单测(不连 LLM/检索/Qdrant)。

只测纯函数:facts 命中、来源命中、余弦、分位、聚合、bad case 标注。
judge 的 LLM/嵌入调用不在单测范围(需服务,属手动端到端)。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval import metrics as M  # noqa: E402
from eval import report as R  # noqa: E402


class TestFactCoverage:
    def test_groups_are_any_match_within_and_all_across(self):
        # 两组:第一组近义词命中"料盒"即可,第二组需命中"更换"或"满"
        facts = [["料盒", "输出系统"], ["更换", "满"]]
        cov, hits = M.fact_coverage("请更换料盒,料盒已满。", facts)
        assert hits == [True, True]
        assert cov == 1.0

    def test_partial_coverage(self):
        facts = [["料盒"], ["报警代码1416"], ["重置"]]
        cov, hits = M.fact_coverage("打开料盒检查。", facts)
        assert hits == [True, False, False]
        assert abs(cov - 1 / 3) < 1e-6

    def test_empty_facts_full_coverage(self):
        # simple 闲聊题 facts=[] -> 不判负
        cov, hits = M.fact_coverage("你好,我是助手。", [])
        assert cov == 1.0 and hits == []


class TestSourceHit:
    def test_stem_contains_match(self):
        sources = [{"source_stem": "BESI Datacon 2200固晶机操作手册"}]
        assert M.source_hit(sources, "BESI固晶机") is True

    def test_no_match(self):
        assert M.source_hit([{"source_stem": "ASM焊线机手册"}], "BESI固晶机") is False

    def test_no_expected_source_passes(self):
        assert M.source_hit([], "") is True
        assert M.source_hit([{"source_stem": "x"}], None) is True


class TestCosine:
    def test_identical_and_orthogonal(self):
        assert abs(M.cosine([1, 0], [1, 0]) - 1.0) < 1e-6
        assert abs(M.cosine([1, 0], [0, 1])) < 1e-6

    def test_zero_vector_safe(self):
        assert M.cosine([0, 0], [1, 1]) == 0.0


class TestPercentile:
    def test_percentiles(self):
        vals = sorted([1.0, 2.0, 3.0, 4.0])
        assert M.percentile(vals, 0.5) == 2.5
        assert M.percentile(vals, 0.0) == 1.0
        assert M.percentile([], 0.5) == 0.0


def _rec(**kw):
    base = {"id": 1, "q": "q", "answer": "答",
            "sources": [{"source_stem": "BESI固晶机手册", "score": 0.9, "content": "x"}],
            "tier": "react",
            "expect_tier": "react", "expect_source": "", "ground_truth": "",
            "latency_s": 5.0, "redos": 0, "escalations": 0, "clarified": False,
            "final_reason": "answer", "retrieval_max_score": 0.9, "search_count": 1,
            "fact_coverage": 1.0, "fact_hits": [True], "source_hit": True}
    base.update(kw)
    return base


class TestAggregate:
    def test_aggregate_counts(self):
        recs = [
            _rec(id=1, latency_s=2.0, retrieval_max_score=0.9),
            _rec(id=2, latency_s=8.0, redos=1, tier="simple",
                 expect_tier="react", retrieval_max_score=0.2, fact_coverage=0.0),
        ]
        agg = M.aggregate(recs)
        assert agg["n"] == 2
        assert agg["redo_rate"] == 0.5
        assert agg["tier_accuracy"] == 0.5
        assert agg["latency_p50_s"] == 5.0
        assert agg["final_reason_dist"] == {"answer": 2}

    def test_ragas_averaged_when_present(self):
        recs = [_rec(id=1, faithfulness=1.0, context_precision=0.5),
                _rec(id=2, faithfulness=0.5)]  # 第二题缺 context_precision
        agg = M.aggregate(recs)
        assert agg["ragas_faithfulness"] == 0.75
        assert agg["ragas_context_precision"] == 0.5


class TestBadCaseTags:
    def test_good_case_no_tags(self):
        assert R.tag_bad_case(_rec()) == []

    def test_low_retrieval_confidence_tagged(self):
        tags = R.tag_bad_case(_rec(retrieval_max_score=0.1, search_count=2))
        assert any("检索低置信" in t for t in tags)

    def test_refusal_tagged(self):
        tags = R.tag_bad_case(_rec(answer="内部资料中暂未找到相关内容,建议联系FAE。"))
        assert any("拒答" in t for t in tags)

    def test_unfaithful_tagged(self):
        tags = R.tag_bad_case(_rec(faithfulness=0.3))
        assert any("不忠实" in t for t in tags)

    def test_wrong_tier_tagged(self):
        tags = R.tag_bad_case(_rec(tier="simple", expect_tier="react"))
        assert any("路由错" in t for t in tags)

    def test_error_tagged(self):
        tags = R.tag_bad_case(_rec(final_reason="timeout"))
        assert any("timeout" in t for t in tags)
