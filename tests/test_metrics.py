# -*- coding: utf-8 -*-
"""Metrics 按推理范式分维度统计测试(两级 simple/react)。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.support.metrics import Metrics  # noqa: E402


def _fresh():
    return Metrics()


def test_per_tier_request_count_and_latency():
    m = _fresh()
    m.record_tier_result("simple", 100)
    m.record_tier_result("simple", 200)
    m.record_tier_result("react", 900)
    stats = m.get_stats()
    by = stats["by_tier"]
    assert by["simple"]["requests"] == 2
    assert by["react"]["requests"] == 1
    assert by["simple"]["latency_ms"]["p50"] in (100, 200)
    assert by["react"]["latency_ms"]["avg"] == 900


def test_escalation_rate_and_per_tier():
    m = _fresh()
    # 4 个请求,2 次升级(simple->react)
    for _ in range(4):
        m.record_request("alice", 100)
    m.record_escalation("simple", "react")
    m.record_escalation("simple", "react")
    stats = m.get_stats()
    assert stats["escalations_total"] == 2
    assert stats["escalation_rate"] == 0.5
    assert stats["by_tier"] == {}  # 还没记 tier_result


def test_tier_tokens_and_errors():
    m = _fresh()
    m.record_tier_result("react", 300,
                         tokens={"prompt": 10, "completion": 20, "total": 30})
    m.record_tier_result("react", 400,
                         tokens={"prompt": 5, "completion": 5, "total": 10})
    m.record_tier_result("react", 500, error=True)
    by = m.get_stats()["by_tier"]["react"]
    assert by["tokens"]["total"] == 40
    assert by["errors"] == 1
    assert by["error_rate"] == round(1 / 3, 4)


def test_record_tier_result_thread_safe_counts():
    import threading
    m = _fresh()

    def worker():
        for _ in range(100):
            m.record_tier_result("simple", 10)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert m.get_stats()["by_tier"]["simple"]["requests"] == 400


# ---- 工具维度增强(category/duration/cache)----
def test_tool_call_backward_compatible():
    """旧签名 (name, success) 仍可工作。"""
    m = _fresh()
    m.record_tool_call("search_text", success=True)
    m.record_tool_call("search_text", success=False)
    t = m.get_stats()["tools"]["search_text"]
    assert t["calls"] == 2
    assert t["failures"] == 1
    assert t["category"] is None
    assert t["cache_hits"] == 0
    assert t["avg_duration_ms"] == 0


def test_tool_category_duration_cache():
    m = _fresh()
    m.record_tool_call("search_text", success=True, category="retrieval",
                       duration_ms=120, cache_hit=True)
    m.record_tool_call("search_text", success=True, category="retrieval",
                       duration_ms=80, cache_hit=False)
    m.record_tool_call("search_text", success=False, category="retrieval",
                       duration_ms=2000, error_type="timeout")
    t = m.get_stats()["tools"]["search_text"]
    assert t["calls"] == 3
    assert t["failures"] == 1
    assert t["category"] == "retrieval"
    assert t["cache_hits"] == 1
    assert t["avg_duration_ms"] == round((120 + 80 + 2000) / 3)
    assert t["p95_duration_ms"] >= 2000


def test_tool_category_aggregation():
    m = _fresh()
    m.record_tool_call("search_text", success=True, category="retrieval")
    m.record_tool_call("search_image", success=False, category="retrieval")
    m.record_tool_call("get_chunk", success=True, category="retrieval")
    cats = m.get_stats()["tool_categories"]
    assert cats["retrieval"]["calls"] == 3
    assert cats["retrieval"]["failures"] == 1
    assert cats["retrieval"]["success_rate"] == round(2 / 3, 4)
