# -*- coding: utf-8 -*-
"""阶段 8.4:Metrics 按推理范式分维度统计测试。"""
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
    m.record_tier_result("complex", 900)
    stats = m.get_stats()
    by = stats["by_tier"]
    assert by["simple"]["requests"] == 2
    assert by["complex"]["requests"] == 1
    assert by["simple"]["latency_ms"]["p50"] in (100, 200)
    assert by["complex"]["latency_ms"]["avg"] == 900


def test_escalation_rate_and_per_tier():
    m = _fresh()
    # 4 个请求,2 次升级(simple->medium, medium->complex)
    for _ in range(4):
        m.record_request("alice", 100)
    m.record_escalation("simple", "medium")
    m.record_escalation("medium", "complex")
    stats = m.get_stats()
    assert stats["escalations_total"] == 2
    assert stats["escalation_rate"] == 0.5
    assert stats["by_tier"] == {}  # 还没记 tier_result


def test_tier_grounding_and_tokens():
    m = _fresh()
    m.record_tier_result("medium", 300, grounding_passed=True,
                         tokens={"prompt": 10, "completion": 20, "total": 30})
    m.record_tier_result("medium", 400, grounding_passed=False,
                         tokens={"prompt": 5, "completion": 5, "total": 10})
    m.record_tier_result("medium", 500, error=True)
    by = m.get_stats()["by_tier"]["medium"]
    assert by["grounding"]["passed"] == 1
    assert by["grounding"]["failed"] == 1
    assert by["grounding"]["pass_rate"] == 0.5
    assert by["tokens"]["total"] == 40
    assert by["errors"] == 1
    assert by["error_rate"] == round(1 / 3, 4)


def test_simple_has_no_grounding_counts_as_none():
    m = _fresh()
    m.record_tier_result("simple", 50)  # grounding_passed=None
    by = m.get_stats()["by_tier"]["simple"]
    assert by["grounding"]["passed"] == 0
    assert by["grounding"]["failed"] == 0
    assert by["grounding"]["pass_rate"] == 0


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
