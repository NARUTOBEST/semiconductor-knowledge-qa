# -*- coding: utf-8 -*-
"""阶段 9.1/9.3:tier 配置结构与 service 按 tier 取预算的测试。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as C  # noqa: E402


def test_tier_config_has_all_three_tiers():
    assert set(C.TIER_CONFIG) == {"simple", "medium", "complex"}
    for tier, cfg in C.TIER_CONFIG.items():
        assert {"model", "max_steps", "max_total_seconds",
                "plan_enabled", "quality_depth"} <= set(cfg)
        assert cfg["max_steps"] >= 1
        assert cfg["max_total_seconds"] >= 1


def test_tier_specific_defaults():
    assert C.TIER_CONFIG["simple"]["quality_depth"] == "light"
    assert C.TIER_CONFIG["simple"]["plan_enabled"] is False
    assert C.TIER_CONFIG["simple"]["max_steps"] == 1
    assert C.TIER_CONFIG["medium"]["quality_depth"] == "standard"
    assert C.TIER_CONFIG["medium"]["plan_enabled"] is True
    assert C.TIER_CONFIG["complex"]["quality_depth"] == "deep"
    assert C.TIER_CONFIG["complex"]["plan_enabled"] is True
    # complex 总预算应不小于 medium(多步执行)
    assert (C.TIER_CONFIG["complex"]["max_total_seconds"]
            >= C.TIER_CONFIG["medium"]["max_total_seconds"])


def test_tier_params_env_overridable(monkeypatch):
    """9.3:所有 tier 参数可经环境变量覆盖(重新加载 config 模块生效)。"""
    monkeypatch.setenv("TIER_MEDIUM_MAX_STEPS", "9")
    monkeypatch.setenv("TIER_MEDIUM_MAX_TOTAL_SECONDS", "99")
    monkeypatch.setenv("TIER_COMPLEX_MAX_TOTAL_SECONDS", "200")
    monkeypatch.setenv("TIER_MEDIUM_PLAN_ENABLED", "0")
    monkeypatch.setenv("ROUTER_CONFIDENCE_MIN", "0.8")

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "config_reloaded",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "config", "config.py"))
    cfg_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg_mod)

    assert cfg_mod.TIER_CONFIG["medium"]["max_steps"] == 9
    assert cfg_mod.TIER_CONFIG["medium"]["max_total_seconds"] == 99
    assert cfg_mod.TIER_CONFIG["medium"]["plan_enabled"] is False
    assert cfg_mod.TIER_CONFIG["complex"]["max_total_seconds"] == 200
    assert cfg_mod.ROUTER_CONFIDENCE_MIN == 0.8


def test_service_passes_per_tier_budgets(monkeypatch):
    """react_stream 按所选 tier 取 max_steps / max_total_seconds 传给路径。"""
    import server.chat.service as svc

    captured = {}

    def _fake_run_agent_graph(message, history, **kw):
        captured.update(kw)
        yield {"type": "assistant_message", "content": "答案"}
        yield {"type": "done", "trace": {"grounding": {"passed": True}}}

    monkeypatch.setattr(svc, "classify_complexity",
                        lambda *a, **k: {"tier": "medium", "confidence": 1.0, "source": "rule"})
    monkeypatch.setattr(svc, "run_agent_graph", _fake_run_agent_graph)
    monkeypatch.setattr(svc, "quality_check",
                        lambda *a, **k: {"verdict": "passed", "feedback": "", "warnings": []})

    events = list(svc.react_stream("ALD 原理", [], thread_id="t1"))
    assert any(e["type"] == "done" for e in events)
    assert captured["max_steps"] == C.TIER_CONFIG["medium"]["max_steps"]
    assert captured["max_total_seconds"] == C.TIER_CONFIG["medium"]["max_total_seconds"]


def test_service_explicit_budget_overrides_config(monkeypatch):
    import server.chat.service as svc

    captured = {}

    def _fake_run_agent_graph(message, history, **kw):
        captured.update(kw)
        yield {"type": "assistant_message", "content": "x"}
        yield {"type": "done", "trace": {"grounding": {"passed": True}}}

    monkeypatch.setattr(svc, "classify_complexity",
                        lambda *a, **k: {"tier": "medium", "confidence": 1.0, "source": "rule"})
    monkeypatch.setattr(svc, "run_agent_graph", _fake_run_agent_graph)
    monkeypatch.setattr(svc, "quality_check",
                        lambda *a, **k: {"verdict": "passed", "feedback": "", "warnings": []})

    list(svc.react_stream("q", [], thread_id="t1",
                          max_steps=3, max_total_seconds=30))
    assert captured["max_steps"] == 3
    assert captured["max_total_seconds"] == 30
