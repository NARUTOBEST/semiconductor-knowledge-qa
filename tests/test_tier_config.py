# -*- coding: utf-8 -*-
"""两级范式(simple/react):tier 配置结构与 service 按 tier 取预算的测试。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as C  # noqa: E402


def test_tier_config_has_three_tiers():
    assert set(C.TIER_CONFIG) == {"simple", "raglite", "react"}
    for tier, cfg in C.TIER_CONFIG.items():
        assert set(cfg) == {"model", "max_steps", "max_total_seconds"}
        assert cfg["max_steps"] >= 1
        assert cfg["max_total_seconds"] >= 1


def test_tier_specific_defaults():
    # simple(L1):单轮直答,无工具循环
    assert C.TIER_CONFIG["simple"]["max_steps"] == 1
    # raglite(L2):单点事实快路径,1 次检索 + 1 次作答,无循环
    assert C.TIER_CONFIG["raglite"]["max_steps"] == 1
    # react(L3):知识问答。默认 2 步 = 1 工具步 + 1 强制作答(压延迟);
    # 循环内自适应 requery 由 reflect_node 在 step<max_steps-1 时触发。
    assert C.TIER_CONFIG["react"]["max_steps"] == 2
    # 预算随层级递增
    assert (C.TIER_CONFIG["react"]["max_total_seconds"]
            >= C.TIER_CONFIG["simple"]["max_total_seconds"])


def test_tier_params_env_overridable(monkeypatch):
    """所有 tier 参数可经环境变量覆盖(重新加载 config 模块生效)。"""
    monkeypatch.setenv("TIER_REACT_MAX_STEPS", "2")
    monkeypatch.setenv("TIER_REACT_MAX_TOTAL_SECONDS", "44")
    monkeypatch.setenv("ROUTER_CONFIDENCE_MIN", "0.8")

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "config_reloaded",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "config", "config.py"))
    cfg_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg_mod)

    assert cfg_mod.TIER_CONFIG["react"]["max_steps"] == 2
    assert cfg_mod.TIER_CONFIG["react"]["max_total_seconds"] == 44
    assert cfg_mod.ROUTER_CONFIDENCE_MIN == 0.8


def test_service_passes_per_tier_budgets(monkeypatch):
    """react_stream 按所选 tier 取 max_steps / max_total_seconds 传给路径。"""
    import server.chat.service as svc
    monkeypatch.setattr(C, "ROUTER_FUSED", 0)

    captured = {}

    def _fake_run_agent_graph(message, history, **kw):
        captured.update(kw)
        yield {"type": "assistant_message", "content": "答案"}
        yield {"type": "done", "trace": {}}

    monkeypatch.setattr(svc, "classify_complexity",
                        lambda *a, **k: {"tier": "react", "confidence": 1.0, "source": "rule"})
    monkeypatch.setattr(svc, "run_agent_graph", _fake_run_agent_graph)
    monkeypatch.setattr(svc, "quality_check",
                        lambda *a, **k: {"verdict": "passed", "feedback": "", "warnings": []})

    events = list(svc.react_stream("ALD 原理", [], thread_id="t1"))
    assert any(e["type"] == "done" for e in events)
    assert captured["max_steps"] == C.TIER_CONFIG["react"]["max_steps"]
    assert captured["max_total_seconds"] == C.TIER_CONFIG["react"]["max_total_seconds"]
    # 两级范式不传任何旁路开关
    assert not any(k.startswith("enable_") for k in captured)


def test_service_explicit_budget_overrides_config(monkeypatch):
    import server.chat.service as svc
    monkeypatch.setattr(C, "ROUTER_FUSED", 0)

    captured = {}

    def _fake_run_agent_graph(message, history, **kw):
        captured.update(kw)
        yield {"type": "assistant_message", "content": "x"}
        yield {"type": "done", "trace": {}}

    monkeypatch.setattr(svc, "classify_complexity",
                        lambda *a, **k: {"tier": "react", "confidence": 1.0, "source": "rule"})
    monkeypatch.setattr(svc, "run_agent_graph", _fake_run_agent_graph)
    monkeypatch.setattr(svc, "quality_check",
                        lambda *a, **k: {"verdict": "passed", "feedback": "", "warnings": []})

    list(svc.react_stream("q", [], thread_id="t1",
                          max_steps=3, max_total_seconds=30))
    assert captured["max_steps"] == 3
    assert captured["max_total_seconds"] == 30
