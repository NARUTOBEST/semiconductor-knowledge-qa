# -*- coding: utf-8 -*-
"""检索冲突反问(20261015 规则:冲突即源限定反问)测试。

规则:投机检索机械冲突信号(top-4 跨文档族)命中后,经 _conflict_objects
前置 LLM 判定——问题已指明具体对象(specific)不反问照常作答;确实没指明
则发无选项源限定反问("请问您指的是哪份资料(手册/文档的名称)?"),
应答轮经 _scope_resolve 消解锁定文档族(与轮次21 Path B 同一机制)。
覆盖:
  - 冲突 + 非具体问题 → clarify(source=retrieval-conflict)+ pending 登记;
  - specific(问题已指明对象)→ 不反问照常作答;
  - 选项口径(offer=True)沿用 LLM 反问句与候选对象;
  - Path A 已锁定 scope 时冲突信号不再反问。
不连真实 LLM / GPU(全部打桩)。
"""
import sys

import pytest
from unittest.mock import patch

import chat.service as svc  # noqa: F401  确保双模块实例存在,清理用
import server.chat.service as svc2

_DONE = {"type": "done", "trace": {},
         "retrieval_max_score": 0.8, "search_count": 1}

_CITEMS = [{"chunk_id": "a1", "source_stem": "bookA手册", "content": "内容A"},
           {"chunk_id": "b1", "source_stem": "bookB手册", "content": "内容B"}]


def _react_stub(calls):
    def _gen(message, history, **k):
        calls.append({"message": message, **k})
        yield {"type": "assistant_message", "content": "答案。"}
        yield _DONE
    return _gen


@pytest.fixture(autouse=True)
def _clean():
    for mod in (svc, sys.modules.get("chat.service"), svc2):
        if mod is not None:
            mod._SCOPE_PENDING.clear()
            mod._SCOPE_LOCKS.clear()
    yield
    for mod in (svc, sys.modules.get("chat.service"), svc2):
        if mod is not None:
            mod._SCOPE_PENDING.clear()
            mod._SCOPE_LOCKS.clear()


def _run_conflict(monkeypatch, specific, citems=_CITEMS, **kw):
    """公共打桩:冲突信号 + 指定 _conflict_objects 判定,跑 react_stream。"""
    import server.chat.service as s
    monkeypatch.setattr(s.C, "RETRIEVAL_CONFLICT_CLARIFY", True)
    monkeypatch.setattr(s.C, "CLARIFY_OFFER_OPTIONS",
                        kw.pop("offer", False))
    offer_q = kw.pop("offer_q", "")
    choices = kw.pop("choices", [])
    probe = kw.pop("probe", lambda f: (["bookA手册", "bookB手册"], citems))
    calls = []
    entry = {"need_clarify": False, "tier": "raglite",
             "confidence": 0.9, "source": "rule"}
    with patch.object(s, "_conflict_probe", side_effect=probe), \
         patch.object(s, "_conflict_objects",
                      return_value=(specific, offer_q, choices)), \
         patch.object(s, "_speculative_search",
                      side_effect=lambda *a, **k: object()), \
         patch.object(s, "classify_and_clarify", return_value=entry), \
         patch.object(s, "run_raglite", _react_stub(calls)), \
         patch.object(s, "quality_check",
                      return_value={"verdict": "passed", "warnings": []}):
        ev = list(s.react_stream("这台设备的保养周期是多久?", [],
                                 thread_id="t", username="alice"))
    return ev, calls


class TestConflictSourceClarify:
    def test_conflict_fires_source_clarify(self, monkeypatch):
        """冲突 + 非具体问题 → 无选项源限定反问 + pending 登记(含原问题)。"""
        ev, _ = _run_conflict(monkeypatch, specific=False)
        cl = [e for e in ev if e["type"] == "clarify"]
        assert len(cl) == 1
        assert cl[0]["question"] == svc2._SOURCE_CLARIFY_QUESTION
        assert cl[0]["source"] == "retrieval-conflict"
        assert cl[0]["options"] == []
        assert not any(e["type"] == "assistant_message" for e in ev)
        assert not any(e["type"] == "tier" for e in ev)
        (pend,) = svc2._SCOPE_PENDING.values()
        assert pend["stems"] == ["bookA手册", "bookB手册"]
        assert pend["message"] == "这台设备的保养周期是多久?"  # 应答轮检索用原问题

    def test_specific_no_clarify(self, monkeypatch):
        """问题已指明具体对象(specific)→ 不反问,照常作答。"""
        ev, calls = _run_conflict(monkeypatch, specific=True)
        assert not any(e["type"] == "clarify" for e in ev)
        assert any(e["type"] == "assistant_message" for e in ev)
        assert calls  # raglite 正常消费

    def test_offer_mode_keeps_llm_question(self, monkeypatch):
        """选项口径(offer=True):沿用 LLM 反问句与候选对象(≥2 才反问)。"""
        ev, _ = _run_conflict(
            monkeypatch, specific=False, offer=True,
            offer_q="你想了解哪个对象?", choices=[("objA", "bookA手册"),
                                                 ("objB", "bookB手册")])
        cl = [e for e in ev if e["type"] == "clarify"]
        assert cl and cl[0]["question"] == "你想了解哪个对象?"
        assert cl[0]["options"] == ["objA", "objB"]

    def test_offer_mode_too_few_choices_no_clarify(self, monkeypatch):
        """offer 口径提不出 ≥2 个候选对象 → 不反问照常作答。"""
        ev, calls = _run_conflict(monkeypatch, specific=False, offer=True,
                                  offer_q="", choices=[])
        assert not any(e["type"] == "clarify" for e in ev)
        assert calls

    def test_scope_locked_conflict_skipped(self, monkeypatch):
        """Path A 已锁定 scope(直推)时冲突信号不再反问,照常作答。"""
        import server.chat.service as s

        def _fail_probe(f):
            raise AssertionError("scope 已锁定不应再探测冲突")

        monkeypatch.setattr(s.C, "SOURCE_LOCK_ENABLED", True)
        monkeypatch.setattr(s, "_infer_stem_via_engine",
                            lambda q: "bookA手册")
        # Path A 需题面有设备 token:打桩设备词面模块(题面含"手册里"触发检测)
        import mcp_servers.retrieval.devices as devices
        monkeypatch.setattr(devices, "extract_device_tokens",
                            lambda q: ["6DS-SP"])
        calls = []
        entry = {"need_clarify": False, "tier": "raglite",
                 "confidence": 0.9, "source": "rule"}
        with patch.object(s, "_conflict_probe", side_effect=_fail_probe), \
             patch.object(s, "_speculative_search",
                          side_effect=lambda *a, **k: object()) as fs, \
             patch.object(s, "classify_and_clarify", return_value=entry), \
             patch.object(s, "run_raglite", _react_stub(calls)), \
             patch.object(s, "quality_check",
                          return_value={"verdict": "passed", "warnings": []}):
            ev = list(s.react_stream("手册里这台设备的保养周期是多久?", [],
                                     thread_id="t", username="alice"))
        assert not any(e["type"] == "clarify" for e in ev)
        assert calls
        # 直推锁定:投机检索带 scope + 硬模式
        assert fs.call_args.kwargs.get("scope_stem") == "bookA手册"
        assert fs.call_args.kwargs.get("scope_hard") is True
