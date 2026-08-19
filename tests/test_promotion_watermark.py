# -*- coding: utf-8 -*-
"""增量升迁水位线逻辑测试(打桩 short_term / long_term,不连真实 PG/LLM)。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memories.storage.long import promotion  # noqa: E402


def _ev(seq, etype="assistant_message", content="x"):
    return {"seq": seq, "event_type": etype,
            "payload": {"content": content}}


def _patch_short(monkeypatch, *, after_seq, new_events, fail_advance=False):
    state = {"watermark": after_seq, "advanced_to": None}
    monkeypatch.setattr(promotion.short_term, "get_watermark",
                        lambda tid: state["watermark"])
    monkeypatch.setattr(promotion.short_term, "list_events_after",
                        lambda tid, seq, **kw: list(new_events) if seq == state["watermark"] else [])
    # 桩掉持久化的失败计数:测试场景是"单次失败",返回 1(< MAX_PROMOTION_FAILURES),
    # 避免真实 PG 中跨运行累积的计数达到上限触发强制推进水位。
    monkeypatch.setattr(promotion.short_term, "record_promotion_failure",
                        lambda tid, max_fail=5: 1)
    if fail_advance:
        def _boom(tid, s):
            raise RuntimeError("pg down")
        monkeypatch.setattr(promotion.short_term, "advance_watermark", _boom)
    else:
        monkeypatch.setattr(promotion.short_term, "advance_watermark",
                            lambda tid, s: state.__setitem__("advanced_to", s))
    return state


def test_incremental_only_fetches_new_events(monkeypatch):
    new_events = [_ev(5), _ev(6)]
    st = _patch_short(monkeypatch, after_seq=4, new_events=new_events)
    monkeypatch.setattr(promotion, "_should_promote", lambda transcript: True)
    monkeypatch.setattr(promotion, "_extract_facts",
                        lambda transcript: ["事实A"])

    added = []
    monkeypatch.setattr(promotion.long_term, "add_memory",
                        lambda **kw: added.append(kw["content"]) or 101)

    ids = promotion.promote_thread("t1", "alice")
    assert ids == [101]
    assert added == ["事实A"]
    # 水位推进到本批最大 seq=6
    assert st["advanced_to"] == 6


def test_zero_facts_still_advances_watermark(monkeypatch):
    # 成功萃取但无事实 -> 也推进水位,避免每轮重放
    st = _patch_short(monkeypatch, after_seq=0, new_events=[_ev(1)])
    monkeypatch.setattr(promotion, "_should_promote", lambda transcript: True)
    monkeypatch.setattr(promotion, "_extract_facts", lambda transcript: [])

    def _no_add(**kw):
        raise AssertionError("无事实时不应写长期库")
    monkeypatch.setattr(promotion.long_term, "add_memory", _no_add)

    ids = promotion.promote_thread("t2", "alice")
    assert ids == []
    assert st["advanced_to"] == 1


def test_llm_failure_keeps_watermark(monkeypatch):
    # _extract_facts 返回 None(LLM 失败) -> 不推进水位,下次重试
    st = _patch_short(monkeypatch, after_seq=3, new_events=[_ev(4)])
    monkeypatch.setattr(promotion, "_should_promote", lambda transcript: True)
    monkeypatch.setattr(promotion, "_extract_facts", lambda transcript: None)

    def _no_add(**kw):
        raise AssertionError("LLM 失败时不应写长期库")
    monkeypatch.setattr(promotion.long_term, "add_memory", _no_add)

    ids = promotion.promote_thread("t3", "alice")
    assert ids == []
    assert st["advanced_to"] is None


# ---------------- 升迁守门员 ----------------

def test_gate_skip_advances_watermark(monkeypatch):
    # 守门员判定无价值(False) -> 跳过萃取、不写长期,但推进水位
    st = _patch_short(monkeypatch, after_seq=0, new_events=[_ev(1, content="你好")])
    called = []
    monkeypatch.setattr(promotion, "_should_promote",
                        lambda transcript: called.append("gate") or False)
    monkeypatch.setattr(promotion, "_extract_facts",
                        lambda transcript: called.append("extract") or ["X"])

    def _no_add(**kw):
        raise AssertionError("守门员跳过时不应写长期库")
    monkeypatch.setattr(promotion.long_term, "add_memory", _no_add)

    ids = promotion.promote_thread("t5", "alice")
    assert ids == []
    assert called == ["gate"]  # 萃取器未被调用
    assert st["advanced_to"] == 1


def test_gate_failure_keeps_watermark(monkeypatch):
    # 守门员返回 None(LLM 故障) -> 不推进水位、不写长期、不调萃取
    st = _patch_short(monkeypatch, after_seq=2, new_events=[_ev(3)])
    called = []
    monkeypatch.setattr(promotion, "_should_promote",
                        lambda transcript: called.append("gate") or None)
    monkeypatch.setattr(promotion, "_extract_facts",
                        lambda transcript: called.append("extract") or ["X"])

    def _no_add(**kw):
        raise AssertionError("守门员故障时不应写长期库")
    monkeypatch.setattr(promotion.long_term, "add_memory", _no_add)

    ids = promotion.promote_thread("t6", "alice")
    assert ids == []
    assert called == ["gate"]  # 萃取器未被调用
    assert st["advanced_to"] is None


def test_empty_transcript_advances_without_llm(monkeypatch):
    # 新增事件全是被白名单过滤的噪声 -> 无 transcript,直接推进,不调任何 LLM
    st = _patch_short(monkeypatch, after_seq=0,
                      new_events=[_ev(1, etype="token", content="x")])
    called = []
    monkeypatch.setattr(promotion, "_should_promote",
                        lambda transcript: called.append("gate") or True)
    monkeypatch.setattr(promotion, "_extract_facts",
                        lambda transcript: called.append("extract") or ["X"])

    ids = promotion.promote_thread("t7", "alice")
    assert ids == []
    assert called == []  # 守门员和萃取器都不调用
    assert st["advanced_to"] == 1


def test_no_new_events_short_circuits(monkeypatch):
    st = _patch_short(monkeypatch, after_seq=9, new_events=[])
    called = []
    monkeypatch.setattr(promotion, "_extract_facts",
                        lambda transcript: called.append(1) or [])
    ids = promotion.promote_thread("t4", "alice")
    assert ids == []
    assert called == []  # 没有新事件,不调 LLM
    assert st["advanced_to"] is None
