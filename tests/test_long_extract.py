# -*- coding: utf-8 -*-
"""长期记忆【升迁门 consolidate_turn】单测:monkeypatch LLM / embed / DAO。

不依赖真实 LLM / 检索微服务 / PG。consolidate_turn 由 memory-loop 沉淀节点调用:
- LLM 成功:解析偏好,仅 importance ≥ MEM_PROMOTE_IMPORTANCE 的条目升迁落库;
- LLM 失败/坏 JSON:【抛异常】(由节点做 retry→degrade),不写库;
- 关闭/匿名/空:不调 LLM、不写。
"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib  # noqa: E402

import pytest  # noqa: E402
import config as C  # noqa: E402

ex_module = importlib.import_module("memories.orchestration.long.extract")


def _fake_resp(content: str):
    msg = types.SimpleNamespace(content=content)
    choice = types.SimpleNamespace(message=msg)
    return types.SimpleNamespace(choices=[choice])


class _FakeDAO:
    def __init__(self):
        self.upserts = []
        self.profiles = []

    def upsert_memory(self, username, **kw):
        self.upserts.append((username, kw))

    def upsert_profile(self, username, **kw):
        self.profiles.append((username, kw))


def _patch(monkeypatch, llm_text=None, llm_err=None, vectors=None,
           importance=0.6):
    dao = _FakeDAO()
    monkeypatch.setattr(ex_module, "long_term", dao)
    monkeypatch.setattr(ex_module, "embed_texts", lambda texts: vectors)
    if llm_err is not None:
        monkeypatch.setattr(ex_module, "chat_completion_with_fallback",
                            lambda **kw: (None, llm_err))
    else:
        monkeypatch.setattr(ex_module, "chat_completion_with_fallback",
                            lambda **kw: (_fake_resp(llm_text or "[]"), None))
    monkeypatch.setattr(C, "LONG_MEM_ENABLED", True, raising=False)
    monkeypatch.setattr(C, "MEM_PROMOTE_IMPORTANCE", importance, raising=False)
    return dao


def _call():
    return ex_module.consolidate_turn(
        "alice", "alice|t1", user_message="hi", assistant_message="hello")


def test_promotes_items_above_threshold(monkeypatch):
    llm_json = (
        '[{"category":"language","key":"response_language","content":"默认用英文回答","importance":0.9},'
        '{"category":"role","key":null,"content":"用户是半导体工艺工程师","importance":0.8},'
        '{"category":"fact","key":null,"content":"某条随口一提的事实","importance":0.3}]'
    )
    dao = _patch(monkeypatch, llm_text=llm_json,
                 vectors=[[0.1] * 4, [0.2] * 4, [0.3] * 4], importance=0.6)
    res = _call()

    # importance 0.3 的未达阈值,只升迁 2 条
    assert res["items"] == 3 and res["promoted"] == 2
    assert len(dao.upserts) == 2
    u0, kw0 = dao.upserts[0]
    assert u0 == "alice"
    assert kw0["category"] == "language" and kw0["key"] == "response_language"
    assert kw0["embedding"] == [0.1] * 4
    assert dao.upserts[1][1]["category"] == "role" and dao.upserts[1][1]["key"] is None
    # 画像汇总
    assert len(dao.profiles) == 1
    _, pkw = dao.profiles[0]
    assert pkw["display_prefs"]["response_language"] == "默认用英文回答"
    assert any("半导体工艺工程师" in x for x in pkw["top_interests"])


def test_strips_markdown_fence(monkeypatch):
    fenced = "```json\n[{\"category\":\"tone\",\"key\":\"answer_tone\",\"content\":\"喜欢简洁回答\",\"importance\":0.7}]\n```"
    dao = _patch(monkeypatch, llm_text=fenced, vectors=[[0.0] * 4])
    res = _call()
    assert res["promoted"] == 1
    assert dao.upserts[0][1]["category"] == "tone"


def test_unknown_category_falls_back_to_fact(monkeypatch):
    llm_json = '[{"category":"weird_cat","key":null,"content":"某条信息","importance":0.9}]'
    dao = _patch(monkeypatch, llm_text=llm_json, vectors=None)
    res = _call()
    assert res["promoted"] == 1
    assert dao.upserts[0][1]["category"] == "fact"
    assert dao.upserts[0][1]["embedding"] is None


def test_no_item_above_threshold_writes_nothing(monkeypatch):
    llm_json = '[{"category":"fact","key":null,"content":"随口一提","importance":0.2}]'
    dao = _patch(monkeypatch, llm_text=llm_json, vectors=None, importance=0.6)
    res = _call()
    assert res["items"] == 1 and res["promoted"] == 0
    assert dao.upserts == [] and dao.profiles == []


def test_empty_array_writes_nothing(monkeypatch):
    dao = _patch(monkeypatch, llm_text="[]", vectors=None)
    res = _call()
    assert res["promoted"] == 0
    assert dao.upserts == [] and dao.profiles == []


def test_llm_error_raises(monkeypatch):
    dao = _patch(monkeypatch, llm_err=RuntimeError("gateway down"))
    # 升迁门 LLM 失败必须抛出(节点据此 retry/degrade)
    with pytest.raises(RuntimeError):
        _call()
    assert dao.upserts == [] and dao.profiles == []


def test_bad_json_treated_as_no_items(monkeypatch):
    # 坏 JSON 被容错解析为空数组 -> 无升迁、不抛(仅 LLM 传输/调用失败才抛)
    dao = _patch(monkeypatch, llm_text="抱歉,我无法输出JSON", vectors=None)
    res = _call()
    assert res["promoted"] == 0
    assert dao.upserts == [] and dao.profiles == []


def test_disabled_or_anonymous_skips(monkeypatch):
    _patch(monkeypatch, llm_text="[]")
    monkeypatch.setattr(C, "LONG_MEM_ENABLED", False, raising=False)
    assert ex_module.consolidate_turn(
        "alice", "t", user_message="q", assistant_message="a")["promoted"] == 0
    monkeypatch.setattr(C, "LONG_MEM_ENABLED", True, raising=False)
    assert ex_module.consolidate_turn(
        None, "t", user_message="q", assistant_message="a")["promoted"] == 0


def test_schedule_thread_skips_when_disabled_or_anonymous(monkeypatch):
    started = []
    monkeypatch.setattr(ex_module.threading, "Thread",
                        lambda **kw: types.SimpleNamespace(start=lambda: started.append(1)))
    monkeypatch.setattr(C, "LONG_MEM_ENABLED", False, raising=False)
    ex_module.schedule_extraction(username="alice", user_message="q", assistant_message="a")
    assert started == []
    monkeypatch.setattr(C, "LONG_MEM_ENABLED", True, raising=False)
    ex_module.schedule_extraction(username=None, user_message="q", assistant_message="a")
    assert started == []
    ex_module.schedule_extraction(username="alice", user_message="q", assistant_message="a")
    assert len(started) == 1
