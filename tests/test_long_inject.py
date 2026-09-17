# -*- coding: utf-8 -*-
"""长期记忆【注入/召回】单测:format_memory_block 文本 + recall 降级 + 图节点接线。

不依赖真实 PG / 检索微服务:monkeypatch long_term DAO 与 embed_one。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib  # noqa: E402

import config as C  # noqa: E402

inj_module = importlib.import_module("memories.orchestration.long.inject")


class _FakeDAO:
    def __init__(self, profile=None, relevant=None):
        self._profile = profile
        self._relevant = relevant or []

    def get_profile(self, username):
        return self._profile

    def search_relevant(self, username, qvec, k=None):
        return self._relevant


# ---------------- format_memory_block ----------------

def test_format_block_empty():
    assert inj_module.format_memory_block([], None) == ""
    assert inj_module.format_memory_block([], {}) == ""


def test_format_block_with_profile_and_relevant():
    profile = {
        "summary": "半导体工艺工程师",
        "top_interests": ["ALD 设备", "刻蚀工艺"],
        "display_prefs": {"response_language": "默认用英文回答"},
    }
    relevant = [
        {"content": "关注 ALD 原子层沉积", "distance": 0.4},
        {"content": "默认用英文回答", "distance": 0.1},
    ]
    block = inj_module.format_memory_block(relevant, profile)
    assert block.startswith("【用户长期记忆】")
    assert "半导体工艺工程师" in block
    assert "ALD 设备" in block
    assert "默认用英文回答" in block
    # 相关偏好按距离升序(更近的在前);去重:display_prefs 与 relevant 同句不重复不报错
    assert "与本问题相关的已知偏好" in block


def test_format_block_dedups_relevant():
    relevant = [
        {"content": "同一条偏好", "distance": 0.2},
        {"content": "同一条偏好", "distance": 0.3},
        {"content": "  ", "distance": 0.1},  # 空内容被跳过
    ]
    block = inj_module.format_memory_block(relevant, None)
    assert block.count("同一条偏好") == 1


# ---------------- recall_memories 降级 ----------------

def test_recall_disabled(monkeypatch):
    monkeypatch.setattr(C, "LONG_MEM_ENABLED", False, raising=False)
    called = {"profile": 0}
    dao = _FakeDAO()
    orig = dao.get_profile
    dao.get_profile = lambda u: (called.__setitem__("profile", called["profile"] + 1) or orig(u))
    monkeypatch.setattr(inj_module, "long_term", dao)
    assert inj_module.recall_memories("alice", "问题") == ([], None)
    assert called["profile"] == 0  # 关闭后根本不查 DAO


def test_recall_anonymous():
    # 无用户名 -> 空,不碰任何依赖
    assert inj_module.recall_memories(None, "问题") == ([], None)
    assert inj_module.recall_memories("", "问题") == ([], None)


def test_recall_embed_failure_still_returns_profile(monkeypatch):
    monkeypatch.setattr(C, "LONG_MEM_ENABLED", True, raising=False)
    profile = {"summary": "工程师", "top_interests": [], "display_prefs": {}}
    monkeypatch.setattr(inj_module, "long_term", _FakeDAO(profile=profile))
    monkeypatch.setattr(inj_module, "embed_one", lambda q: None)  # 嵌入失败
    relevant, prof = inj_module.recall_memories("alice", "问题")
    assert relevant == []          # 无向量 -> 不做语义召回
    assert prof is profile         # 但画像仍返回


def test_recall_happy_path(monkeypatch):
    monkeypatch.setattr(C, "LONG_MEM_ENABLED", True, raising=False)
    hits = [{"content": "关注 ALD", "distance": 0.2}]
    dao = _FakeDAO(profile={"summary": "s", "top_interests": [], "display_prefs": {}},
                   relevant=hits)
    searched = {}
    orig = dao.search_relevant

    def search(u, qvec, k=None):
        searched["qvec"] = qvec
        return orig(u, qvec, k=k)
    dao.search_relevant = search
    monkeypatch.setattr(inj_module, "long_term", dao)
    monkeypatch.setattr(inj_module, "embed_one", lambda q: [0.1] * 1024)
    relevant, prof = inj_module.recall_memories("alice", "ALD 是什么")
    assert relevant == hits
    assert searched["qvec"] == [0.1] * 1024


# ---------------- build_messages_node 不注入任何记忆 ----------------
# 记忆(短期近期对话 + 长期偏好)唯一入口是【模型 auto 决策触发的专用图节点
# recall_memory】;build_messages 只注入对话摘要(summary),system 里不塞任何记忆块,
# 也不调用近期对话读取(因此不依赖 Redis)。

def test_build_messages_does_not_inject_memory_block(monkeypatch):
    from agent_reasoning.ReAct.core import nodes

    # 即便记忆/近期对话函数能返回内容,build_messages 也不应调用或在 system 里出现记忆块
    monkeypatch.setattr(nodes, "format_summary_block", lambda summary: "")

    state = {
        "question": "ALD 设备怎么选?",
        "messages": [],
        "history": [],
        "trace_id": "t",
    }
    config = {"configurable": {"user_id": "alice", "thread_id": "th"}}
    patch = nodes.build_messages_node(state, config)
    sys_msg = patch["messages"][0]  # 第一条是 system
    content = sys_msg.content if hasattr(sys_msg, "content") else sys_msg["content"]
    assert "【用户长期记忆】" not in content
    assert "【会话历史对话】" not in content
    # nodes 模块不再持有任何强制召回/近期对话注入用的函数
    assert not hasattr(nodes, "_long_memory_block")
    assert not hasattr(nodes, "recent_dialogue_block")
