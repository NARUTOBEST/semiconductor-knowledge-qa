# -*- coding: utf-8 -*-
"""首轮(冷启动)上下文种子来源测试。

冷启动 = checkpointer 无既有 messages(首轮/重启恢复/换设备)。此时多轮上下文以
服务端 Redis 短期流水为权威来源(结构化 user/assistant 消息),不再依赖前端重发
history;仅当 Redis 不可用/无流水时才退回前端 history 兜底。冷启动轮预取跳过②近期
文本块(避免与结构化种子重复);暖启动(续跑)维持 include_recent=True。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import HumanMessage, AIMessage  # noqa: E402

from agent_reasoning.ReAct.core import nodes  # noqa: E402
import memories.orchestration.long.prefetch as pf  # noqa: E402
import memories.orchestration.short.recall as recall  # noqa: E402


def _patch(monkeypatch, *, seed_turns, prefetch_block=""):
    """打桩预取与 Redis 近期读取,记录 prefetch 的 include_recent 入参。"""
    calls = {}

    def _fake_prefetch(username, thread_id, question, *, include_recent=True):
        calls["include_recent"] = include_recent
        return {"block": prefetch_block, "mem_ids": set(), "degraded": []}

    monkeypatch.setattr(pf, "build_prefetch_block", _fake_prefetch)
    monkeypatch.setattr(recall, "recent_dialogue_messages",
                        lambda *a, **k: list(seed_turns))
    return calls


def _contents(patch):
    return [m.content for m in patch["messages"]]


def test_cold_start_seeds_from_redis_not_frontend(monkeypatch):
    calls = _patch(monkeypatch, seed_turns=[
        {"role": "user", "content": "REDIS-上一问"},
        {"role": "assistant", "content": "REDIS-上一答"},
    ])
    state = {
        "question": "这一问",
        "messages": [],  # 冷启动
        # 前端即使带了 history,Redis 有流水时应被忽略
        "history": [{"role": "user", "content": "FRONTEND-旧问"}],
        "trace_id": "t",
    }
    cfg = {"configurable": {"user_id": "alice", "thread_id": "alice|c1"}}
    patch = nodes.build_messages_node(state, cfg)

    contents = _contents(patch)
    assert "REDIS-上一问" in contents
    assert "REDIS-上一答" in contents
    assert "FRONTEND-旧问" not in contents      # 前端 history 不作来源
    assert contents[-1] == "这一问"              # 本轮问题在末尾
    # 角色顺序正确:user/assistant 结构化铺排
    roles = [type(m) for m in patch["messages"]]
    assert roles[1] is HumanMessage and roles[2] is AIMessage and roles[3] is HumanMessage
    # 冷启动:预取跳过近期块(避免与结构化种子重复)
    assert calls["include_recent"] is False


def test_cold_start_falls_back_to_frontend_when_redis_empty(monkeypatch):
    _patch(monkeypatch, seed_turns=[])  # Redis 无流水/不可用
    state = {
        "question": "这一问",
        "messages": [],
        "history": [{"role": "user", "content": "FE-旧问"},
                    {"role": "assistant", "content": "FE-旧答"}],
        "trace_id": "t",
    }
    cfg = {"configurable": {"user_id": "alice", "thread_id": "alice|c1"}}
    patch = nodes.build_messages_node(state, cfg)
    contents = _contents(patch)
    assert "FE-旧问" in contents and "FE-旧答" in contents  # 退回前端兜底
    assert contents[-1] == "这一问"


def test_warm_start_keeps_existing_and_includes_recent(monkeypatch):
    calls = _patch(monkeypatch, seed_turns=[
        {"role": "user", "content": "不应出现在暖启动种子里"}])
    state = {
        "question": "这一问",
        # 暖启动:checkpointer 已有消息
        "messages": [HumanMessage(content="早先的问题")],
        "history": [{"role": "user", "content": "FRONTEND-旧"}],
        "trace_id": "t",
    }
    cfg = {"configurable": {"user_id": "alice", "thread_id": "alice|c1"}}
    patch = nodes.build_messages_node(state, cfg)
    contents = _contents(patch)
    # 续跑:不重灌种子/前端历史,只更新 system + 追加本轮问题
    assert "FRONTEND-旧" not in contents
    assert "不应出现在暖启动种子里" not in contents
    assert contents[-1] == "这一问"
    # 暖启动:预取仍注入近期块
    assert calls["include_recent"] is True
