# -*- coding: utf-8 -*-
"""tool_status reducer 单测(故障隔离粘性 + setup 复位)。"""
from agent_reasoning.ReAct.core.state import _merge_tool_status
from tools import Category


def test_reset_overrides_everything():
    old = {"retrieval": "down"}
    reset = {"__reset__": True, "retrieval": "up"}
    out = _merge_tool_status(old, reset)
    assert out == {"retrieval": "up"}


def test_down_is_sticky():
    base = {"retrieval": "down"}
    # 后续工具节点尝试写 retrieval up 不应覆盖已 down 状态
    out = _merge_tool_status(base, {"retrieval": "up"})
    assert out["retrieval"] == "down"


def test_new_down_propagates():
    out = _merge_tool_status(
        {"retrieval": "up"},
        {Category.RETRIEVAL: "down"})
    assert out["retrieval"] == "down"


def test_empty_right_keeps_left():
    base = {"retrieval": "up"}
    assert _merge_tool_status(base, {}) == base


def test_none_left_uses_right():
    out = _merge_tool_status(None, {"retrieval": "down"})
    assert out == {"retrieval": "down"}
