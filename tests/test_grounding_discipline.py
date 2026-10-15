# -*- coding: utf-8 -*-
"""断言纪律(20261015 幻觉根治轮):system prompt 注入测试。

v4 全量实测 6 道新增幻觉题中 4-5 道属"答案方向正确但混入上下文之外的断言"。
GROUNDING_DISCIPLINE 三条红线必须同时出现在 raglite 与 react 的 system prompt
中(静态注入,全链路生效)。不连 LLM,纯常量断言。
"""
import pytest

from system_prompt import (
    GROUNDING_DISCIPLINE,
    RAGLITE_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
)

# 三条红线各自的关键词锚点(改文案时同步)
_RULE_ANCHORS = [
    "不得具体化到",      # 1) 覆盖范围描述不得超出块原文
    "按题面给定",        # 2) 题面数字不得说成检索所得
    "来自题面还是资料",  # 3) 推导输入值标注来源
]


def test_discipline_covers_three_rules():
    for anchor in _RULE_ANCHORS:
        assert anchor in GROUNDING_DISCIPLINE, anchor


@pytest.mark.parametrize("prompt", [RAGLITE_SYSTEM_PROMPT, SYSTEM_PROMPT],
                         ids=["raglite", "react"])
def test_injected_into_both_prompts(prompt):
    assert "【断言纪律】" in prompt
    for anchor in _RULE_ANCHORS:
        assert anchor in prompt


def test_discipline_before_safety_section():
    # 纪律段须在安全规则之前(组装顺序:作答规则 → 断言纪律 → 安全规则)
    for prompt in (RAGLITE_SYSTEM_PROMPT, SYSTEM_PROMPT):
        assert prompt.index("【断言纪律】") < prompt.index("# 安全规则")


def test_scope_guidance_unchanged():
    # 源限定口径(轮次21)不受影响
    from system_prompt import scope_lock_guidance
    g = scope_lock_guidance("bookA")
    assert "《bookA》" in g and "未涉及" in g
