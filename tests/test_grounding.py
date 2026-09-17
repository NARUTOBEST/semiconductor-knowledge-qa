# -*- coding: utf-8 -*-
"""grounding 置信度门控单测:层1数字硬校验 / 引文机械验证 / 低置信→人工引导。

只测纯规则与门控策略,LLM 调用路径打桩(见 _stub_llm)。
"""
import pytest

import agent_reasoning.ReAct.support.grounding as G


# ---------- 句子拆分 / 数字抽取 ----------

def test_split_sentences():
    sents = G._split_sentences("垫脚数量为6个。每个垫脚承重500kg。")
    assert sents == ["垫脚数量为6个。", "每个垫脚承重500kg。"]


def test_extract_numbers_normalization():
    assert G._extract_numbers("4,000 个") == {"4000"}
    assert G._extract_numbers("1.2万转") == {"12000"}
    assert G._extract_numbers("步骤1") == set()  # 个位数忽略


# ---------- 层1:数字硬校验 ----------

def test_layer1_kills_unsupported_numbers():
    sents = ["温度设置为150度。", "泵速为9999转。"]
    src_nums = G._extract_numbers("温度设置范围 20~150 度。")
    ok, killed = G._layer1_check(sents, src_nums)
    assert ok == [True, False]
    assert killed == 1


# ---------- 层3:置信度门控 ----------

SOURCES = [{"source_stem": "NXT-II 安装手册", "page": 33,
            "content": "4M-2基座使用6个调整垫脚。"}]


def test_low_confidence_returns_guidance(monkeypatch):
    """判 false 句占比超 REMOVAL_CAP → 答案整段替换为人工引导文本。"""
    monkeypatch.setattr(G.C, "GROUNDING_MIN_CONFIDENCE", 0.6, raising=False)
    monkeypatch.setattr(G.C, "GROUNDING_REMOVAL_CAP", 0.4, raising=False)
    sents = ["A。", "B。", "C。", "D。"]  # 4 句全 false → confidence 0
    out, info = G._apply_fuse_and_delete("原文", sents,
                                         [False, False, False, False], {}, SOURCES)
    assert info["action"] == "guidance"
    assert info["cap_hit"] is True
    assert "人工" in out and "FAE" in out
    assert "NXT-II 安装手册" in out and "33" in out


def test_confidence_at_threshold_prunes(monkeypatch):
    """置信度达标(≥0.6)→ 走删句路径,不替换。"""
    monkeypatch.setattr(G.C, "GROUNDING_MIN_CONFIDENCE", 0.6, raising=False)
    sents = ["A。", "B。", "C。", "D。"]  # 3/4 = 0.75 达标
    out, info = G._apply_fuse_and_delete("原文", sents,
                                         [True, True, True, False], {}, SOURCES)
    assert info["action"] == "pruned"
    assert info["removed"] == 1
    assert out == "A。B。C。"


def test_min_confidence_boundary(monkeypatch):
    """置信度恰在阈值上不替换;低于阈值替换。"""
    monkeypatch.setattr(G.C, "GROUNDING_MIN_CONFIDENCE", 0.6, raising=False)
    sents = ["A。", "B。", "C。", "D。", "E。"]
    # 3/5 = 0.6 达标边界
    _, info = G._apply_fuse_and_delete("原文", sents,
                                       [True, True, True, False, False], {}, SOURCES)
    assert info["action"] == "pruned"
    # 2/5 = 0.4 < 0.6 → 引导
    out, info2 = G._apply_fuse_and_delete("原文", sents,
                                          [True, True, False, False, False], {}, SOURCES)
    assert info2["action"] == "guidance"
    assert "未能通过资料一致性校验" in out


def test_all_supported_passthrough():
    out, info = G._apply_fuse_and_delete("原文", ["A。", "B。"],
                                         [True, True], {}, SOURCES)
    assert info["action"] == "passthrough"
    assert info["removed"] == 0
    assert out == "原文"


def test_guidance_text_without_sources():
    out = G._guidance_text([])
    assert "翻阅" in out and "FAE" in out


# ---------- grounding_filter 快速旁路 ----------

def test_filter_skips_short_answers():
    out, info = G.grounding_filter("一句话。", SOURCES)
    assert info["enabled"] is False
    assert out == "一句话。"


def test_filter_skips_no_contexts():
    out, info = G.grounding_filter("第一句。第二句。", [])
    assert info["enabled"] is False
    assert out == "第一句。第二句。"


# ---------- LLM 校验主流程(打桩) ----------

class _FakeMsg:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMsg(content)


class _FakeResp:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


def _stub_llm(monkeypatch, verdicts_json):
    """打桩 llm_create_with_retry:返回预设 JSON。
    grounding_filter 在调用点 ``from .llm import llm_create_with_retry``,
    故 patch 模块属性即可(monkeypatch 自动恢复)。"""
    def _fake(client, **kwargs):
        return _FakeResp(verdicts_json), None
    monkeypatch.setattr("agent_reasoning.ReAct.support.llm.llm_create_with_retry",
                        _fake)


def test_filter_full_flow_guidance(monkeypatch):
    """层1+层2 全判 false → 置信度 0 → 人工引导文本。"""
    _stub_llm(monkeypatch,
              '{"verdicts":[{"i":1,"supported":false},{"i":2,"supported":false}]}')
    out, info = G.grounding_filter(
        "答案里写4000个垫脚。这是编造的机理说明。",
        SOURCES)
    assert info["error"] is None
    assert info["action"] == "guidance"
    assert "人工" in out


def test_filter_full_flow_prune(monkeypatch):
    """多数句有支撑(带真实 quote),少数无 → 删句放行。"""
    _stub_llm(monkeypatch,
              '{"verdicts":[{"i":1,"supported":true,"quote":"6个调整垫脚"},'
              '{"i":2,"supported":true,"quote":"6个调整垫脚"},'
              '{"i":3,"supported":true,"quote":"6个调整垫脚"},'
              '{"i":4,"supported":false}]}')
    out, info = G.grounding_filter(
        "4M-2基座使用6个调整垫脚。垫脚用于调平。安装时需对称布置。这是编造的机理说明。",
        SOURCES)
    assert info["action"] == "pruned"
    assert info["removed"] == 1
    assert info["confidence"] == 0.75
    assert "编造" not in out


def test_filter_two_sentences_one_false_is_guidance(monkeypatch):
    """2 句答案删 1 句 → 置信度 0.5 < 0.6,整段引导(严格门控的边界后果)。"""
    _stub_llm(monkeypatch,
              '{"verdicts":[{"i":1,"supported":true,"quote":"6个调整垫脚"},'
              '{"i":2,"supported":false}]}')
    out, info = G.grounding_filter(
        "4M-2基座使用6个调整垫脚。这是编造的机理说明。",
        SOURCES)
    assert info["action"] == "guidance"
    assert info["confidence"] == 0.5


def test_filter_quote_mismatch_counts_false(monkeypatch):
    """判 true 但 quote 对不上原文 → 按 false 处理。"""
    _stub_llm(monkeypatch,
              '{"verdicts":[{"i":1,"supported":true,"quote":"不存在的引文"}]}')
    out, info = G.grounding_filter(
        "第一句编造内容。第二句也编造内容。",
        SOURCES)
    assert info["quote_failed"] == 1
    assert info["action"] == "guidance"  # 两句全 false
