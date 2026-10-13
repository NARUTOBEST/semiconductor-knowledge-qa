# -*- coding: utf-8 -*-
"""设备型号 token 抽取(mcp_servers/retrieval/devices.py)单测。

覆盖查询侧/入库侧同一规则:字母+数字型号、纯大写品牌、报警码/纯数字排除、
入库三源并集与 None 语义。
"""
from mcp_servers.retrieval.devices import chunk_device_tags, extract_device_tokens


def test_mixed_letter_digit_tokens():
    # D4i/I12/H4 类型号;分隔符规范化为单一 token
    assert "D4I" in extract_device_tokens("ASM-D4i设备保养手册")
    assert "I12" in extract_device_tokens("I12-GD-STDOP-241-E1 的操作")
    assert "H4" in extract_device_tokens("H4 - Description(机器描叙)")


def test_uppercase_brand_tokens():
    toks = extract_device_tokens("FUJI NXT 贴片机 SIPLACE 保养")
    assert {"FUJI", "NXT", "SIPLACE"} <= set(toks)


def test_pure_numbers_excluded():
    # 报警码 1420/2412 与型号数字 2200 均不产生 token(走 exact_code_recall 兜底)
    assert extract_device_tokens("Datacon 2200 固晶机 1420 报警怎么处理") == ["DATACON"]


def test_no_tokens():
    assert extract_device_tokens("固晶机 1420 报警是什么原因") == []
    assert extract_device_tokens("") == []
    assert extract_device_tokens(None) == []


def test_dedup_and_cap():
    toks = extract_device_tokens("NXT NXT NXT-IIc V12 M32", max_tokens=3)
    assert len(toks) == 3
    assert len(set(toks)) == 3


def test_chunk_tags_union_of_content_heading_stem():
    tags = chunk_device_tags("腔体压力设定值 0.8 Torr", source_stem="ESMA 维修手册")
    assert "ESMA" in tags
    # 正文里出现型号也能命中(书名没有的情况)
    tags2 = chunk_device_tags("适用于 Datacon 2200 platform 的固晶工艺")
    assert "DATACON" in tags2


def test_chunk_tags_none_when_empty():
    assert chunk_device_tags("腔体压力设定值 0.8 Torr", source_stem="工艺讲义") is None
