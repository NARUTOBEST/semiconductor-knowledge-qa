# -*- coding: utf-8 -*-
"""轮次19 术语表字符串替换单测:词典构建/最长优先/大小写归一/开关与
容错直通/召回查询构造接线(不依赖网络与模型)。"""
import json
import os

import pytest

import config as C
import mcp_servers.retrieval.engine_api as E
import mcp_servers.retrieval.glossary as G


@pytest.fixture
def gl_file(tmp_path, monkeypatch):
    data = {
        "version": "test",
        "zh2en": {
            "兆声波清洗": "megasonic cleaning",
            "兆声清洗": "megasonic cleaning",
            "埋氧层": "buried oxide",
            "放置重复精度": "Placement Repeatability",
        },
        "en2en": {
            "box": "buried oxide",
            "buried oxide layer": "buried oxide",
        },
    }
    p = tmp_path / "glossary.json"
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(C, "GLOSSARY_FILE", str(p), raising=False)
    G._cache.update(mtime=0.0, path=None)   # 强制重载
    yield p
    G._cache.update(mtime=0.0, path=None)


def _on(monkeypatch, flag=True):
    monkeypatch.setattr(C, "RETRIEVAL_GLOSSARY_REPLACE", flag, raising=False)


# ==================== zh -> en 替换 ====================
def test_zh_replaced_with_en_term(gl_file, monkeypatch):
    _on(monkeypatch)
    out = G.apply("SOI 硅纳米膜转移前释放埋氧层用的刻蚀液")
    assert "buried oxide" in out
    assert "埋氧层" not in out


def test_longest_match_first(gl_file, monkeypatch):
    _on(monkeypatch)
    # "兆声波清洗"(长键)不被 "兆声清洗"(短键)切碎;替换后与相邻 CJK 补空格
    out = G.apply("兆声波清洗的频率是多少")
    assert out == "megasonic cleaning 的频率是多少"
    assert "兆声" not in out


def test_no_match_passthrough(gl_file, monkeypatch):
    _on(monkeypatch)
    q = "今天天气怎么样"
    assert G.apply(q) == q


def test_switch_off_passthrough(gl_file, monkeypatch):
    _on(monkeypatch, False)
    q = "埋氧层刻蚀"
    assert G.apply(q) == q


def test_missing_file_passthrough(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "GLOSSARY_FILE",
                        str(tmp_path / "nope.json"), raising=False)
    monkeypatch.setattr(C, "RETRIEVAL_GLOSSARY_REPLACE", True, raising=False)
    G._cache.update(mtime=0.0, path=None)
    q = "埋氧层"
    assert G.apply(q) == q          # 文件缺失零伤害
    assert G.stats()["zh"] == 0


def test_corrupt_file_passthrough(tmp_path, monkeypatch):
    p = tmp_path / "bad.json"
    p.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(C, "GLOSSARY_FILE", str(p), raising=False)
    monkeypatch.setattr(C, "RETRIEVAL_GLOSSARY_REPLACE", True, raising=False)
    G._cache.update(mtime=0.0, path=None)
    assert G.apply("埋氧层") == "埋氧层"


# ==================== en 别名归一(大小写不敏感) ====================
def test_en_alias_case_insensitive(gl_file, monkeypatch):
    _on(monkeypatch)
    assert G.apply("BOX release etchant") == "buried oxide release etchant"
    assert G.apply("buried oxide layer strip") == "buried oxide strip"
    assert G.apply("box layer") == "buried oxide layer"   # 大小写不敏感


def test_mixed_zh_en_uses_zh_map(gl_file, monkeypatch):
    _on(monkeypatch)
    out = G.apply("SOI 释放埋氧层(BOX)")
    assert "buried oxide" in out          # zh 命中 + BOX 原样(中文句走 zh2en)


def test_short_acronym_targets_filtered(gl_file, monkeypatch):
    # zh->en 目标 ≤2 字符(如 光刻胶->PR)不生效:中文稠密语义不值得换两个字母
    data = json.loads(gl_file.read_text(encoding="utf-8"))
    data["zh2en"]["光刻胶"] = "PR"
    gl_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    G._cache.update(mtime=0.0, path=None)   # 触发热重载
    _on(monkeypatch)
    assert G.apply("光刻胶选择比") == "光刻胶选择比"
    assert G.apply("兆声清洗") == "megasonic cleaning"   # 正常条目不受影响


def test_en2en_acronym_targets_filtered(gl_file, monkeypatch):
    # 批译把常见英文词记成缩写别名(release->RLS/wafer->WAF):归一不许把
    # 完整词逆向缩成行话(毁稠密语义);正常别名(BOX->buried oxide)不受影响
    data = json.loads(gl_file.read_text(encoding="utf-8"))
    data["en2en"]["release"] = "RLS"
    data["en2en"]["etching"] = "ETCH"
    gl_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    G._cache.update(mtime=0.0, path=None)
    _on(monkeypatch)
    assert G.apply("release the wafer") == "release the wafer"
    assert G.apply("etching solution") == "etching solution"
    assert G.apply("BOX release") == "buried oxide release"


# ==================== _recall_queries 接线 ====================
def test_recall_queries_glossary_before_translate(gl_file, monkeypatch):
    _on(monkeypatch)
    monkeypatch.setattr(C, "RETRIEVAL_QUERY_TRANSLATE", True, raising=False)
    # 翻译吃的是替换后的原句(库内术语直接进翻译)
    monkeypatch.setattr(E, "_translate_query",
                        lambda q: "EN<%s>" % q, raising=False)
    out = E._recall_queries("释放埋氧层的刻蚀液")
    assert len(out) == 2
    assert "buried oxide" in out[0]                       # 原句路被替换
    assert "buried oxide" in out[1] and out[1].startswith("EN<")


def test_translation_output_not_re_replaced(gl_file, monkeypatch):
    _on(monkeypatch)
    monkeypatch.setattr(C, "RETRIEVAL_QUERY_TRANSLATE", True, raising=False)
    # 译文(纯英文)不被 en2en 二次替换——实测归一会把译文 release 缩成 RLS
    monkeypatch.setattr(E, "_translate_query",
                        lambda q: "release the BOX layer", raising=False)
    out = E._recall_queries("埋氧层")
    assert out[1] == "release the BOX layer"


def test_recall_queries_translate_fail_single_path(gl_file, monkeypatch):
    _on(monkeypatch)
    monkeypatch.setattr(C, "RETRIEVAL_QUERY_TRANSLATE", True, raising=False)
    monkeypatch.setattr(E, "_translate_query", lambda q: "", raising=False)
    out = E._recall_queries("埋氧层")
    assert out == ["buried oxide"]                        # 翻译空,单路仍替换


def test_recall_queries_no_glossary_file(gl_file, monkeypatch):
    monkeypatch.setattr(C, "RETRIEVAL_GLOSSARY_REPLACE", True, raising=False)
    monkeypatch.setattr(C, "RETRIEVAL_QUERY_TRANSLATE", False, raising=False)
    monkeypatch.delattr(C, "GLOSSARY_FILE", raising=False)
    G._cache.update(mtime=0.0, path=None)
    q = "原始查询保持不变"
    assert E._recall_queries(q) == [q]                    # 全链直通
