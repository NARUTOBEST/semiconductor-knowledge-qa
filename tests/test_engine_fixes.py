# -*- coding: utf-8 -*-
"""方案一(补录席位)/方案二A(行上下文)的单元测试(不依赖模型/网络)。"""
from types import SimpleNamespace

import pytest

import mcp_servers.retrieval.engine_api as E
import config as C


def _item(cid, score):
    return {"chunk_id": cid, "score": score}


def _pt(cid):
    return SimpleNamespace(payload={"chunk_id": cid, "content": f"row {cid}"})


class TestRescueSeat:
    """固定 top-K 的"弱末位让位"补录席位。"""

    def test_weak_tail_displaced_by_rescue(self, monkeypatch):
        monkeypatch.setattr(C, "RERANK_TAU", 0.5, raising=False)
        out = [_item("a", 0.9), _item("b", 0.8), _item("c", 0.7), _item("d", 0.3)]
        rescued = [_pt("r1"), _pt("r2")]
        new = E._rescue_seat(out, rescued)
        assert len(new) == 4                      # 席位数不变
        assert [d["chunk_id"] for d in new][:3] == ["a", "b", "c"]
        assert new[3]["chunk_id"] == "r1"          # 弱末位被补录块顶替
        assert new[3]["score"] <= 0.3              # 评分不高于被顶替者

    def test_strong_tail_kept(self, monkeypatch):
        monkeypatch.setattr(C, "RERANK_TAU", 0.5, raising=False)
        out = [_item("a", 0.9), _item("b", 0.8), _item("c", 0.7), _item("d", 0.6)]
        assert E._rescue_seat(out, [_pt("r1")]) == out   # 末位硬朗,不动

    def test_rescue_already_present(self, monkeypatch):
        monkeypatch.setattr(C, "RERANK_TAU", 0.5, raising=False)
        out = [_item("a", 0.9), _item("r1", 0.3)]
        assert E._rescue_seat(out, [_pt("r1")]) == out   # 补录块已在场

    def test_no_rescue(self):
        out = [_item("a", 0.9), _item("b", 0.3)]
        assert E._rescue_seat(out, []) is out


class TestSiblingPrefix:
    """方案二A:短表格行重排时前置同表相邻行。"""

    def test_off_by_default(self):
        pl = {"chunk_id": "x", "heading_path": "8.1 Alarms", "content": "row", "source_stem": "s"}
        assert E._sibling_prefix(pl) == ""

    def test_on_fetches_siblings(self, monkeypatch):
        monkeypatch.setattr(C, "RERANK_ROW_CONTEXT", True, raising=False)
        E._sib_cache.clear()
        pts = [SimpleNamespace(payload={"chunk_id": "s1", "content": "header row 1 2 3"}),
               SimpleNamespace(payload={"chunk_id": "x", "content": "self"}),
               SimpleNamespace(payload={"chunk_id": "s2", "content": "next row"})]

        class FakeQ:
            models = SimpleNamespace(
                Filter=lambda **kw: kw,
                FieldCondition=lambda **kw: kw,
                MatchValue=lambda **kw: kw,
            )

            @staticmethod
            def get_client():
                return SimpleNamespace(scroll=lambda **kw: (pts, None))

        monkeypatch.setattr(E, "Q", FakeQ, raising=False)
        pl = {"chunk_id": "x", "heading_path": "8.1 Alarms",
              "content": "short row", "source_stem": "doc1"}
        prefix = E._sibling_prefix(pl)
        assert "[同表其他行]" in prefix and "header row" in prefix
        assert "self" not in prefix                    # 自身行不进前缀
        assert E._sib_cache["x"] == prefix             # 已缓存

    def test_long_block_skipped(self, monkeypatch):
        monkeypatch.setattr(C, "RERANK_ROW_CONTEXT", True, raising=False)
        E._sib_cache.clear()
        pl = {"chunk_id": "y", "heading_path": "h", "content": "x" * 300, "source_stem": "s"}
        assert E._sibling_prefix(pl) == ""             # 长块不查邻居


class TestSourceScope:
    """检索域软收窄:目标文档推断 + 重排分加权。"""

    def test_base_stem_merges_split_suffix(self):
        assert E._base_stem("机器人magnatran_71_p1-5") == "机器人magnatran_71"
        assert E._base_stem("机器人magnatran_71_p501-794") == "机器人magnatran_71"
        assert E._base_stem("普通文档名") == "普通文档名"      # 无后缀不动

    def _fake_registry(self, monkeypatch):
        reg = {
            "晶圆CMP研磨机6DS-SP": {"晶圆CMP研磨机6DS-SP"},
            "双臂取晶圆机器人magnatran_71": {"双臂取晶圆机器人magnatran_71_p1-5",
                                             "双臂取晶圆机器人magnatran_71_p501-794"},
            "晶圆减薄strasbaugh Model 7AF": {"晶圆减薄strasbaugh Model 7AF"},
        }
        monkeypatch.setattr(E, "_stem_registry", reg, raising=False)

    def test_infer_confident_match(self, monkeypatch):
        self._fake_registry(monkeypatch)
        # 20261008 修复:base 先 lower 再切 token,大写型号(6DS-SP)才能成混合
        # token 计分;纯字母词(magnatran)仍被轮次9 严格护栏跳过,须整段 CJK 命中
        assert E.infer_target_stem("这台双臂取晶圆机器人整机的质保期是几年?") \
            == "双臂取晶圆机器人magnatran_71"
        assert E.infer_target_stem("我们厂想在 6DS-SP 上抛 300mm 晶圆") \
            == "晶圆CMP研磨机6DS-SP"

    def test_infer_pure_alpha_token_ignored(self, monkeypatch):
        self._fake_registry(monkeypatch)
        # 纯字母 token(magnatran)无数字且非全大写 → 不计分,CJK 段不完整 → None
        assert E.infer_target_stem("这台 magnatran 机器人整机的质保期是几年?") is None

    def test_infer_no_match(self, monkeypatch):
        self._fake_registry(monkeypatch)
        assert E.infer_target_stem("手册里 slurry 推荐的配比是多少?") is None
        assert E.infer_target_stem("") is None

    def test_infer_ambiguous(self, monkeypatch):
        self._fake_registry(monkeypatch)
        # 只命中公共词"晶圆"(2 分 < MIN_SCORE=4)→ 不确信
        assert E.infer_target_stem("晶圆怎么装?") is None

    def test_apply_scope_boost_and_resort(self, monkeypatch):
        monkeypatch.setattr(C, "SOURCE_SCOPE_BOOST", 1.35, raising=False)
        pts = [SimpleNamespace(payload={"source_stem": "别的手册A"}, id="a"),
               SimpleNamespace(payload={"source_stem": "目标_p1-500"}, id="b")]
        ranked = [(pts[0], 0.9), (pts[1], 0.8)]
        res = E._apply_source_scope(ranked, "目标")
        assert [p.id for p, _ in res] == ["b", "a"]     # 0.8*1.35 > 0.9
        assert res[0][1] == pytest.approx(1.08)
        assert E._apply_source_scope(ranked, "") == ranked  # 无目标零伤害


class TestNeighborExpand:
    """邻块扩展:top-N 命中块尾部拼接后接块内容。"""

    def test_appends_next_chunk(self, monkeypatch):
        monkeypatch.setattr(C, "RETRIEVAL_NEIGHBOR_CHARS", 500, raising=False)
        nxt = [SimpleNamespace(payload={"content": "下一页内容"})]

        class FakeQ:
            models = SimpleNamespace(
                Filter=lambda **kw: kw,
                FieldCondition=lambda **kw: kw,
                MatchValue=lambda **kw: kw,
            )

            @staticmethod
            def get_client():
                return SimpleNamespace(scroll=lambda **kw: (nxt, None))

        monkeypatch.setattr(E, "Q", FakeQ, raising=False)
        d = {"source_stem": "手册A", "chunk_index": 3, "content": "本块"}
        E._expand_neighbor(d)
        assert d["content"].startswith("本块")
        assert "[后接块] 下一页内容" in d["content"]

    def test_no_neighbor_kept_intact(self, monkeypatch):
        class FakeQ:
            models = SimpleNamespace(
                Filter=lambda **kw: kw,
                FieldCondition=lambda **kw: kw,
                MatchValue=lambda **kw: kw,
            )

            @staticmethod
            def get_client():
                return SimpleNamespace(scroll=lambda **kw: ([], None))

        monkeypatch.setattr(E, "Q", FakeQ, raising=False)
        d = {"source_stem": "手册A", "chunk_index": 3, "content": "本块"}
        E._expand_neighbor(d)
        assert d["content"] == "本块"                   # 无后接块原样

    def test_missing_index_kept_intact(self):
        d = {"source_stem": "手册A", "content": "本块"}
        assert E._expand_neighbor(d)["content"] == "本块"  # 缺 chunk_index 跳过
