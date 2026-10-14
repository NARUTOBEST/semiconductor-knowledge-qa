# -*- coding: utf-8 -*-
"""复合题子检索席位绑定(20261014)测试。

用户口径:复合题拆成子问题后,每个子问题各自检索的 top-4 必须"绑定"进
上下文,不能照单问题口径只给总 top-4。覆盖:
  - service._speculative_search:子路 top-PIN_K 打 sub_pin、去重、cap 不截席位;
  - (回归)call_with_resilience (result, err) 元组解包——20261013 版因把元组
    当 dict 判定,整段子检索合并从未执行(死代码),此处钉死;
  - raglite stream._pin_select:常规 top-cap 之外保留全部席位块(去重);
  - compressor.compress_chunks:sub_pin 块不受 MAX_CHUNKS 截断且返回列表
    与调用方输入等长对齐;
  - nodes._pin_fill:sources 事件常规配额之外追加席位来源;
  - extract_doc_sources:sub_pin 透传进来源条目。
不连真实 LLM / MCP(全打桩)。
"""
import chat.service as svc
import config as C


def _mk(cid, stem="bookA", score=1.0, content="内容" * 100):
    return {"chunk_id": cid, "source_stem": stem, "page_num": 1,
            "score": score, "content": content}


class _SyncPool:
    """同步池:submit 直接在当前线程执行并回填结果(便于断言)。"""

    def submit(self, fn, *args, **kwargs):
        import concurrent.futures as cf
        fut = cf.Future()
        try:
            fut.set_result(fn(*args, **kwargs))
        except Exception as e:  # noqa: BLE001
            fut.set_exception(e)
        return fut


def _patch_env(monkeypatch, main_items, sub_results, calls):
    """打桩投机检索的全部外部依赖,返回 future.result() 的值。"""
    import agent_reasoning.ReAct.support.tool_resilience as tr
    import agent_reasoning.ReAct.support.tool_pool as tp
    import tools

    class _Spec:
        retry_times = 0

    class _Reg:
        def get(self, name):
            return _Spec()

    def _fake_cwr(name, args, spec, **kwargs):
        calls.append(dict(args))
        if "with_scope_info" in args and args.get("with_scope_info"):
            return {"items": list(main_items),
                    "scope": {"inferred_stem": "", "locked_stem": ""}}, None
        # 子路 with_scope_info=False:引擎回裸列表(20261014 实测形状),钉死
        sq = args["query"]
        for frag, items in sub_results.items():
            if frag in sq:
                return list(items), None
        return [], None

    monkeypatch.setattr(tools, "registry", _Reg())
    monkeypatch.setattr(tools, "dispatch", lambda *a, **k: None)
    monkeypatch.setattr(tr, "call_with_resilience", _fake_cwr)
    monkeypatch.setattr(tp, "get_pool", lambda: _SyncPool())


COMPOSITE = "6DS-SP的主轴转速是多少?设备的外形尺寸是多少?"


class TestSpeculativePin:
    def test_sub_topk_pinned_and_deduped(self, monkeypatch):
        main = [_mk(f"m{i}", score=1.0 - i * 0.1) for i in range(5)]
        # 子问题1:前2块与主路重复,另2块新;子问题2:4块全新
        sub1 = [_mk("m0"), _mk("m1"), _mk("s1a"), _mk("s1b")]
        sub2 = [_mk(f"s2{i}") for i in range(4)]
        calls = []
        _patch_env(monkeypatch, main, {"主轴转速": sub1, "外形尺寸": sub2}, calls)
        out, err = svc._speculative_search(COMPOSITE).result()
        assert err is None
        ids = [it["chunk_id"] for it in out["items"]]
        assert ids[:5] == [m["chunk_id"] for m in main]  # 主路原样在前
        for cid in ("s1a", "s1b", "s20", "s21", "s22", "s23"):
            assert ids.count(cid) == 1  # 去重且全部保留
        pinned = {it["chunk_id"] for it in out["items"] if it.get("sub_pin")}
        assert pinned == {"s1a", "s1b", "s20", "s21", "s22", "s23"}
        assert not any(it.get("sub_pin") for it in out["items"][:5])

    def test_sub_pin_k_limit(self, monkeypatch):
        main = [_mk("m0")]
        sub = [_mk(f"x{i}") for i in range(10)]  # 子路给 10 块,只保 top-PIN_K
        calls = []
        _patch_env(monkeypatch, main, {"主轴转速": sub}, calls)
        out, _ = svc._speculative_search(COMPOSITE).result()
        pinned = [it["chunk_id"] for it in out["items"] if it.get("sub_pin")]
        assert pinned == [f"x{i}" for i in range(C.RETRIEVAL_SUBQUERY_PIN_K)]

    def test_cap_never_cuts_pins(self, monkeypatch):
        main = [_mk(f"m{i}", score=1.0 - i * 0.1) for i in range(5)]
        sub = [_mk(f"x{i}") for i in range(4)]
        calls = []
        _patch_env(monkeypatch, main, {"主轴转速": sub, "外形尺寸": sub}, calls)
        monkeypatch.setattr(C, "RETRIEVAL_SUBQUERY_MERGE_CAP", 2)  # 极小 cap
        out, _ = svc._speculative_search(COMPOSITE).result()
        ids = [it["chunk_id"] for it in out["items"]]
        for i in range(4):
            assert f"x{i}" in ids  # 席位块不被 cap 截掉
        assert len(ids) == len(set(ids))

    def test_tuple_unpack_regression(self, monkeypatch):
        """20261013 死代码回归钉子:主/子路都按 (result, err) 元组解包。"""
        main = [_mk("m0")]
        sub = [_mk("s0")]
        calls = []
        _patch_env(monkeypatch, main, {"主轴转速": sub}, calls)
        out, _ = svc._speculative_search(COMPOSITE).result()
        assert any(it["chunk_id"] == "s0" for it in out["items"])
        # 主路带 with_scope_info,子路不带
        assert sum(1 for c in calls if c.get("with_scope_info")) == 1

    def test_single_question_no_sub(self, monkeypatch):
        main = [_mk(f"m{i}") for i in range(5)]
        calls = []
        _patch_env(monkeypatch, main, {}, calls)
        out, _ = svc._speculative_search("6DS-SP的主轴转速是多少?").result()
        assert [it["chunk_id"] for it in out["items"]] == [
            m["chunk_id"] for m in main]
        assert not any(it.get("sub_pin") for it in out["items"])


class TestPinSelect:
    def test_pins_survive_beyond_cap(self):
        from agent_reasoning.raglite.stream import _pin_select
        main = [{"chunk_id": f"m{i}", "content": "x"} for i in range(6)]
        pins = [{"chunk_id": f"p{i}", "sub_pin": True} for i in range(4)]
        sel = _pin_select(main + pins, 4)
        ids = [c["chunk_id"] for c in sel]
        assert ids == ["m0", "m1", "m2", "m3", "p0", "p1", "p2", "p3"]

    def test_pin_dup_dropped(self):
        from agent_reasoning.raglite.stream import _pin_select
        main = [{"chunk_id": "m0", "content": "x"}]
        pins = [{"chunk_id": "m0", "sub_pin": True},
                {"chunk_id": "p0", "sub_pin": True}]
        sel = _pin_select(main + pins, 4)
        assert [c["chunk_id"] for c in sel] == ["m0", "p0"]


class TestCompressPin:
    def test_pins_survive_max_chunks_and_alignment(self, monkeypatch):
        import compressor as comp
        monkeypatch.setattr(comp.C, "CONTEXT_COMPRESS_MAX_CHUNKS", 3)
        # 桩 _compress:回显每个块的 chunk_id 作关键句
        monkeypatch.setattr(comp, "_compress",
                            lambda chunks, q, **k: [c["chunk_id"] for c in chunks])
        full = ([{"chunk_id": f"m{i}", "content": "x" * 50} for i in range(6)]
                + [{"chunk_id": f"p{i}", "content": "x" * 50, "sub_pin": True}
                   for i in range(3)])
        picked = comp.compress_chunks(full, "q")
        assert picked is not None and len(picked) == len(full)  # 等长契约
        assert picked[:3] == ["m0", "m1", "m2"]      # 常规配额
        assert picked[3:6] != ["m3", "m4", "m5"]     # m3-m5 被截,回退截断内容
        assert picked[6:] == ["p0", "p1", "p2"]      # 席位块全保留
        for i in (3, 4, 5):
            assert picked[i] != f"m{i}"  # 回退为截断内容而非关键句

    def test_no_pins_passthrough_alignment(self, monkeypatch):
        import compressor as comp
        monkeypatch.setattr(comp.C, "CONTEXT_COMPRESS_MAX_CHUNKS", 3)
        monkeypatch.setattr(comp, "_compress",
                            lambda chunks, q, **k: [c["chunk_id"] for c in chunks])
        full = [{"chunk_id": f"m{i}", "content": "x" * 50} for i in range(5)]
        picked = comp.compress_chunks(full, "q")
        # 超配额块回退截断内容(非关键句),列表仍与输入等长
        assert len(picked) == 5
        assert picked[:3] == ["m0", "m1", "m2"]
        assert picked[3].startswith("x") and picked[4].startswith("x")


class TestPinFill:
    def test_appends_pins_beyond_limit(self):
        from agent_reasoning.ReAct.core.nodes import _pin_fill
        items = [{"chunk_id": f"m{i}"} for i in range(10)]
        items.append({"chunk_id": "p0", "sub_pin": True})
        out = _pin_fill(items, 8)
        assert [s["chunk_id"] for s in out] == [
            *(f"m{i}" for i in range(8)), "p0"]

    def test_pin_within_limit_not_duplicated(self):
        from agent_reasoning.ReAct.core.nodes import _pin_fill
        items = [{"chunk_id": "p0", "sub_pin": True}] + [
            {"chunk_id": f"m{i}"} for i in range(5)]
        out = _pin_fill(items, 8)
        assert [s["chunk_id"] for s in out] == ["p0", *(f"m{i}" for i in range(5))]


class TestSourcePinPropagation:
    def test_extract_doc_sources_keeps_sub_pin(self):
        from tools.mcp_policies import extract_doc_sources
        res = {"items": [_mk("m0"), dict(_mk("p0"), sub_pin=True)]}
        out = extract_doc_sources(res)
        assert out[0].get("sub_pin") is None
        assert out[1].get("sub_pin") is True
        assert out[1]["chunk_id"] == "p0"
