# -*- coding: utf-8 -*-
"""方案2(20261015):复合题主+子问题合并单次引擎调用(search_batch)测试。

覆盖:
  - service._speculative_search 批量快速路径:一次 search_batch 调用,
    main+subs 分组结果经同一席位合并逻辑打 sub_pin,不再逐子问题调用;
  - scope_stem/scope_hard 透传进批量调用;
  - 批量失败(err)→ 回退逐子问题 search_text 路径(行为与 20261014 版一致);
  - 旧引擎未注册 search_batch → 同样回退;
  - main 裸列表形状归一;
  - engine_api.search_batch:主路异常向上抛、单子问题异常降级 [](不拖垮主结果)。
不连真实 LLM / MCP / GPU(全打桩)。
"""
import chat.service as svc
import config as C

COMPOSITE = "6DS-SP的主轴转速是多少?设备的外形尺寸是多少?"


def _mk(cid, stem="bookA", score=1.0, content="内容" * 100):
    return {"chunk_id": cid, "source_stem": stem, "page_num": 1,
            "score": score, "content": content}


class _SyncPool:
    def submit(self, fn, *args, **kwargs):
        import concurrent.futures as cf
        fut = cf.Future()
        try:
            fut.set_result(fn(*args, **kwargs))
        except Exception as e:  # noqa: BLE001
            fut.set_exception(e)
        return fut


class _Spec:
    retry_times = 0


def _patch_env(monkeypatch, *, batch=None, batch_err=None, main_items=None,
               sub_results=None, tools_names=("search_text", "search_batch"),
               calls):
    """打桩投机检索依赖。batch:search_batch 的返回;batch_err 非 None 时
    search_batch 返回该错误;main_items/sub_results 供逐路回退路径使用。"""
    import agent_reasoning.ReAct.support.tool_resilience as tr
    import agent_reasoning.ReAct.support.tool_pool as tp
    import tools

    class _Reg:
        def get(self, name):
            return _Spec() if name in tools_names else None

    def _fake_cwr(name, args, spec, **kwargs):
        calls.append({"tool": name, **args})
        if name == "search_batch":
            if batch_err is not None:
                return None, batch_err
            return batch, None
        if args.get("with_scope_info"):
            return {"items": list(main_items or []),
                    "scope": {"inferred_stem": "", "locked_stem": ""}}, None
        sq = args["query"]
        for frag, items in (sub_results or {}).items():
            if frag in sq:
                return list(items), None
        return [], None

    monkeypatch.setattr(tools, "registry", _Reg())
    monkeypatch.setattr(tools, "dispatch", lambda *a, **k: None)
    monkeypatch.setattr(tr, "call_with_resilience", _fake_cwr)
    monkeypatch.setattr(tp, "get_pool", lambda: _SyncPool())


class TestBatchFastPath:
    def test_one_batch_call_merges_pins(self, monkeypatch):
        main = [_mk(f"m{i}", score=1.0 - i * 0.1) for i in range(5)]
        sub1 = [_mk("m0"), _mk("s1a"), _mk("s1b")]      # 首块与主路重复
        sub2 = [_mk(f"s2{i}") for i in range(4)]
        batch = {"main": {"items": list(main),
                          "scope": {"inferred_stem": ""}},
                 "subs": {svc._split_composite_query(COMPOSITE)[0]: sub1,
                          svc._split_composite_query(COMPOSITE)[1]: sub2}}
        calls = []
        _patch_env(monkeypatch, batch=batch, main_items=main,
                   sub_results={}, calls=calls)
        out, err = svc._speculative_search(COMPOSITE).result()
        assert err is None
        # 只有 1 次 search_batch,零逐子问题 search_text
        assert [c["tool"] for c in calls] == ["search_batch"]
        ids = [it["chunk_id"] for it in out["items"]]
        assert ids[:5] == [m["chunk_id"] for m in main]  # 主路原样在前
        for cid in ("s1a", "s1b", "s20", "s21", "s22", "s23"):
            assert ids.count(cid) == 1
        pinned = {it["chunk_id"] for it in out["items"] if it.get("sub_pin")}
        assert pinned == {"s1a", "s1b", "s20", "s21", "s22", "s23"}

    def test_scope_args_propagated(self, monkeypatch):
        batch = {"main": {"items": [_mk("m0")]}, "subs": {}}
        calls = []
        _patch_env(monkeypatch, batch=batch, calls=calls)
        svc._speculative_search(COMPOSITE, scope_stem="6DS-SP手册",
                                scope_hard=True).result()
        (call,) = calls
        assert call["tool"] == "search_batch"
        assert call["scope_stem"] == "6DS-SP手册"
        assert call["scope_hard"] is True

    def test_main_bare_list_normalized(self, monkeypatch):
        # 旧引擎主路可能回裸列表:归一为 {"items": [...]} 再合并
        batch = {"main": [_mk("m0")],
                 "subs": {svc._split_composite_query(COMPOSITE)[0]: [_mk("s0")]}}
        calls = []
        _patch_env(monkeypatch, batch=batch, calls=calls)
        out, _ = svc._speculative_search(COMPOSITE).result()
        ids = [it["chunk_id"] for it in out["items"]]
        assert ids == ["m0", "s0"]
        assert any(it.get("sub_pin") for it in out["items"])

    def test_batch_sub_missing_key_ok(self, monkeypatch):
        # subs 里缺某个子问题的键(引擎侧单子路故障降级)→ 该子问题无席位,不炸
        main = [_mk("m0")]
        batch = {"main": {"items": list(main)}, "subs": {}}
        calls = []
        _patch_env(monkeypatch, batch=batch, calls=calls)
        out, err = svc._speculative_search(COMPOSITE).result()
        assert err is None
        assert [it["chunk_id"] for it in out["items"]] == ["m0"]
        assert not any(it.get("sub_pin") for it in out["items"])


class TestBatchFallback:
    def test_batch_err_falls_back_per_sub(self, monkeypatch):
        """批量失败 → 回退逐子问题路径,行为与 20261014 版一致。"""
        from agent_reasoning.ReAct.support.tool_resilience import ToolCallError
        main = [_mk(f"m{i}", score=1.0 - i * 0.1) for i in range(5)]
        sub = [_mk("s0")]
        calls = []
        _patch_env(monkeypatch, batch_err=ToolCallError(
            stage=None, kind=None, message="x", tool="search_batch"),
            main_items=main, sub_results={"主轴转速": sub}, calls=calls)
        out, err = svc._speculative_search(COMPOSITE).result()
        assert err is None
        tools_used = [c["tool"] for c in calls]
        assert tools_used[0] == "search_batch"          # 先试批量
        assert tools_used.count("search_text") >= 2     # 主+子逐路回退
        ids = [it["chunk_id"] for it in out["items"]]
        assert "s0" in ids
        assert any(it.get("sub_pin") for it in out["items"] if it["chunk_id"] == "s0")

    def test_no_batch_tool_falls_back(self, monkeypatch):
        """旧引擎未注册 search_batch → 直接走逐路路径。"""
        main = [_mk("m0")]
        sub = [_mk("s0")]
        calls = []
        _patch_env(monkeypatch, main_items=main,
                   sub_results={"主轴转速": sub},
                   tools_names=("search_text",), calls=calls)
        out, _ = svc._speculative_search(COMPOSITE).result()
        # 主路 + 2 子问题逐路回退(COMPOSITE 拆出 2 个子问题)
        assert [c["tool"] for c in calls] == ["search_text"] * 3
        assert any(it["chunk_id"] == "s0" for it in out["items"])

    def test_flag_off_falls_back(self, monkeypatch):
        """RETRIEVAL_SUBQUERY_BATCH=0 → 回退逐路路径。"""
        main = [_mk("m0")]
        calls = []
        _patch_env(monkeypatch, main_items=main, sub_results={}, calls=calls)
        monkeypatch.setattr(C, "RETRIEVAL_SUBQUERY_BATCH", False)
        out, _ = svc._speculative_search(COMPOSITE).result()
        assert all(c["tool"] == "search_text" for c in calls)


class TestEngineSearchBatch:
    def test_main_error_propagates(self, monkeypatch):
        import mcp_servers.retrieval.engine_api as ea
        def _boom(*a, **k):
            raise RuntimeError("engine down")
        monkeypatch.setattr(ea, "search_text", _boom)
        import pytest
        with pytest.raises(RuntimeError):
            ea.search_batch("q", subs=["s1"])

    def test_sub_error_degrades_to_empty(self, monkeypatch):
        import mcp_servers.retrieval.engine_api as ea
        calls = []

        def _fake(query, **k):
            calls.append(query)
            if query == "s1":
                raise RuntimeError("sub boom")
            if k.get("with_scope_info"):
                return {"items": [_mk("m0")], "scope": {}}
            return [_mk("s0")]

        monkeypatch.setattr(ea, "search_text", _fake)
        out = ea.search_batch("main q", subs=["s1", "s2"])
        assert out["subs"]["s1"] == []            # 单子路失败降级空列表
        assert [it["chunk_id"] for it in out["subs"]["s2"]] == ["s0"]
        assert [it["chunk_id"] for it in out["main"]["items"]] == ["m0"]
        # 主路 with_scope_info=True,子路 False
        assert calls[0] == "main q"

    def test_subs_cap_and_blank_filtered(self, monkeypatch):
        import mcp_servers.retrieval.engine_api as ea
        monkeypatch.setattr(ea, "search_text",
                            lambda q, **k: [] if not k.get("with_scope_info")
                            else {"items": [], "scope": {}})
        out = ea.search_batch("q", subs=["", "  ", "a", "b", "c", "d", "e"])
        assert set(out["subs"].keys()) == {"a", "b", "c", "d"}  # ≤4 且去空白
