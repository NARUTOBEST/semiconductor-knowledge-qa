# -*- coding: utf-8 -*-
"""源限定(轮次21,负证据题治理)测试。

覆盖:
  - 检测正则:题面限定词(手册里/书里/说明书里)命中,普通问题不命中;
  - Path A 直推:题面含设备 token 且引擎推断确信 → 锁定 scope 硬模式,
    不反问,单轮透传 _speculative_search/_run_tier;
  - Path B 反问:直推失败 → 无选项 clarify 事件 + pending 登记;应答轮
    消解出 stem 后 scope+hard 透传;
  - 引擎硬过滤:query_text 各路 prefetch 带 source_stem MatchAny 过滤,
    硬模式 τ 回填;
  - 作答口径:raglite/react 在 scope 锁定时追加《stem》限定 system 消息;
  - 质检豁免:scoped + search_count==0 → passed 不升级。
不连真实 LLM / GPU(全部打桩)。
"""
import sys

import pytest
from unittest.mock import patch, MagicMock

import chat.service as svc  # noqa: F401  确保双模块实例存在,清理用
import server.chat.service as svc2

_DONE = {"type": "done", "trace": {},
         "retrieval_max_score": 0.8, "search_count": 1}


def _react_stub(calls):
    def _gen(message, history, **k):
        calls.append({"message": message, **k})
        yield {"type": "assistant_message", "content": "答案。"}
        yield _DONE
    return _gen


@pytest.fixture(autouse=True)
def _clean():
    """_SCOPE_PENDING/_SCOPE_LOCKS 是模块级全局:用例间互清。
    chat.service 与 server.chat.service 是两个独立模块实例,都要清。"""
    for mod in (svc, sys.modules.get("chat.service"), svc2):
        if mod is not None:
            mod._SCOPE_PENDING.clear()
            mod._SCOPE_LOCKS.clear()
    yield
    for mod in (svc, sys.modules.get("chat.service"), svc2):
        if mod is not None:
            mod._SCOPE_PENDING.clear()
            mod._SCOPE_LOCKS.clear()


class TestBoundedCorpusDetection:
    def test_bounded_phrases_hit(self):
        for q in ["手册里有没有蓝宝石衬底的减薄配方?",
                  "书里兆声清洗的频率标称值是多少?",
                  "说明书里 Mx 设备配套的回流焊炉温曲线是怎么给的?",
                  "资料中关于 lockout 的程序是什么?"]:
            assert svc._BOUNDED_CORPUS_RE.search(q), q

    def test_plain_questions_miss(self):
        for q in ["TMA 的存储温度是多少", "ALD 原理是什么",
                  "介绍一下 Mi30 的上电流程", "你好"]:
            assert not svc._BOUNDED_CORPUS_RE.search(q), q


class TestSourceLockDirectPath:
    """Path A:题面含设备 token + 引擎直推确信 → 锁定不反问。"""

    Q = "手册里 6DS-SP 支持哪些载具尺寸?"

    def test_direct_lock_passes_scope_hard(self, monkeypatch):
        import server.chat.service as s
        monkeypatch.setattr(s.C, "SOURCE_LOCK_ENABLED", True)
        monkeypatch.setattr(s, "_infer_stem_via_engine",
                            lambda q: "晶圆CMP研磨机Strasbaugh 6DS-SP")
        cap_spec, cap_tier = [], []
        entry = {"need_clarify": False, "tier": "raglite",
                 "confidence": 0.9, "source": "rule"}

        def fake_spec(q, scope_stem="", scope_hard=False):
            cap_spec.append((q, scope_stem, scope_hard))
            return None

        with patch.object(s, "_speculative_search", side_effect=fake_spec), \
             patch.object(s, "classify_and_clarify", return_value=entry), \
             patch.object(s, "run_raglite", _react_stub(cap_tier)), \
             patch.object(s, "quality_check",
                          return_value={"verdict": "passed", "warnings": []}):
            ev = list(s.react_stream(self.Q, [], thread_id="t",
                                     username="alice"))
        assert not any(e["type"] == "clarify" for e in ev)   # 直推,不反问
        # 投机检索带 scope + 硬模式
        assert cap_spec and cap_spec[0][1] == "晶圆CMP研磨机Strasbaugh 6DS-SP"
        assert cap_spec[0][2] is True
        # tier 下游同样透传(升级/重做轮同参)
        assert cap_tier and cap_tier[0]["scope_stem"] == \
            "晶圆CMP研磨机Strasbaugh 6DS-SP"
        assert cap_tier[0]["scope_hard"] is True

    def test_infer_fail_clarifies_with_no_options(self, monkeypatch):
        """Path B:直推失败 → 无选项反问 + pending 登记(应答轮可消解)。"""
        import server.chat.service as s
        monkeypatch.setattr(s.C, "SOURCE_LOCK_ENABLED", True)
        monkeypatch.setattr(s.C, "CLARIFY_ENABLED", True)
        monkeypatch.setattr(s.C, "CLARIFY_OFFER_OPTIONS", False)
        monkeypatch.setattr(s, "_infer_stem_via_engine", lambda q: "")
        with patch.object(s, "_speculative_search") as spec, \
             patch.object(s, "classify_and_clarify") as router, \
             patch.object(s, "run_raglite", _react_stub([])):
            ev = list(s.react_stream("书里兆声清洗的频率是多少?", [],
                                     thread_id="t", username="bob"))
        cl = next(e for e in ev if e["type"] == "clarify")
        assert cl["options"] == []                    # 无选项口径
        assert "哪份资料" in cl["question"]
        assert router.assert_not_called is not None and not spec.called
        (pend,) = s._SCOPE_PENDING.values()
        assert pend["message"] == "书里兆声清洗的频率是多少?"
        assert pend["choices"] == []

    def test_answer_round_resolved_stem_transparent(self, monkeypatch):
        """应答轮:自由回复消解出 stem → scope+hard 透传到投机检索与 tier。"""
        import server.chat.service as s
        monkeypatch.setattr(s.C, "SOURCE_LOCK_ENABLED", True)
        monkeypatch.setattr(s.C, "CLARIFY_OFFER_OPTIONS", False)
        q = "手册里有没有蓝宝石衬底的减薄配方?"
        s._register_clarify_pending("carol", [], q, [])
        cap_spec, cap_tier = [], []
        entry = {"need_clarify": False, "tier": "raglite",
                 "confidence": 0.9, "source": "rule"}
        with patch.object(s, "_resolve_free_reply",
                          return_value=("晶圆减薄strasbaugh Model 7AF",
                                        "晶圆减薄strasbaugh Model 7AF")), \
             patch.object(s, "_speculative_search",
                          side_effect=lambda q, scope_stem="", scope_hard=False:
                              cap_spec.append((q, scope_stem, scope_hard))), \
             patch.object(s, "classify_and_clarify", return_value=entry), \
             patch.object(s, "run_raglite", _react_stub(cap_tier)), \
             patch.object(s, "quality_check",
                          return_value={"verdict": "passed", "warnings": []}):
            list(s.react_stream("晶圆减薄strasbaugh Model 7AF",
                                [{"role": "user", "content": q},
                                 {"role": "assistant", "content": "哪份资料?"}],
                                thread_id="t", username="carol"))
        assert cap_spec and cap_spec[0][1] == "晶圆减薄strasbaugh Model 7AF"
        assert cap_spec[0][2] is True                 # 锁定即硬模式
        assert cap_tier and cap_tier[0]["scope_hard"] is True

    def test_source_lock_off_no_detection(self, monkeypatch):
        """开关关闭:回退轮次20 行为,有限定词也不检测不反问(投机检索照常)。"""
        import server.chat.service as s
        monkeypatch.setattr(s.C, "SOURCE_LOCK_ENABLED", False)
        entry = {"need_clarify": False, "tier": "simple",
                 "confidence": 0.9, "source": "rule"}
        calls = []
        with patch.object(s, "_speculative_search") as spec, \
             patch.object(s, "classify_and_clarify", return_value=entry), \
             patch.object(s, "run_simple", _react_stub(calls)):
            ev = list(s.react_stream("书里兆声清洗的频率是多少?", [],
                                     thread_id="t", username="dave"))
        assert not any(e["type"] == "clarify" for e in ev)
        # 不带 scope(轮次20 行为:无锁定无加权)
        for call in spec.call_args_list:
            assert not call.kwargs.get("scope_stem")


class TestEngineHardFilter:
    def test_query_text_scope_filter(self, monkeypatch):
        """硬过滤:各路 prefetch 带 source_stem MatchAny;无 scope_stems 不过滤。"""
        from mcp_servers.retrieval import query as Q
        captured = {}

        class FakePoints:
            def __init__(self):
                self.points = []

        class FakeClient:
            def query_points(self, **kw):
                captured.setdefault("calls", []).append(kw)
                return FakePoints()

        monkeypatch.setattr(Q, "get_client", lambda: FakeClient())

        class FakeTE:
            def encode(self, qs):
                dense = [MagicMock()] * len(qs)
                sparse = [MagicMock()] * len(qs)
                return dense, sparse
        monkeypatch.setattr(Q, "_te", FakeTE())
        Q.query_text("载具尺寸", k=5,
                     scope_stems=["A_p1-500", "A_p501-1000"])
        (call,) = captured["calls"]
        pf = call["prefetch"]
        assert len(pf) >= 2
        for p in pf[:2]:
            cond = p.filter.must[0]
            assert cond.key == "source_stem"
            assert set(cond.match.any) == {"A_p1-500", "A_p501-1000"}
        # 无 scope:filter 为 None
        Q.query_text("载具尺寸", k=5)
        (call2,) = captured["calls"][1:]
        assert call2["prefetch"][0].filter is None

    def test_search_text_hard_backfill(self, monkeypatch):
        """硬模式 τ 回填:τ 截断不足 k 时族内 top-k 补齐。"""
        import mcp_servers.retrieval.engine_api as E

        pts = [MagicMock() for _ in range(6)]
        for i, p in enumerate(pts):
            p.payload = {"chunk_id": f"c{i}", "source_stem": f"族A_p{i}"}

        ranked = [(p, 0.9 - i * 0.1) for i, p in enumerate(pts)]
        monkeypatch.setattr(E, "_recall_queries", lambda q: [q])
        monkeypatch.setattr(E.C, "RETRIEVAL_RERANK_CASCADE", True)
        monkeypatch.setattr(E.Q, "query_text", lambda *a, **k: pts[:2])
        monkeypatch.setattr(E.Q, "sparse_rescue", lambda *a, **k: [])
        monkeypatch.setattr(E, "_rerank", lambda q, f, k=0, text_fn=None: ranked)
        monkeypatch.setattr(E, "_dynamic_cut",
                            lambda r, k: [r[0]])          # τ 只留 1 块
        monkeypatch.setattr(E, "_load_stem_registry",
                            lambda: {"族A": {"族A_p0", "族A_p1"}})
        monkeypatch.setattr(E, "_base_stem", lambda s: s.split("_p")[0])
        out = E.search_text("q", k=3, scope_stem="族A_p0", scope_hard=True,
                            with_scope_info=True)
        assert len(out["items"]) == 3                     # 回填到 k
        assert [d["chunk_id"] for d in out["items"]] == \
            ["c0", "c1", "c2"]
        assert out["scope"]["scope_hard"] is True

    def test_search_text_hard_no_variants_falls_back_soft(self, monkeypatch):
        """寄存器缺该族:退软加权(不误滤全库)。"""
        import mcp_servers.retrieval.engine_api as E
        seen = {}
        monkeypatch.setattr(E, "_recall_queries", lambda q: [q])

        def fake_query_text(q, k=5, scope_stems=None):
            seen["scope_stems"] = scope_stems
            return []
        monkeypatch.setattr(E.Q, "query_text", fake_query_text)
        monkeypatch.setattr(E, "_load_stem_registry", lambda: {})
        monkeypatch.setattr(E, "_base_stem", lambda s: s)
        E.search_text("q", k=3, scope_stem="不存在族", scope_hard=True)
        assert seen["scope_stems"] is None


class TestScopeGuidance:
    def test_raglite_messages_get_scope_guidance(self, monkeypatch):
        """scope 锁定:raglite 消息组装追加《stem》限定 system 消息。"""
        import agent_reasoning.raglite.stream as RS

        class _Gen:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        captured = {}

        def fake_llm_create(client, **kw):
            captured["messages"] = kw["messages"]
            raise RuntimeError("stop-before-stream")
        monkeypatch.setattr(RS, "_do_search",
                            lambda *a, **k: ({"items": []}, None))
        monkeypatch.setattr(RS, "llm_create_with_retry", fake_llm_create)
        monkeypatch.setattr(RS, "recent_dialogue_messages",
                            lambda *a, **k: [], raising=False)
        try:
            list(RS.raglite_answer_stream(
                "q", [], recorder=MagicMock(), trace_id="t", t0=0.0,
                scope_stem="晶圆减薄strasbaugh Model 7AF", scope_hard=True))
        except RuntimeError:
            pass
        sys_msgs = [m["content"] for m in captured["messages"]
                    if m["role"] == "system"]
        assert any("晶圆减薄strasbaugh Model 7AF" in c and "未涉及" in c
                   for c in sys_msgs)

    def test_react_build_messages_get_scope_guidance(self, monkeypatch):
        """react 冷启动:configurable.scope_stem 存在时 system 含限定口径。"""
        import agent_reasoning.ReAct.core.nodes as N
        monkeypatch.setattr(N, "build_prefetch_block",
                            lambda *a, **k: {"block": "", "mem_ids": set()},
                            raising=False)
        monkeypatch.setattr(N, "recent_dialogue_messages",
                            lambda *a, **k: [], raising=False)
        state = {"question": "q", "messages": [], "history": [],
                 "summary": ""}
        config = {"configurable": {
            "scope_stem": "半导体器件制造中的清洗技术 VIII",
            "scope_hard": True}}
        patch = N.build_messages_node(state, config)
        sys_msg = patch["messages"][0]
        assert "半导体器件制造中的清洗技术 VIII" in sys_msg.content
        assert "未涉及" in sys_msg.content


class TestQualityGateScoped:
    def test_scoped_zero_search_passes(self):
        from agent_reasoning.quality_gate import check
        r = check("该资料未涉及…", {"search_count": 0, "scoped": True},
                  tier="raglite")
        assert r["verdict"] == "passed"

    def test_unscoped_zero_search_escalates(self):
        from agent_reasoning.quality_gate import check
        r = check("答案", {"search_count": 0}, tier="raglite")
        assert r["verdict"] == "needs_escalation"
