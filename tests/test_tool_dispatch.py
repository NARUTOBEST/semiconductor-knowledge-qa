# -*- coding: utf-8 -*-
"""dispatch 纯路由层单元测试(重构后)。

dispatch 现在只做:按 name 取 spec → 调用【一次】 handler → 返回裸结果;
未知/禁用工具返回 fatal dict。重试/熔断/限流已上移到韧性中间件
(agent_reasoning.ReAct.support.tool_resilience,见 test_circuit_breaker.py);
参数校验/schema 检查在 validate_runtime 节点(见 test_tool_error_nodes.py /
tool_validate)。handler 异常由 dispatch【原样上抛】,交中间件捕获。
"""
import httpx
import pytest

from tools import registry, dispatch, ToolSpec, Category  # noqa: E402
from tools.base import ErrorType  # noqa: E402


@pytest.fixture
def fake_tool(isolated_registry):
    """注册一个可控 handler 的工具,返回 holder dict。"""
    holder = {"calls": 0, "result": "ok", "side_effect": None}

    def handler(q, k=3):
        holder["calls"] += 1
        if holder["side_effect"] is not None:
            raise holder["side_effect"]
        return holder["result"]

    spec = ToolSpec(
        name="_ut_dispatch",
        description="unit test tool",
        category=Category.RETRIEVAL,
        parameters={"type": "object",
                    "properties": {"q": {"type": "string"},
                                   "k": {"type": "integer", "default": 3}},
                    "required": ["q"]},
        handler=handler,
        produces_sources=False,
        retry_times=2,
    )
    registry.register(spec)
    holder["spec"] = spec
    return holder


class TestDispatchRouting:
    def test_unknown_tool_returns_fatal(self):
        r = dispatch("no_such_tool", {})
        assert r["error_type"] == ErrorType.FATAL
        assert r["tool"] == "no_such_tool"
        assert "未知工具" in r["error"]

    def test_disabled_tool_returns_fatal(self, isolated_registry):
        spec = ToolSpec(
            name="_ut_disabled", description="d", category=Category.RETRIEVAL,
            parameters={"type": "object", "properties": {}},
            handler=lambda: 1, enabled=False,
        )
        registry.register(spec)
        r = dispatch("_ut_disabled", {})
        assert r["error_type"] == ErrorType.FATAL
        assert "暂不可用" in r["error"]

    def test_success_returns_bare_result(self, fake_tool):
        fake_tool["result"] = [{"a": 1}]
        r = dispatch(fake_tool["spec"].name, {"q": "hi"})
        assert r == [{"a": 1}]

    def test_business_error_dict_passed_through(self, fake_tool):
        # handler 返回 {"error":...}:dispatch 原样返回(不分类、不重试),
        # 由韧性中间件识别为下游失败。
        fake_tool["result"] = {"error": "检索服务不可用"}
        r = dispatch(fake_tool["spec"].name, {"q": "hi"})
        assert r == {"error": "检索服务不可用"}
        assert fake_tool["calls"] == 1

    def test_handler_exception_propagates(self, fake_tool):
        # dispatch 不再捕获异常:原样上抛,交韧性中间件分类/重试。
        fake_tool["side_effect"] = httpx.ReadTimeout("slow")
        with pytest.raises(httpx.ReadTimeout):
            dispatch(fake_tool["spec"].name, {"q": "hi"})

    def test_extra_kwargs_absorbed(self):
        # timeout 等历史 kwarg 被 **_ignored 吸收,不报错
        spec = ToolSpec(
            name="_ut_to", description="d", category=Category.RETRIEVAL,
            parameters={"type": "object",
                        "properties": {"q": {"type": "string"}},
                        "required": ["q"]},
            handler=lambda q: "ok",
        )
        registry.register(spec)
        assert dispatch("_ut_to", {"q": "x"}, timeout=1.0) == "ok"

    def test_unknown_args_not_filtered_here(self, fake_tool):
        # 参数过滤已移到 validate_runtime;dispatch 直接透传,未声明参数会让 handler 报错
        # (中间件捕获为 crash)。这里验证 dispatch 不再静默过滤。
        with pytest.raises(TypeError):
            dispatch(fake_tool["spec"].name, {"q": "hi", "bogus": 123})


class TestRetrievalToolsWiring:
    """三检索工具经 dispatch 单次路由可达(handler 打桩,不打真实 HTTP)。"""

    def test_search_text_routes_to_registered_handler(self):
        called = {}

        def fake_post(query=None, **kw):
            called["kw"] = {"query": query, **kw}
            return [{"chunk_id": "c1", "source_stem": "doc", "score": 0.9}]

        spec = registry.require("search_text")
        original = spec.handler
        spec.handler = fake_post
        try:
            r = dispatch("search_text", {"query": "ALD"})
        finally:
            spec.handler = original
        assert r == [{"chunk_id": "c1", "source_stem": "doc", "score": 0.9}]
        assert called["kw"].get("query") == "ALD"

    def test_get_chunk_routes(self):
        spec = registry.require("get_chunk")
        original = spec.handler
        spec.handler = lambda chunk_id: {"chunk_id": chunk_id, "content": "full"}
        try:
            r = dispatch("get_chunk", {"chunk_id": "abc__t00001"})
        finally:
            spec.handler = original
        assert r["content"] == "full"
