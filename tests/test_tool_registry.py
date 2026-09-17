# -*- coding: utf-8 -*-
"""工具注册中心单元测试。"""
import pytest

from tools import registry, ToolSpec, Category, TruncatePolicy
from tools.base import ErrorType, make_error


@pytest.fixture
def isolated_registry():
    """每个用例后还原全局 registry(保留 import 期注册的检索工具)。"""
    saved = dict(registry._specs)
    yield registry
    registry._specs = saved


def _make_spec(name="dummy", enabled=True, produces_sources=False,
               category=Category.RETRIEVAL, handler=lambda **k: "ok"):
    return ToolSpec(
        name=name, description=f"desc {name}", category=category,
        parameters={"type": "object",
                    "properties": {"q": {"type": "string"}},
                    "required": ["q"]},
        handler=handler,
        produces_sources=produces_sources,
        enabled=enabled,
    )


class TestRegistry:
    def test_search_tools_registered_at_import(self):
        names = registry.names()
        assert {"search_text", "search_image", "get_chunk"} <= names

    def test_register_and_get(self, isolated_registry):
        spec = _make_spec("t1")
        isolated_registry.register(spec)
        assert isolated_registry.get("t1") is spec
        assert isolated_registry.require("t1") is spec

    def test_require_unknown_raises(self, isolated_registry):
        with pytest.raises(KeyError):
            isolated_registry.require("does_not_exist")

    def test_disabled_excluded_from_schemas_but_gettable(self, isolated_registry):
        on = _make_spec("on", enabled=True)
        off = _make_spec("off", enabled=False)
        isolated_registry.register(on)
        isolated_registry.register(off)
        names = isolated_registry.names(enabled_only=True)
        assert "on" in names and "off" not in names
        all_names = isolated_registry.names(enabled_only=False)
        assert {"on", "off"} <= all_names
        # disabled 仍可取(dispatch 防御性返回不可用错误)
        assert isolated_registry.get("off") is off

    def test_by_category(self, isolated_registry):
        isolated_registry.register(_make_spec("r1", category=Category.RETRIEVAL))
        isolated_registry.register(_make_spec("r2", category=Category.RETRIEVAL))
        names = {s.name for s in isolated_registry.by_category(Category.RETRIEVAL)}
        assert {"r1", "r2"} <= names
        # 不存在的类别返回空
        assert isolated_registry.by_category("nonexistent") == []

    def test_names_that_produce_sources(self, isolated_registry):
        isolated_registry.register(_make_spec("src", produces_sources=True))
        isolated_registry.register(_make_spec("nosrc", produces_sources=False))
        ps = isolated_registry.names_that_produce_sources()
        assert "src" in ps and "nosrc" not in ps
        # 检索工具迁移后默认为来源工具
        assert {"search_text", "search_image"} <= ps
        assert "get_chunk" not in ps

    def test_schemas_shape(self, isolated_registry):
        isolated_registry.register(_make_spec("sh"))
        schemas = isolated_registry.schemas()
        match = [s for s in schemas if s["function"]["name"] == "sh"]
        assert len(match) == 1
        fn = match[0]
        assert fn["type"] == "function"
        assert set(fn["function"]) == {"name", "description", "parameters"}
        assert fn["function"]["parameters"]["required"] == ["q"]

    def test_search_specs_produce_sources_flag(self):
        assert registry.get("search_text").produces_sources is True
        assert registry.get("search_image").produces_sources is True
        assert registry.get("get_chunk").produces_sources is False

    def test_search_specs_category(self):
        for n in ("search_text", "search_image", "get_chunk"):
            assert registry.get(n).category == Category.RETRIEVAL

    def test_search_specs_retry_times(self):
        # 与重构前 nodes._MAX_TOOL_RETRIES=1 等价
        assert registry.get("search_text").retry_times == 1
        assert registry.get("search_image").retry_times == 1
        assert registry.get("get_chunk").retry_times == 1

    def test_search_specs_have_source_extractor(self):
        assert registry.get("search_text").source_extractor is not None
        assert registry.get("search_image").source_extractor is not None
        assert registry.get("get_chunk").source_extractor is None


class TestBaseHelpers:
    def test_make_error_shape(self):
        err = make_error("boom", ErrorType.TIMEOUT, "t1")
        assert err == {"error": "boom", "error_type": "timeout", "tool": "t1"}

    def test_default_retry_on_timeout(self):
        import httpx
        from tools.base import default_retry_on
        assert default_retry_on(httpx.ConnectTimeout("x")) is True
        assert default_retry_on(httpx.ReadTimeout("x")) is True

    def test_default_retry_on_5xx_and_429(self):
        import httpx
        from tools.base import default_retry_on
        for code in (500, 502, 503, 429):
            req = httpx.Request("POST", "http://x")
            resp = httpx.Response(code, request=req)
            assert default_retry_on(httpx.HTTPStatusError("x", request=req, response=resp)) is True

    def test_default_retry_off_on_4xx(self):
        import httpx
        from tools.base import default_retry_on
        req = httpx.Request("POST", "http://x")
        for code in (400, 401, 403, 404):
            resp = httpx.Response(code, request=req)
            assert default_retry_on(httpx.HTTPStatusError("x", request=req, response=resp)) is False

    def test_default_retry_off_on_plain_exception(self):
        from tools.base import default_retry_on
        assert default_retry_on(ValueError("x")) is False

    def test_truncate_policy_defaults(self):
        p = TruncatePolicy()
        assert p.long_fields == set()
        assert p.custom_fn is None
        assert p.max_chars_override is None
