# -*- coding: utf-8 -*-
"""MCP 工具桥测试(memory 传输,进程内直连,不触网/不加载模型)。

覆盖:
  - import tools 后三件套经 MCP 注册进 registry,策略表字段齐全
    (category/produces_sources/source_extractor/truncate/retry);
  - dispatch 经真实 MCP 往返调用 get_chunk(打桩 engine_api,不打 Qdrant);
  - MCP 业务错误(is_error)转 RuntimeError 上抛(交韧性中间件分类);
  - 策略表回退:未知工具取空表(全默认,category=EXTERNAL);
  - schema:inputSchema 无 "title" 噪音,含 required。
"""
import pytest

import mcp_servers.retrieval.engine_api as engine_api
import tools.mcp_bridge as bridge_mod
from tools import registry, dispatch
from tools.base import Category
from tools.mcp_policies import policy_for


@pytest.fixture
def _bridge_ready():
    """import tools 时桥已按 conftest 的 memory 配置启动;等待注册完成。"""
    for b in bridge_mod.start_all():
        assert b.wait_ready(10), f"MCP server {b.name} 未就绪"
    yield


class TestRegistration:
    def test_retrieval_tools_registered_via_mcp(self, _bridge_ready):
        names = registry.names()
        assert {"search_text", "search_image", "get_chunk"} <= names

    def test_policy_fields_applied(self, _bridge_ready):
        for n in ("search_text", "search_image"):
            spec = registry.get(n)
            assert spec.category == Category.RETRIEVAL
            assert spec.produces_sources is True
            assert spec.source_extractor is not None
            assert spec.truncate.long_fields >= {"content", "table_html"}
            assert spec.retry_times == 1
        spec = registry.get("get_chunk")
        assert spec.produces_sources is False
        assert spec.source_extractor is None

    def test_schema_clean(self, _bridge_ready):
        sch = registry.require("search_text").parameters
        assert "title" not in sch
        assert sch["required"] == ["query"]
        assert set(sch["properties"]) >= {"query", "k"}


class TestDispatch:
    def test_get_chunk_roundtrip_via_mcp(self, _bridge_ready, monkeypatch):
        monkeypatch.setattr(
            engine_api, "get_chunk",
            lambda chunk_id: {"chunk_id": chunk_id, "content": "full"})
        r = dispatch("get_chunk", {"chunk_id": "abc__t00001"})
        assert r == {"chunk_id": "abc__t00001", "content": "full"}

    def test_mcp_none_result_passthrough(self, _bridge_ready, monkeypatch):
        monkeypatch.setattr(engine_api, "get_chunk", lambda chunk_id: None)
        assert dispatch("get_chunk", {"chunk_id": "nope"}) is None

    def test_mcp_business_error_raises(self, _bridge_ready, monkeypatch):
        def _boom(chunk_id):
            raise RuntimeError("qdrant down")
        monkeypatch.setattr(engine_api, "get_chunk", _boom)
        with pytest.raises(RuntimeError, match="qdrant down"):
            dispatch("get_chunk", {"chunk_id": "x"})


class TestPolicyLookup:
    def test_unknown_tool_falls_back_to_empty(self):
        assert policy_for("retrieval", "no_such_tool") == {}

    def test_server_scoped_key_wins(self):
        # retrieval:search_text 命中;假想另一 server 同名工具取空表
        assert policy_for("retrieval", "search_text").get(
            "category") == Category.RETRIEVAL
        assert policy_for("other", "search_text") == {}
