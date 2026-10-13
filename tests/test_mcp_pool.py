# -*- coding: utf-8 -*-
"""MCP 连接池(多隧道×多会话)单测。

背景(2026-10-09):单隧道+单会话是 20 并发的队头阻塞单点(轮次13 实测首题
齐发 11 题 SSE 超时)。_ServerBridge 现按 urls 持有多条 _Conn,handler 路由取
在途调用最少的就绪连接(round-robin 打散同分),一条挂死只摘一条。

覆盖:
  - _pick 路由:就绪者中取 inflight 最少,同分轮转;全未就绪回第一个(旧语义);
    会话死亡的成员被绕开;
  - 配置展开:urls 列表 → N 连接;url 单值 → 1 连接;module → 1 连接;
  - memory 传输双连接端到端:两条会话都能服务调用,一条会话摘除后另一条兜底。
"""
import pytest

import mcp_servers.retrieval.engine_api as engine_api
from tools.mcp_bridge import _Conn, _ServerBridge


def _fake_conn(idx, ready=True, has_session=True, inflight=0):
    c = _Conn("retrieval", url=None, index=idx)
    if ready:
        c._ready.set()
    if has_session:
        c._session = object()          # 哨兵:_pick 只判 None
    c.inflight = inflight
    return c


def _pool(conns, urls=None, module="mcp_servers.retrieval.tools"):
    b = _ServerBridge.__new__(_ServerBridge)   # 绕过构造,直接注入连接
    b.name = "retrieval"
    b.prefix = ""
    b.module = module
    b._conns = conns
    b._rr = 0
    b._rr_lock = __import__("threading").Lock()
    b._registered = True
    b._registered_conn = None
    return b


class TestPickRouting:
    def test_round_robin_among_equal(self):
        conns = [_fake_conn(i) for i in range(3)]
        b = _pool(conns)
        picked = [b._pick() for _ in range(6)]
        # 同分轮转:3 条连接各被选中 2 次
        assert [picked.count(c) for c in conns] == [2, 2, 2]

    def test_least_inflight_wins_over_rr(self):
        c0, c1 = _fake_conn(0, inflight=5), _fake_conn(1, inflight=0)
        b = _pool([c0, c1])
        assert all(b._pick() is c1 for _ in range(4))

    def test_all_not_ready_returns_first(self):
        conns = [_fake_conn(0, ready=False, has_session=False),
                 _fake_conn(1, ready=False, has_session=False)]
        b = _pool(conns)
        assert b._pick() is conns[0]      # 保持旧"未连接"错误语义

    def test_dead_session_bypassed(self):
        c0 = _fake_conn(0, has_session=False)   # ready 但会话已死
        c1 = _fake_conn(1)
        b = _pool([c0, c1])
        assert all(b._pick() is c1 for _ in range(4))


class TestConfigExpansion:
    def test_urls_list_creates_pool(self):
        b = _ServerBridge({"name": "retrieval",
                           "urls": ["http://a/mcp", "http://b/mcp", "http://c/mcp"]})
        assert len(b._conns) == 3
        assert [c.url for c in b._conns] == ["http://a/mcp", "http://b/mcp",
                                             "http://c/mcp"]
        assert [c.label for c in b._conns] == ["retrieval#0", "retrieval#1",
                                               "retrieval#2"]

    def test_single_url_keeps_old_behavior(self):
        b = _ServerBridge({"name": "retrieval", "url": "http://a/mcp"})
        assert len(b._conns) == 1

    def test_module_transport_single_conn(self):
        b = _ServerBridge({"name": "retrieval",
                           "module": "mcp_servers.retrieval.tools"})
        assert len(b._conns) == 1


class TestPoolEndToEnd:
    """memory 传输双连接端到端:同进程内两条会话均可服务(不打桩线程机械)。"""

    @pytest.fixture
    def pool(self):
        b = _ServerBridge({"name": "retrieval", "module":
                           "mcp_servers.retrieval.tools",
                           "urls": ["m0", "m1"]})
        b.start()
        assert b.wait_ready(15), "memory 池未就绪"
        yield b
        b.stop()

    def test_both_sessions_serve(self, pool, monkeypatch):
        import asyncio
        monkeypatch.setattr(
            engine_api, "get_chunk",
            lambda chunk_id: {"chunk_id": chunk_id, "via": "ok"})
        for i, conn in enumerate(pool._conns):   # 每条会话各自直达调用
            fut = asyncio.run_coroutine_threadsafe(
                conn._call("get_chunk", {"chunk_id": f"c{i}"}, 10.0),
                conn._loop)
            assert fut.result(15) == {"chunk_id": f"c{i}", "via": "ok"}
            assert conn._session is not None

    def test_one_conn_dead_others_take_over(self, pool, monkeypatch):
        monkeypatch.setattr(
            engine_api, "get_chunk", lambda chunk_id: {"chunk_id": chunk_id})
        pool._conns[0]._session = None          # 模拟会话挂死(未重建)
        handler = pool._make_handler("get_chunk", 30.0)
        assert handler(chunk_id="x") == {"chunk_id": "x"}   # 走 #1 兜底
