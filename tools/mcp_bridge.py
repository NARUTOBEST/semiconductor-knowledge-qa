# -*- coding: utf-8 -*-
"""MCP 工具桥:把外部 MCP server 的工具取回并注册进全局 registry。

职责(唯一):连接 MCP server -> list_tools -> 逐个映射为 ToolSpec 注册;
之后 dispatch/validate/韧性中间件(重试、熔断、错误分类、截断)对 MCP 工具
与本地工具一视同仁,零特殊路径。

连接方式(env ``MCP_SERVERS``,JSON 数组;缺省连本地检索服务的 /mcp):
    [{"name": "retrieval", "url": "http://127.0.0.1:8002/mcp"}]      # HTTP 传输
    [{"name": "retrieval", "module": "mcp_servers.retrieval.tools"}]  # 进程内 memory 传输(测试)
可选项:``"name_prefix": "mcp_"``(工具名加前缀防多 server 撞名;默认不加)。

线程模型:MCP 客户端是 async 的,而 handler/dispatch 是同步的——每个 server
一个专职 daemon 线程跑常驻事件循环,handler 经 run_coroutine_threadsafe 投递,
fut.result(带超时)回同步。注册发生在桥线程,registry 仅 dict 写,与
主线程读取兼容(与既有 import 期注册同级别,非严格线程安全)。

启动时序:tools/__init__ 调 start_all(),首次连接有界等待(MCP_CONNECT_TIMEOUT_S,
默认 3s,0=不等);连不上不阻塞启动,桥线程按 MCP_RETRY_INTERVAL_S 后台重试,
重连成功后重新注册(schema 消费方 nodes 每次活取 registry.schemas(),晚到工具
下一请求即生效)。MCP SDK 未安装时降级:告警一条,不注册任何工具。
"""
from __future__ import annotations

import asyncio
import importlib
import json
import logging
import os
import threading

from .base import Category, ToolSpec, TruncatePolicy

logger = logging.getLogger("tools.mcp")

_CONNECT_TIMEOUT_S = float(os.getenv("MCP_CONNECT_TIMEOUT_S", "3"))
_RETRY_INTERVAL_S = float(os.getenv("MCP_RETRY_INTERVAL_S", "10"))
_DEFAULT_SERVERS = [{"name": "retrieval",
                     "url": None}]  # url 运行时取 C.RETRIEVAL_SERVICE_URL + "/mcp"


def _servers_config() -> list[dict]:
    """解析 MCP_SERVERS env(JSON 数组);未配置时默认连本地检索服务 /mcp。"""
    raw = os.getenv("MCP_SERVERS", "").strip()
    if not raw:
        cfgs = [dict(c) for c in _DEFAULT_SERVERS]
    else:
        cfgs = json.loads(raw)
        if not isinstance(cfgs, list):
            raise ValueError("MCP_SERVERS 须为 JSON 数组")
    for c in cfgs:
        if not c.get("url") and not c.get("module"):
            import config as C
            c["url"] = getattr(C, "RETRIEVAL_SERVICE_URL",
                               "http://127.0.0.1:8002").rstrip("/") + "/mcp"
        if not c.get("name"):
            raise ValueError(f"MCP server 配置缺 name: {c}")
    return cfgs


def _extract_result(res) -> object:
    """CallToolResult -> 裸结果(与旧本地 handler 返回形状一致)。

    FastMCP 把返回值包成 structured_content={"result": <return>},解包;
    业务错误(is_error)转 RuntimeError 上抛,交韧性中间件分类(不重试)。
    """
    if getattr(res, "is_error", False):
        text = "".join(getattr(c, "text", "") for c in (res.content or []))
        raise RuntimeError((text or "MCP tool error").strip()[:300])
    sc = getattr(res, "structured_content", None)
    if isinstance(sc, dict) and set(sc.keys()) == {"result"}:
        return sc["result"]
    if sc is not None:
        return sc
    for c in (res.content or []):
        if getattr(c, "type", "") == "text":
            try:
                return json.loads(c.text)
            except Exception:
                return c.text
    return None


class _ServerBridge:
    """一个 MCP server 的常驻连接 + 工具注册 + 同步调用门面。"""

    def __init__(self, cfg: dict):
        self.name = cfg["name"]
        self.url = cfg.get("url")
        self.module = cfg.get("module")
        self.prefix = cfg.get("name_prefix", "")
        self._loop: asyncio.AbstractEventLoop | None = None
        self._session = None                       # ClientSession(_serve 内赋值)
        self._ready = threading.Event()            # 首次注册尝试结束(成功或失败)
        self._stop = threading.Event()
        self._reopen = False                       # 会话失效时置位,请求 _serve 退出重连

    # ---------- 生命周期 ----------
    def start(self) -> None:
        threading.Thread(target=self._run, name=f"mcp-{self.name}",
                         daemon=True).start()

    def wait_ready(self, timeout: float) -> bool:
        """有界等待首次注册尝试完成(供 import 期同步时序;失败也返回)。"""
        return self._ready.wait(timeout)

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        while not self._stop.is_set():
            try:
                loop.run_until_complete(self._serve())
                # 正常返回:stop 或会话失效重建(_reopen)后落到这里继续循环
                if self._stop.is_set():
                    break
            except Exception as e:
                logger.warning("MCP server %s 连接失败,%ss 后重试: %s",
                               self.name, _RETRY_INTERVAL_S, str(e)[:160])
            finally:
                self._session = None
                self._ready.set()
            if self._stop.wait(_RETRY_INTERVAL_S):
                break

    async def _serve(self) -> None:
        from mcp.client.session import ClientSession
        # 传输层 yield (read, write) 二元组(memory 与 http 一致),其上再包 ClientSession
        async with self._open_session() as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                self._session = session
                self._register(await session.list_tools())
                self._ready.set()
                logger.info("MCP server %s 已连接,工具就绪", self.name)
                # 保活直到 stop 或会话失效重建(_session 须存续,handler 才能继续调用)
                while not self._stop.is_set() and not self._reopen:
                    await asyncio.sleep(0.2)
                self._reopen = False

    def _open_session(self):
        """返回传输层 async CM,进入后 yield (read, write) 流二元组。"""
        if self.module:
            # 进程内 memory 传输(测试/同进程直连):导入模块取其 MCPServer 单例
            from mcp.client._memory import InMemoryTransport
            server_obj = getattr(importlib.import_module(self.module), "mcp")
            return InMemoryTransport(server_obj)
        # HTTP 传输:mcp 2.x 的 streamable_http_client 直接 yield (read, write) 二元组;
        # 配置了服务间鉴权时经自定义 http client 带上 X-Internal-Token
        # (外层 CM 托管 client 生命周期,重连不泄漏)。
        from contextlib import asynccontextmanager
        from mcp.client.session_group import streamable_http_client, httpx2
        tok = self._internal_token()
        if not tok:
            return streamable_http_client(self.url)

        @asynccontextmanager
        async def _with_token():
            async with httpx2.AsyncClient(
                    headers={"X-Internal-Token": tok}) as client:
                async with streamable_http_client(
                        self.url, http_client=client) as streams:
                    yield streams
        return _with_token()

    @staticmethod
    def _internal_token() -> str:
        try:
            import config as C
            return getattr(C, "RETRIEVAL_INTERNAL_TOKEN", "")
        except Exception:
            return ""

    # ---------- 注册 ----------
    def _register(self, tools_res) -> None:
        from .registry import registry
        from .mcp_policies import policy_for
        names = []
        for t in tools_res.tools:
            pol = policy_for(self.name, t.name)
            schema = dict(t.input_schema or {})
            schema.pop("title", None)          # pydantic 附加项,对 LLM 是噪音
            timeout = float(pol.get("timeout", 30.0))
            spec = ToolSpec(
                name=self.prefix + t.name,
                description=t.description or "",
                category=pol.get("category", Category.EXTERNAL),
                parameters=schema,
                handler=self._make_handler(t.name, timeout),
                produces_sources=bool(pol.get("produces_sources", False)),
                timeout=timeout,
                retry_times=int(pol.get("retry_times", 0)),
                truncate=pol.get("truncate") or TruncatePolicy(),
                source_extractor=pol.get("source_extractor"),
                max_input_length=pol.get("max_input_length") or {},
            )
            registry.register(spec)
            names.append(spec.name)
        logger.info("MCP server %s 注册工具: %s", self.name, names)

    def _make_handler(self, tool: str, timeout: float):
        def _handler(**args):
            loop = self._loop
            session = self._session
            if loop is None or session is None:
                raise RuntimeError(f"MCP server {self.name} 未连接")
            fut = asyncio.run_coroutine_threadsafe(
                self._call(tool, args, timeout), loop)
            # 超时在 _call 内用 asyncio.wait_for 保证;此处 fut 超时再加一层兜底
            return fut.result(timeout + 10)
        return _handler

    @staticmethod
    def _is_session_dead(e: Exception) -> bool:
        """服务端会话已不存在(SSE GET 流断开后 FastMCP 回收会话 -> 404)。"""
        text = str(e).lower()
        return ("session not found" in text
                or "session terminated" in text
                or getattr(e, "status_code", None) == 404)

    async def _call(self, tool: str, args: dict, timeout: float):
        session = self._session
        if session is None:
            raise RuntimeError(f"MCP server {self.name} 未连接")
        try:
            res = await asyncio.wait_for(
                session.call_tool(tool, args or {}), timeout=timeout)
            return _extract_result(res)
        except Exception as e:
            if not self._is_session_dead(e) or self._stop.is_set():
                raise
            # 会话失效:请求桥线程重建连接,就绪后在**新会话**上重试一次
            logger.warning("MCP server %s 会话失效,重建连接后重试: %s",
                           self.name, str(e)[:120])
            self._reopen = True
            for _ in range(600):                    # 最多等 ~60s(重建+退避)
                await asyncio.sleep(0.1)
                new = self._session
                if new is not None and new is not session:
                    break
            else:
                raise RuntimeError(f"MCP server {self.name} 会话重建失败") from e
            res = await asyncio.wait_for(
                self._session.call_tool(tool, args or {}), timeout=timeout)
            return _extract_result(res)


# ==================== 模块级入口 ====================
_bridges: list[_ServerBridge] = []


def start_all() -> list[_ServerBridge]:
    """按配置启动全部 MCP 桥(幂等);有界等待首次注册,失败不阻塞。"""
    global _bridges
    if _bridges:
        return _bridges
    try:
        import mcp  # noqa: F401  SDK 探测:未安装则降级启动(不注册任何 MCP 工具)
    except ImportError:
        logger.warning("MCP SDK 未安装(pip install mcp),MCP 工具桥未启动")
        return _bridges
    try:
        cfgs = _servers_config()
    except Exception:
        logger.exception("MCP_SERVERS 配置解析失败,MCP 工具桥未启动")
        return _bridges
    for cfg in cfgs:
        b = _ServerBridge(cfg)
        b.start()
        _bridges.append(b)
    for b in _bridges:
        b.wait_ready(_CONNECT_TIMEOUT_S)   # 连不上也会在失败后返回,不阻塞
    return _bridges


def stop_all() -> None:
    """停掉全部桥(仅供测试/优雅关闭;连接由 daemon 线程退出自然回收)。"""
    for b in _bridges:
        b._stop.set()
    _bridges.clear()
