# -*- coding: utf-8 -*-
"""MCP 工具桥:把外部 MCP server 的工具取回并注册进全局 registry。

职责(唯一):连接 MCP server -> list_tools -> 逐个映射为 ToolSpec 注册;
之后 dispatch/validate/韧性中间件(重试、熔断、错误分类、截断)对 MCP 工具
与本地工具一视同仁,零特殊路径。

连接方式(env ``MCP_SERVERS``,JSON 数组;缺省连本地检索服务的 /mcp):
    [{"name": "retrieval", "url": "http://127.0.0.1:8002/mcp"}]      # 单连接
    [{"name": "retrieval", "urls": ["http://127.0.0.1:8002/mcp",
                                    "http://127.0.0.1:8003/mcp"]}]   # 连接池
    [{"name": "retrieval", "module": "mcp_servers.retrieval.tools"}]  # memory 传输(测试)
未显式配置时按 ``RETRIEVAL_SERVICE_URLS``(逗号分隔多隧道端口,见 config)展开。
可选项:``"name_prefix": "mcp_"``(工具名加前缀防多 server 撞名;默认不加)。

线程模型:MCP 客户端是 async 的,而 handler/dispatch 是同步的——每条连接一个
专职 daemon 线程跑常驻事件循环,handler 经 run_coroutine_threadsafe 投递到
所选连接的循环,fut.result(带超时)回同步。注册只在任一连接首次就绪时发生
一次(注册连接重连时补注册刷新 schema)。

连接池(2026-10-09):此前单会话+单隧道是 20 并发的队头阻塞单点(轮次13
实测首题齐发 11 题 SSE 超时)。现在 K 条连接(一一绑定 K 条隧道)各自独立
生命周期,handler 路由取在途调用最少的就绪连接(round-robin 打散同分)——
一条隧道挂死只摘一条,其余成员继续服务。同一连接内的并发调用由 MCP SDK
request-id 复用;跨连接才真正分摊 SSH transport 的流控窗口。

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
import time

from .base import Category, ToolSpec, TruncatePolicy

logger = logging.getLogger("tools.mcp")

_CONNECT_TIMEOUT_S = float(os.getenv("MCP_CONNECT_TIMEOUT_S", "3"))
# 桥层读超时(20261015 环境可调):检索(翻译双路+重排排队)在并发下普遍 >5s,
# 且席位绑定/冲突探测后调用数 ~3 倍——偶发 >60s 属正常排队,ReadTimeout 会被
# 上层判成会话失效连环重建(轮次19 20x10 srcs=0 全员故障根因)。默认放宽 300s:
# 方案2 search_batch(主+≤3 子路串行)策略超时 200s 也要在此之下,桥不能先掐。
# 与工具层策略超时(90s/200s)+重试对齐。
_READ_TIMEOUT_S = float(os.getenv("MCP_READ_TIMEOUT_S", "300"))
_RETRY_INTERVAL_S = float(os.getenv("MCP_RETRY_INTERVAL_S", "10"))
# 握手超时(20261015):ClientSession 默认无请求超时——initialize 直返 200 后
# list_tools 的响应走 SSE 流,SSE 流没建立就永久挂起(既不抛错也不重试,方案3
# 20u 实测 200 连接只 7 条就绪、零失败日志、GPU 侧零流量,队列风暴全挤 7 条)。
# 握手两步各给硬超时,超时即按连接失败走 10s 重试循环,自愈。
_HANDSHAKE_TIMEOUT_S = float(os.getenv("MCP_HANDSHAKE_TIMEOUT_S", "20"))
# 启动错峰:200 连接同秒齐发握手会挤爆服务端 SSE 建流(实测根因),
# 按连接序号线性错开,60 连接内摊开 ~15s
_STARTUP_STAGGER_S = float(os.getenv("MCP_STARTUP_STAGGER_S", "0.25"))
_DEFAULT_SERVERS = [{"name": "retrieval",
                     "url": None}]  # url 运行时取 C.RETRIEVAL_SERVICE_URLS + "/mcp"


def _servers_config() -> list[dict]:
    """解析 MCP_SERVERS env(JSON 数组);未配置时按 RETRIEVAL_SERVICE_URLS
    展开为连接池(单地址时即单连接,与旧行为一致)。"""
    raw = os.getenv("MCP_SERVERS", "").strip()
    if not raw:
        cfgs = [dict(c) for c in _DEFAULT_SERVERS]
    else:
        cfgs = json.loads(raw)
        if not isinstance(cfgs, list):
            raise ValueError("MCP_SERVERS 须为 JSON 数组")
    for c in cfgs:
        if not c.get("url") and not c.get("urls") and not c.get("module"):
            import config as C
            base = list(getattr(C, "RETRIEVAL_SERVICE_URLS", None) or
                        [getattr(C, "RETRIEVAL_SERVICE_URL",
                                 "http://127.0.0.1:8002").rstrip("/")])
            c["urls"] = [u.rstrip("/") + "/mcp" for u in base]
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


def _is_session_dead(e: Exception) -> bool:
    """服务端会话已不可用。两类信号:
    - 会话被服务端回收(SSE GET 流断开后 FastMCP 回收会话 -> 404);
    - 常驻会话的 SSE 流已被服务端 idle timeout / 隧道中断掐死,后续调用
      统一报 "SSE stream ended without a response"(实测:GPU 检索服务
      空闲超时后,backend 常驻会话所有调用皆此错,必须重建连接)。"""
    text = str(e).lower()
    return ("session not found" in text
            or "session terminated" in text
            or "sse stream ended" in text
            or getattr(e, "status_code", None) == 404)


def _internal_token() -> str:
    try:
        import config as C
        return getattr(C, "RETRIEVAL_INTERNAL_TOKEN", "")
    except Exception:
        return ""


class _Conn:
    """单条 MCP 连接:一个常驻会话 + 一个专职事件循环线程。

    可绑定一个隧道端口(url)或进程内 memory 传输(module)。原 _ServerBridge
    的连接机械(_run/_serve/_open_session/_call)整体迁入;池内各连接生命周期
    完全独立——一条隧道/会话挂死只重建自己,不拖累其他成员。
    """

    def __init__(self, server: str, url, module=None, index=0,
                 on_ready=None):
        self.server = server
        self.label = f"{server}#{index}"   # 日志标识(区分同池成员)
        self.url = url
        self.module = module
        self._on_ready = on_ready          # 就绪回调(conn, tools_res)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._session = None
        self._ready = threading.Event()    # 首次注册尝试结束(成功或失败)
        self._stop = threading.Event()
        self._reopen = False               # 会话失效时置位,请求 _serve 退出重连
        self.inflight = 0                  # 在途调用数(handler 路由依据)

    # ---------- 生命周期 ----------
    def start(self) -> None:
        threading.Thread(target=self._run, name=f"mcp-{self.label}",
                         daemon=True).start()

    def wait_ready(self, timeout: float) -> bool:
        """有界等待首次注册尝试完成(供 import 期同步时序;失败也返回)。"""
        return self._ready.wait(timeout)

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        # 启动错峰:同 worker 内按序号摊开,20 连接摊 5s,避免齐发建流挤爆服务端
        if self.index and _STARTUP_STAGGER_S > 0:
            time.sleep(self.index * _STARTUP_STAGGER_S)
        while not self._stop.is_set():
            try:
                loop.run_until_complete(self._serve())
                # 正常返回:stop 或会话失效重建(_reopen)后落到这里继续循环
                if self._stop.is_set():
                    break
            except Exception as e:
                logger.warning("MCP server %s 连接失败,%ss 后重试: %s",
                               self.label, _RETRY_INTERVAL_S, str(e)[:160])
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
                # 握手两步各给硬超时(默认无超时会永久挂起,见 _HANDSHAKE_TIMEOUT_S 注释)
                await asyncio.wait_for(session.initialize(),
                                       timeout=_HANDSHAKE_TIMEOUT_S)
                self._session = session
                if self._on_ready is not None:
                    tools_res = await asyncio.wait_for(
                        session.list_tools(), timeout=_HANDSHAKE_TIMEOUT_S)
                    self._on_ready(self, tools_res)
                self._ready.set()
                logger.info("MCP server %s 已连接,工具就绪", self.label)
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
        tok = _internal_token()
        if not tok:
            return streamable_http_client(self.url)

        @asynccontextmanager
        async def _with_token():
            # 显式放宽读超时:httpx 默认 5s,检索在并发下普遍 >5s(见
            # _READ_TIMEOUT_S 注释)。工具层 90s 策略超时兜底。
            async with httpx2.AsyncClient(
                    headers={"X-Internal-Token": tok},
                    timeout=httpx2.Timeout(_READ_TIMEOUT_S + 30,
                                           read=_READ_TIMEOUT_S)) as client:
                async with streamable_http_client(
                        self.url, http_client=client) as streams:
                    yield streams
        return _with_token()

    # ---------- 调用 ----------
    async def _call(self, tool: str, args: dict, timeout: float):
        self.inflight += 1
        try:
            session = self._session
            if session is None:
                raise RuntimeError(f"MCP server {self.label} 未连接")
            try:
                res = await asyncio.wait_for(
                    session.call_tool(tool, args or {}), timeout=timeout)
                return _extract_result(res)
            except Exception as e:
                if not _is_session_dead(e) or self._stop.is_set():
                    raise
                # 会话失效:请求桥线程重建连接,就绪后在**新会话**上重试一次
                logger.warning("MCP server %s 会话失效,重建连接后重试: %s",
                               self.label, str(e)[:120])
                self._reopen = True
                for _ in range(600):                # 最多等 ~60s(重建+退避)
                    await asyncio.sleep(0.1)
                    new = self._session
                    if new is not None and new is not session:
                        break
                else:
                    raise RuntimeError(
                        f"MCP server {self.label} 会话重建失败") from e
                res = await asyncio.wait_for(
                    self._session.call_tool(tool, args or {}), timeout=timeout)
                return _extract_result(res)
        finally:
            self.inflight -= 1


class _ServerBridge:
    """一个 MCP server 的连接池(多隧道×多会话)+ 工具注册 + 同步调用门面。

    cfg 给 "url"(单连接,与旧版行为一致)或 "urls"(列表,每项绑定一条
    独立隧道/会话;module 传输默认单连接,显式给 urls 可多连接——池化
    路由逻辑可在进程内直接测试)。注册只发生一次(任一连接
    首次就绪时;注册连接重连才补注册);handler 路由取在途调用最少且就绪的
    连接(round-robin 打散同分),一条挂死只摘一条,其余继续服务。
    """

    def __init__(self, cfg: dict):
        self.name = cfg["name"]
        self.prefix = cfg.get("name_prefix", "")
        self.module = cfg.get("module")
        urls = list(cfg.get("urls") or [])
        if cfg.get("url") and not urls:
            urls = [cfg["url"]]
        if not urls:
            urls = [None]              # 运行时默认(_servers_config 已填)
        self._conns = [_Conn(self.name, u, module=self.module, index=i,
                             on_ready=self._on_conn_ready)
                       for i, u in enumerate(urls)]
        self._rr = 0                   # round-robin 打散同分连接
        self._rr_lock = threading.Lock()
        self._registered = False
        self._registered_conn: _Conn | None = None

    # ---------- 生命周期 ----------
    def start(self) -> None:
        for c in self._conns:
            c.start()

    def wait_ready(self, timeout: float) -> bool:
        """有界等待任一连接完成首次注册尝试(失败也返回)。"""
        deadline = time.monotonic() + max(0.0, timeout)
        for c in self._conns:
            if c.wait_ready(max(0.0, deadline - time.monotonic())):
                return True
        return False

    def stop(self) -> None:
        for c in self._conns:
            c.stop()

    def _pick(self) -> _Conn:
        """路由:在途调用最少且就绪的连接;同分 round-robin 轮转。
        全未就绪走第一个,保持"未连接"错误语义与旧版一致。"""
        ready = [c for c in self._conns
                 if c._ready.is_set() and c._session is not None]
        if not ready:
            return self._conns[0]
        with self._rr_lock:
            self._rr += 1
            k = self._rr % len(ready)
        rot = ready[k:] + ready[:k]
        return min(rot, key=lambda c: c.inflight)

    def _on_conn_ready(self, conn: _Conn, tools_res) -> None:
        # 任一连接就绪即注册一次;仅"注册过的那条"重连才补注册(刷新 schema)
        if not self._registered or self._registered_conn is conn:
            self._register(tools_res)
            self._registered = True
            self._registered_conn = conn

    # ---------- 注册 ----------
    def _register(self, tools_res) -> None:
        from .registry import registry
        from .mcp_policies import policy_for
        names = []
        for t in tools_res.tools:
            pol = policy_for(self.name, t.name)
            schema = dict(t.input_schema or {})
            schema.pop("title", None)          # pydantic 附加项,对 LLM 是噪音
            # 策略可给 schema 打补丁(加可选参数,如 search_text 的 queries 数组);
            # 服务端不感知——handler 在发出 MCP 调用前消化这些参数。
            for k, v in (pol.get("parameters_patch", {})
                         .get("properties", {}) or {}).items():
                schema.setdefault("properties", {})[k] = v
            timeout = float(pol.get("timeout", 30.0))
            spec = ToolSpec(
                name=self.prefix + t.name,
                description=(t.description or "") + pol.get("description_suffix", ""),
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
            conn = self._pick()
            if conn._loop is None or conn._session is None:
                raise RuntimeError(f"MCP server {self.name} 未连接")
            # 轮次11 回退并行检索:恢复单 query 串行(实测并行合并伤 precision,
            # 见 eval/optimization_log/20261008_round10_20u200.md)。此处仅防御性
            # 丢弃可能残留的 queries 参数,不让其透传到服务端。
            args.pop("queries", None)
            fut = asyncio.run_coroutine_threadsafe(
                conn._call(tool, args, timeout), conn._loop)
            # 超时在 _call 内用 asyncio.wait_for 保证;此处 fut 超时再加一层兜底
            return fut.result(timeout + 10)
        return _handler


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
        b.stop()
    _bridges.clear()
