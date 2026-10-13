#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""qdrant 反向隧道负载均衡(GPU 侧)。

每条 VM→GPU 隧道各绑一个错开的反向端口 -R 127.0.0.1:6334+i → VM qdrant:6333
(见 start_tunnel.sh;同绑 6333 会端口冲突把其余隧道顶死)。本进程监听 GPU 本机
6333,把每个新 TCP 连接轮询分发到 6334..6333+K,让 qdrant 回查流量摊到 K 条
隧道的独立 transport 上(与 10 条 MCP 正向隧道互不挤压)。

检索服务零改动:QDRANT_URL=http://127.0.0.1:6333 不变。
某条隧道瞬断时按序尝试下一个后端(连通性探测以实际 connect 为准)。
QDRANT_TUNNELS 控制后端数,默认 10。
"""
import asyncio
import itertools
import os

_K = int(os.getenv("QDRANT_TUNNELS", "10"))
_BACKENDS = [("127.0.0.1", 6334 + i) for i in range(_K)]
_rr = itertools.count()


async def _pipe(reader, writer):
    try:
        while True:
            data = await reader.read(65536)
            if not data:
                break
            writer.write(data)
            await writer.drain()
    except Exception:
        pass
    finally:
        try:
            writer.close()
        except Exception:
            pass


async def _handle(client_reader, client_writer):
    start = next(_rr) % len(_BACKENDS)
    for off in range(len(_BACKENDS)):            # 连不上就试下一个,最多绕一圈
        host, port = _BACKENDS[(start + off) % len(_BACKENDS)]
        try:
            backend_reader, backend_writer = await asyncio.wait_for(
                asyncio.open_connection(host, port), timeout=3)
            break
        except Exception:
            continue
    else:
        client_writer.close()
        return
    try:
        await asyncio.gather(_pipe(client_reader, backend_writer),
                             _pipe(backend_reader, client_writer))
    finally:
        try:
            backend_writer.close()
        except Exception:
            pass


async def main():
    server = await asyncio.start_server(_handle, "127.0.0.1", 6333)
    print(f"qdrant_lb up: 127.0.0.1:6333 -> {_BACKENDS}", flush=True)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
