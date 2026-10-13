# -*- coding: utf-8 -*-
"""20 并发 MCP tools/call 压测:每位用户独立 streamable HTTP 会话,顺序发 3 次
search_text,统计排队延迟(口径对齐轮次19 事故:tools/call 47s vs 底层 2.6s)。
GPU 机本地跑:python mcp_burst.py [--users 20] [--calls 3]
"""
import argparse, asyncio, json, statistics, time

BASE = "http://127.0.0.1:8002/mcp"
TOKEN = ""
QUERIES = [
    "ALD 工艺中 TMA 前驱体的饱和脉冲参数",
    "wafer chuck 温度控制原理",
    "plasma pretreatment 等离子预处理流程",
    "腔体压力控制与节气门(throttle valve)调节",
    "removal chamber 清洗腔的维护周期",
    "RF 射频电源功率设置范围",
    "recipe 建立与编辑步骤",
    "prefix 配气系统的组成",
    "泄漏率检漏方法与验收标准",
    "氧气等离子清洗的工艺窗口",
    "远程等离子体源 RPS 的工作原理",
    "基片传输机器人手臂校准流程",
    "ALD 沉积速率的影响因素",
    "尾气处理与尾气洗涤器维护",
    "MFC 质量流量计校准步骤",
    "设备联锁 interlock 逻辑说明",
    "生长温度对薄膜均匀性的影响",
    "脉冲时间与饱和曲线的关系",
    "日常点检项目清单",
    "报警代码 E001-E050 含义",
]

async def one_user(uid, calls):
    from contextlib import asynccontextmanager
    from mcp.client.session import ClientSession
    from mcp.client.session_group import streamable_http_client, httpx2

    @asynccontextmanager
    async def _with_token():
        async with httpx2.AsyncClient(
                headers={"X-Internal-Token": TOKEN},
                timeout=httpx2.Timeout(300.0, read=300.0)) as client:
            async with streamable_http_client(BASE, http_client=client) as streams:
                yield streams

    lat, errs = [], 0
    async with _with_token() as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            for i in range(calls):
                q = QUERIES[(uid * calls + i) % len(QUERIES)]
                t0 = time.perf_counter()
                try:
                    res = await asyncio.wait_for(
                        session.call_tool(
                            "search_text",
                            {"query": q, "k": 4, "score_ratio": 0.5}),
                        timeout=300)
                    if getattr(res, "isError", False):
                        errs += 1
                except Exception:
                    errs += 1
                lat.append(time.perf_counter() - t0)
    return {"user": uid, "lat": lat, "errors": errs}

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=20)
    ap.add_argument("--calls", type=int, default=3)
    a = ap.parse_args()
    t_start = time.perf_counter()
    results = await asyncio.gather(*[one_user(u, a.calls) for u in range(a.users)])
    wall = time.perf_counter() - t_start
    all_lat = [x for r in results for x in r["lat"]]
    errs = sum(r["errors"] for r in results)
    all_lat.sort()
    def pct(p):
        return all_lat[min(len(all_lat) - 1, int(len(all_lat) * p))]
    print(json.dumps({
        "users": a.users, "calls_per_user": a.calls, "total_calls": len(all_lat),
        "errors": errs, "wall_s": round(wall, 1),
        "p50_s": round(pct(0.50), 2), "p95_s": round(pct(0.95), 2),
        "max_s": round(all_lat[-1], 2), "mean_s": round(statistics.mean(all_lat), 2),
    }, ensure_ascii=False))
    for r in results:
        print(f"u{r['user']:02d}: " + " ".join(f"{x:.1f}" for x in r["lat"])
              + (f" err={r['errors']}" if r["errors"] else ""))

if __name__ == "__main__":
    for line in open("/root/autodl-tmp/app/ret.env", encoding="utf-8"):
        line = line.strip()
        if line.startswith("RETRIEVAL_INTERNAL_TOKEN="):
            TOKEN = line.split("=", 1)[1].strip()
    asyncio.run(main())
