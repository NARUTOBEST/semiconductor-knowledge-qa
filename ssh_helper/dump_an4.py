#!/usr/bin/env python3
"""取挂死窗口深处的完整 dump,列出所有非空闲线程的完整栈。"""
import re, collections

text = open("/tmp/dwt.log", encoding="utf-8", errors="replace").read()
TS = r"^(\d{4}-\d{2}-\d{2}T(?:\d{2}:\d{2}):\d{2}\.\d+)"

# 先按"worker 的一次完整转储"切:连续的线程块属于同一 worker 同一时刻
blocks = []  # (ts, tid, frames)
cur = None
for ln in text.splitlines():
    m = re.match(TS + r"\s?(.*)$", ln)
    if m:
        ts, body = m.group(1), re.sub(r"^Z\s*", "", m.group(2))
    else:
        ts, body = None, ln
    mh = re.match(r"(?:Current )?[Tt]hread 0x([0-9a-f]+)", body)
    if mh:
        if cur and cur[2]:
            blocks.append(cur)
        cur = [ts, mh.group(1), [body]]
    elif cur is not None:
        if body.lstrip().startswith("File "):
            cur[2].append(body.strip())
        else:
            if cur[2]:
                blocks.append(cur)
            cur = None
if cur and cur[2]:
    blocks.append(cur)

def sig(fr):
    return " <- ".join(f.split("/")[-1] for f in fr[1:])

IDLE = re.compile(
    r"tool_pool\.py.*57 in _worker|selectors\.py.*select"
    r"|anyio/_backends/_asyncio\.py.*1022 in run"
    r"|concurrent/futures/thread\.py.*90 in _worker"
    r"|mcp_bridge\.py.*_run|lifecycle\.py.*prune_loop|multiprocess\.py.*pong"
    r"|queue\.py.*171 in get|_queue\.py.*64 in wait|_queue\.py.*114 in wait"
    r"|pipeline\.py.*135 in _loop"
)

# 按秒聚成 dump 群(同一 worker 转储的块时间戳几乎相同)
groups = collections.OrderedDict()
for ts, tid, fr in blocks:
    if not ts:
        continue
    g = ts[:19]
    groups.setdefault(g, []).append((ts, tid, fr))

targets = ["04:14:52", "04:15:22", "04:15:52", "04:16:22"]
seen = set()
for g, lst in sorted(groups.items()):
    hhmmss = g[11:]
    for t in targets:
        if hhmmss >= t and t not in seen:
            seen.add(t)
            print(f"\n================ DUMP @ {g}  ({len(lst)} threads) ================")
            for ts, tid, fr in lst:
                s = sig(fr)
                if IDLE.search("\n".join(fr)):
                    continue
                print(f"\n-- tid {tid}")
                for f in fr:
                    print("   " + f)
