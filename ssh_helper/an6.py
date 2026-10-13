#!/usr/bin/env python3
"""dw4(05:19-05:26)非空闲线程全栈;含时间窗口与目标段。"""
import re, collections, sys

text = open("C:/Users/33423/AppData/Local/Temp/dw4_local.log", encoding="utf-8", errors="replace").read()
TS = r"^(\d{4}-\d{2}-\d{2}T(?:\d{2}:\d{2}):\d{2}\.\d+)"

blocks = []
cur = None
last_ts = ""
for ln in text.splitlines():
    m = re.match(TS + r"\s?(.*)$", ln)
    if m:
        last_ts, body = m.group(1), re.sub(r"^Z\s*", "", m.group(2))
    else:
        body = ln
    mh = re.match(r"(?:Current )?[Tt]hread 0x([0-9a-f]+)", body)
    if mh:
        if cur and cur[2]:
            blocks.append(cur)
        cur = [last_ts, mh.group(1), [body]]
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

groups = collections.OrderedDict()
for ts, tid, fr in blocks:
    if not ts:
        continue
    groups.setdefault(ts[:19], []).append((ts, tid, fr))

seen = set()
for g, lst in sorted(groups.items()):
    hh = g[11:]
    if hh < "05:22:30" or hh in seen:
        continue
    seen.add(hh)
    print(f"\n===== DUMP @ {g}  ({len(lst)} threads) =====")
    for ts, tid, fr in lst:
        s = sig(fr)
        if IDLE.search("\n".join(fr)):
            continue
        print(f"\n-- tid {tid}")
        for f in fr:
            print("   " + f)
