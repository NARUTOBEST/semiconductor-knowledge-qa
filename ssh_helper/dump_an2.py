#!/usr/bin/env python3
"""带时间戳重新抓 docker logs,按挂死窗口(04:08-04:30 UTC)逐 dump 统计线程栈。"""
import re, collections

# 1) 抓带时间戳日志(外部已生成 /tmp/dwt.log)
try:
    text = open("/tmp/dwt.log", encoding="utf-8", errors="replace").read()
except FileNotFoundError:
    raise SystemExit("no /tmp/dwt.log")

TS = r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+)"
WIN_LO, WIN_HI = "2026-10-10T04:06", "2026-10-10T04:32"

lines = text.splitlines()
# 每行:timestamp line 或 续行(无时间戳)
# thread 块:以 Thread 0x / Current thread 0x 开头(可能带时间戳前缀)
blocks = []   # (ts, [frames])
cur = None
for ln in lines:
    m = re.match(TS + r"\s?(.*)$", ln)
    ts, body = (m.group(1), m.group(2)) if m else (None, ln)
    body = re.sub(r"^Z\s*", "", body)
    if body.startswith("Thread 0x") or body.startswith("Current thread 0x"):
        if cur and cur[1]:
            blocks.append(cur)
        cur = [ts or (blocks[-1][0] if blocks else ""), [body]]
    elif cur is not None:
        if body.startswith("  File ") or body.startswith("File "):
            cur[1].append(body.strip())
        else:
            if cur[1]:
                blocks.append(cur)
            cur = None
    # 其他行忽略
if cur and cur[1]:
    blocks.append(cur)

# 过滤窗口
def inwin(ts):
    return ts and WIN_LO <= ts <= WIN_HI

win = [b for b in blocks if inwin(b[0])]
print(f"blocks total={len(blocks)} in-window={len(win)}")

def sig(fr):
    return " <- ".join(f.split("/")[-1] for f in fr[1:])

IDLE = re.compile(
    r"tool_pool\.py.*57 in _worker|selectors\.py.*select"
    r"|anyio/_backends/_asyncio\.py.*1022 in run"
    r"|concurrent/futures/thread\.py.*90 in _worker"
    r"|mcp_bridge\.py.*_run|lifecycle\.py.*prune_loop|multiprocess\.py.*pong"
    r"|queue\.py.*171 in get|_queue\.py.*64 in wait|_queue\.py.*114 in wait"
)

cnt = collections.Counter()
ex = {}
for ts, fr in win:
    s = sig(fr)
    if IDLE.search("\n".join(fr)):
        cnt["IDLE:" + s.split(" <- ")[0]] += 1
        continue
    cnt[s] += 1
    ex.setdefault(s, (ts, fr))

for s, n in cnt.most_common(50):
    print(f"\n### x{n}  {s[:400]}")
    if n <= 4 and s in ex:
        ts, fr = ex[s]
        print(f"   first@ {ts}")
        for f in fr:
            print("   " + f)
