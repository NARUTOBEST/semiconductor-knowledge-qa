#!/usr/bin/env python3
"""纵向追踪:窗口内每个线程 ID 的签名序列,找持续卡死的线程。"""
import re, collections

text = open("/tmp/dwt.log", encoding="utf-8", errors="replace").read()
TS = r"^(\d{4}-\d{2}-\d{2}T(?:\d{2}:\d{2}):\d{2}\.\d+)"
WIN_LO, WIN_HI = "04:06", "04:34"

sightings = collections.defaultdict(list)  # tid -> [(ts, sig)]
last_ts = None
for ln in text.splitlines():
    m = re.match(TS + r"\s?(.*)$", ln)
    if m:
        last_ts, body = m.group(2), re.sub(r"^Z\s*", "", m.group(3)) if False else re.sub(r"^Z\s*", "", m.group(1 + 1))
        # group(1)=HH:MM  实际重写:
        last_ts = m.group(1)
        body = re.sub(r"^Z\s*", "", m.group(2))
    else:
        body = ln
    mh = re.match(r"(?:Current )?[Tt]hread 0x([0-9a-f]+)", body)
    if mh:
        tid = mh.group(1)
        sightings[tid].append([last_ts, []])
        continue
    if body.startswith("  File ") and sightings:
        # 归属最近出现的 tid
        # (同一次 dump 中线程头连续出现;Frame 行属于上一个线程头)
        pass
# 上面的逐行归属不可靠,改为块状解析

blocks = []  # (ts, tid, frames)
cur = None
for ln in text.splitlines():
    m = re.match(TS + r"\s?(.*)$", ln)
    if m:
        last_ts = m.group(1)
        body = re.sub(r"^Z\s*", "", m.group(2))
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

def inwin(ts):
    hhmm = ts[11:16] if ts else ""
    return WIN_LO <= hhmm <= WIN_HI

def sig(fr):
    return " <- ".join(f.split("/")[-1] for f in fr[1:])

IDLE = re.compile(
    r"tool_pool\.py.*57 in _worker|selectors\.py.*select"
    r"|anyio/_backends/_asyncio\.py.*1022 in run"
    r"|concurrent/futures/thread\.py.*90 in _worker"
    r"|mcp_bridge\.py.*_run|lifecycle\.py.*prune_loop|multiprocess\.py.*pong"
    r"|queue\.py.*171 in get|_queue\.py.*64 in wait|_queue\.py.*114 in wait"
)

# tid -> 窗口内 [(ts, sig)]
seq = collections.defaultdict(list)
for ts, tid, fr in blocks:
    if not inwin(ts):
        continue
    s = sig(fr)
    if IDLE.search("\n".join(fr)):
        continue
    seq[tid].append((ts, s))

print("== 持续卡死候选(同一 tid 同一签名 >=4 次)==")
for tid, lst in sorted(seq.items(), key=lambda kv: -len(kv[1])):
    if len(lst) < 4:
        continue
    cnt = collections.Counter(s for _, s in lst)
    s, n = cnt.most_common(1)[0]
    if n < 4:
        continue
    ts0, ts1 = lst[0][0], lst[-1][0]
    print(f"\ntid {tid}: {n}x  {ts0}..{ts1}")
    print(f"  {s[:500]}")
    other = {k: v for k, v in cnt.items() if k != s}
    for k, v in other.items():
        print(f"  (+{v}x {k[:160]})")
