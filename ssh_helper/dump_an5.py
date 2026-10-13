#!/usr/bin/env python3
"""追踪关键 tid 的存活区间:流式读/tick/headers 等待/闸等待。"""
import re, collections

text = open("/tmp/dwt.log", encoding="utf-8", errors="replace").read()
TS = r"^(\d{4}-\d{2}-\d{2}T(?:\d{2}:\d{2}):\d{2}\.\d+)"

blocks = []
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

PATS = {
    "STREAM_BODY": r"iter_bytes|_receive_response_body",
    "TICK": r"_runner\.py\", line 283 in tick",
    "HEADERS": r"_receive_response_headers",
    "GATE": r"acquire_global_slot",
    "HEARTBEAT": r"_heartbeat_gen",
    "CONNECT": r"create_connection",
}
CAT = [(k, re.compile(v)) for k, v in PATS.items()]

def catof(fr):
    joined = "\n".join(fr)
    for k, p in CAT:
        if p.search(joined):
            return k
    return None

seq = collections.defaultdict(list)
for ts, tid, fr in blocks:
    if not ts or not ("04:10" <= ts[11:16] <= "04:34"):
        continue
    c = catof(fr)
    if c:
        seq[(tid, c)].append(ts)

print("tid / cat: count first..last")
rows = []
for (tid, c), tss in seq.items():
    rows.append((min(tss), max(tss), tid, c, len(tss)))
for r in sorted(rows):
    print(f"{r[0][11:19]} .. {r[1][11:19]}  {r[3]:12s} x{r[4]:3d}  tid {r[2]}")
