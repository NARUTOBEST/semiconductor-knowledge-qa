#!/usr/bin/env python3
"""dw4t:所有含 queue.get/Condition.wait/Event.wait 的线程,打印完整栈(含 /app 调用者)。"""
import re, collections

text = open("C:/Users/33423/AppData/Local/Temp/dw4t.log", encoding="utf-8", errors="replace").read()
TS = r"^(\d{4}-\d{2}-\d{2}T(\d{2}:\d{2}):\d{2}\.\d+)"

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

cnt = collections.Counter()
ex = {}
for ts, tid, fr in blocks:
    if not ts or ts[11:16] < "05:22":
        continue
    joined = "\n".join(fr)
    if re.search(r"line 355 in wait|line 171 in get|line 451 in result|line 390 in result|app_events|_app", joined):
        sig = " <- ".join(f.split("/")[-1] for f in fr[1:])
        # 归一:去掉纯底层等待,保留有 /app 帧的
        if "/app/" in joined:
            key = sig[:300]
            cnt[key] += 1
            ex.setdefault(key, (ts, tid, fr))

for s, n in cnt.most_common(20):
    ts, tid, fr = ex[s]
    print(f"\n### x{n} first@{ts} tid {tid}")
    for f in fr:
        print("   " + f)
