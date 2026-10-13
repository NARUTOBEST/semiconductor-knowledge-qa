#!/usr/bin/env python3
"""分析 docker logs 中的 faulthandler 线程栈转储:分类空闲/可疑线程块。"""
import re, sys, collections

LOG = "/tmp/dw.log"
IDLE_PAT = re.compile(
    r"tool_pool\.py\", line 57 in _worker"          # tool_pool 空闲 worker
    r"|selectors\.py\", line .* in select"          # asyncio epoll
    r"|anyio/_backends/_asyncio\.py\", line 1022 in run"  # anyio to_thread 空闲
    r"|concurrent/futures/thread\.py\", line 90 in _worker"  # cf 空闲
    r"|mcp_bridge\.py\", line .* in _run"           # mcp 重连循环
    r"|threading\.py\", line 355 in wait\n.*queue\.py\", line 171 in get"  # queue.get
    r"|asyncio/locks\.py"                            # asyncio 原语(待看调用方)
)

def blocks(text):
    lines = text.splitlines()
    cur = None
    for ln in lines:
        m = re.match(r"^(Current )?[Tt]hread 0x[0-9a-f]+", ln)
        if m or ln.startswith("Thread 0x") or ln.startswith("Current thread 0x"):
            if cur:
                yield cur
            cur = [ln]
        elif cur is not None:
            if ln.startswith('  File '):
                cur.append(ln)
            else:
                # 非栈行:块结束
                if len(cur) > 1:
                    yield cur
                cur = None
    if cur and len(cur) > 1:
        yield cur

def key(block):
    return "\n".join(block)

def sig(block):
    """签名:去掉线程 id,保留文件:行"""
    body = [re.sub(r"\s+", " ", l.strip()) for l in block[1:]]
    return " <- ".join(f.split('/')[-1] for f in body)

with open(LOG, encoding="utf-8", errors="replace") as f:
    text = f.read()

allb = list(blocks(text))
print(f"total blocks: {len(allb)}")

idle, sus = 0, collections.Counter()
examples = {}
for b in allb:
    k = key(b)
    if IDLE_PAT.search(k):
        idle += 1
    else:
        s = sig(b)
        sus[s] += 1
        examples.setdefault(s, k)

print(f"idle: {idle}, suspicious-unique: {len(sus)}")
for s, n in sus.most_common(40):
    print(f"\n===== x{n} =====")
    print(s)
    if n <= 3:
        print(examples[s])
