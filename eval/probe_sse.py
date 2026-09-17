# -*- coding: utf-8 -*-
"""单轮 SSE 全事件抓取:复现 recall_direct 空答案。"""
import json, sys, urllib.request

BASE = "http://127.0.0.1:3000"
USER, PWD = "memtest01", "Memtest@2026"

def http_json(path, payload):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)

try:
    tok = http_json("/api/auth/login", {"username": USER, "password": PWD})["token"]
except urllib.error.HTTPError:
    tok = http_json("/api/auth/register", {"username": USER, "password": PWD})["token"]

msg = sys.argv[1] if len(sys.argv) > 1 else "我们厂3号冷却塔的年度维保窗口是什么时候?现场联系人是谁?"
req = urllib.request.Request(BASE + "/api/chat",
    data=json.dumps({"message": msg, "history": [],
                     "thread_id": "memtest-20260914-a"}).encode(),
    headers={"Content-Type": "application/json", "Authorization": f"Bearer {tok}"},
    method="POST")
with urllib.request.urlopen(req, timeout=300) as resp:
    for raw in resp:
        line = raw.decode("utf-8", "replace").strip()
        if line.startswith("data: "):
            print(line[:600], flush=True)
