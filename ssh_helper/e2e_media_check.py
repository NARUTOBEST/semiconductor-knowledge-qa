# -*- coding: utf-8 -*-
"""E2E 媒体验证:登录 → 提问 → 解析 sources 的 image_url/video_url → 逐个 GET 验 200。"""
import json
import subprocess
import urllib.request

BASE = "http://127.0.0.1:3000"

def http(url, data=None, headers=None, timeout=120):
    req = urllib.request.Request(url, data=json.dumps(data).encode() if data else None,
                                 headers=headers or {})
    return urllib.request.urlopen(req, timeout=timeout)

# 登录
try:
    tok = json.loads(http(f"{BASE}/api/auth/login",
                          {"username": "eval1", "password": "Eval#pass1"},
                          {"Content-Type": "application/json"}).read())["token"]
except Exception:
    tok = json.loads(http(f"{BASE}/api/auth/register",
                          {"username": "eval1", "password": "Eval#pass1"},
                          {"Content-Type": "application/json"}).read())["token"]

questions = [
    # 图片向:设备结构/照片类
    "BESI Datacon 2200 固晶机的结构图或设备照片是什么样的",
    # 视频向:工艺/操作类(视频块多为工艺讲解)
    "固晶机贴片工艺的操作过程视频讲解",
]
for qi, q in enumerate(questions, 1):
    print(f"\n===== Q{qi}: {q}")
    resp = http(f"{BASE}/api/chat", {"message": q, "history": []},
                {"Content-Type": "application/json",
                 "Authorization": f"Bearer {tok}"}, timeout=180)
    srcs, ans = [], ""
    for raw in resp:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data: "):
            continue
        try:
            ev = json.loads(line[6:])
        except Exception:
            continue
        if ev.get("type") == "sources":
            srcs.extend(ev.get("items", []))
        elif ev.get("type") == "assistant_message":
            ans = ev.get("content", "")
    imgs = [s.get("image_url") for s in srcs if s.get("image_url")]
    vids = [s.get("video_url") for s in srcs if s.get("video_url")]
    types = [s.get("item_type", "") for s in srcs]
    print("sources:", len(srcs), "item_type:", types[:8])
    print("answer 前120字:", ans[:120].replace("\n", " "))
    print("image_url 数:", len(imgs), "| video_url 数:", len(vids))
    ok = bad = 0
    for u in (imgs + vids)[:6]:
        host = u.split("/")[2] if "://" in u else "?"
        try:
            r = urllib.request.urlopen(urllib.request.Request(u, method="GET"), timeout=30)
            code, ctype = r.status, r.headers.get("Content-Type", "")
            if 200 <= code < 300:
                ok += 1
            else:
                bad += 1
        except Exception as e:
            code, ctype = getattr(e, "code", type(e).__name__), ""
            bad += 1
        print(f"  [{code}] {ctype} {host} ...{u[-40:]}")
    print(f"媒体 GET: ok={ok} bad={bad}")
