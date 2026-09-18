# -*- coding: utf-8 -*-
"""单用户单会话记忆链路测评:同一 thread_id 三轮对话。
turn1 种植事实 -> turn2 会话内回忆 -> turn3 指代性回忆。
仅 stdlib,在 VM 上运行:python3 mem_chain_test.py"""
import json
import time
import urllib.request

BASE = "http://127.0.0.1:3000"
USER = "memtest01"
PWD = "Memtest@2026"
THREAD = "memtest-20260914-a"

TURNS = [
    ("plant", "请记住一个信息:我们厂3号冷却塔的年度维保窗口是每年6月的第二周,"
              "现场联系人叫赵敏。请确认你已记住。"),
    ("recall_direct", "我们厂3号冷却塔的年度维保窗口是什么时候?现场联系人是谁?"),
    ("recall_anaphora", "我之前告诉过你的那位联系人叫什么名字?"),
]


def http_json(path, payload, token=None, timeout=30):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def ask(token, message, thread_id):
    req = urllib.request.Request(
        BASE + "/api/chat",
        data=json.dumps({"message": message, "history": [],
                         "thread_id": thread_id}).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {token}"},
        method="POST")
    out = {"answer": "", "tier": None, "meta": None, "n_sources": 0,
           "statuses": [], "error": None, "done": False}
    seen = set()
    with urllib.request.urlopen(req, timeout=300) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            try:
                ev = json.loads(line[6:])
            except Exception:
                continue
            t = ev.get("type")
            if t == "sources":
                for s in ev.get("items", []):
                    seen.add(s.get("chunk_id") or (s.get("source_stem", "") +
                                                   s.get("page", "")))
            elif t == "assistant_message":
                out["answer"] = ev.get("content", "")
            elif t == "tier":
                out["tier"] = ev.get("tier")
            elif t == "status":
                out["statuses"].append(ev.get("message", ""))
            elif t == "meta":
                out["meta"] = ev
            elif t == "error":
                out["error"] = ev.get("message")
            elif t == "done":
                out["done"] = True
    out["n_sources"] = len(seen)
    return out


def main():
    token = None
    for attempt in range(8):
        try:
            try:
                token = http_json("/api/auth/register",
                                  {"username": USER, "password": PWD})["token"]
                break
            except urllib.error.HTTPError as e:
                if e.code == 409:
                    token = http_json("/api/auth/login",
                                      {"username": USER, "password": PWD})["token"]
                    break
                if e.code == 429:
                    time.sleep(10 * (attempt + 1))
                    continue
                raise
        except Exception as e:
            print("auth retry:", repr(e))
            time.sleep(5)
    assert token, "auth failed"
    print("[auth] ok")

    results = []
    for name, msg in TURNS:
        t0 = time.time()
        try:
            r = ask(token, msg, THREAD)
        except Exception as e:
            r = {"error": repr(e), "answer": "", "statuses": []}
        r["wall_s"] = round(time.time() - t0, 1)
        r["turn"] = name
        results.append(r)
        print(f"\n===== {name} (wall={r['wall_s']}s) =====")
        print("Q:", msg)
        print("A:", (r.get("answer") or r.get("error") or "")[:500])
        print("tier:", r.get("tier"), "| sources:", r.get("n_sources"),
              "| statuses:", r.get("statuses", [])[-3:])
        time.sleep(10)  # 给后台记忆管道留处理窗口

    with open("/tmp/mem_chain_result.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\nRESULT saved to /tmp/mem_chain_result.json")


if __name__ == "__main__":
    main()
