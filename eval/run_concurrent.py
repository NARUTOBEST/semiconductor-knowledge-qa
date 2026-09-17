# -*- coding: utf-8 -*-
"""5 用户并发评测执行器(仅标准库,在项目机上运行)。

与 eval/run_eval.py(单用户白盒直调)不同:本脚本走真实 HTTP 链路——
5 个线程各模拟 1 个用户,按各自 100 题测试集逐题请求 POST /api/chat(SSE)。
每题收集:最终答案、全部 sources 事件条目(含分数,按分数降序=检索名次)、
tier、meta、错误。结果逐题追加 results/<user>.jsonl,断点续跑。

用法: python3 run_concurrent.py --base http://127.0.0.1:3000
"""
import argparse
import json
import os
import threading
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
LOCK = threading.Lock()
USERS = [(f"eval{i}", "Eval#pass1") for i in range(1, 51)]


def http_json(base, path, payload, token=None, timeout=30):
    req = urllib.request.Request(
        base + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def ensure_user(base, username, password):
    """注册(409→登录),429 退避重试;串行调用避免并发限流。"""
    last = None
    for attempt in range(10):
        try:
            try:
                return http_json(base, "/api/auth/register",
                                 {"username": username,
                                  "password": password})["token"]
            except urllib.error.HTTPError as e:
                if e.code != 409:      # 已存在 → 走登录
                    raise
                return http_json(base, "/api/auth/login",
                                 {"username": username,
                                  "password": password})["token"]
        except urllib.error.HTTPError as e:
            last = e
            if e.code == 429:          # 登录/注册限流:退避后重试
                time.sleep(15 * (attempt + 1))
                continue
            raise
    raise RuntimeError(f"{username}: register/login 重试耗尽 ({last})")


def ask_once(base, token, question):
    req = urllib.request.Request(
        base + "/api/chat",
        data=json.dumps({"message": question, "history": []}).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {token}"},
        method="POST")
    out = {"answer": "", "sources": [], "tier": None, "meta": None,
           "error": None, "done": False}
    seen = {}
    with urllib.request.urlopen(req, timeout=600) as resp:
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
                    key = s.get("chunk_id") or (s.get("source_stem", "") +
                                                s.get("page", ""))
                    if not key:
                        continue
                    prev = seen.get(key)
                    if prev is None or float(s.get("score") or 0) > float(
                            prev.get("score") or 0):
                        seen[key] = s
            elif t == "assistant_message":
                out["answer"] = ev.get("content", "")
            elif t == "tier":
                out["tier"] = ev.get("tier")
            elif t == "meta":
                out["meta"] = ev
            elif t == "error":
                out["error"] = ev.get("message")
            elif t == "done":
                out["done"] = True
    out["sources"] = sorted(seen.values(),
                            key=lambda s: float(s.get("score") or 0),
                            reverse=True)
    return out


def run_user(base, username, token, qa_set, out_path):
    done_ids = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["id"])
                except Exception:
                    pass
    n = 0
    for it in qa_set:
        if it["id"] in done_ids:
            n += 1
            continue
        t0 = time.time()
        rec = {"id": it["id"], "q": it["q"], "source": it.get("source"),
               "facts": it.get("facts"), "tier_expected": it.get("tier"),
               "user": username}
        try:
            r = ask_once(base, token, it["q"])
            rec.update(r)
        except Exception as e:
            rec["error"] = repr(e)
        rec["wall_s"] = round(time.time() - t0, 1)
        with LOCK:
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n += 1
        print(f"[{username}] {n}/{len(qa_set)} id={it['id']} {rec['wall_s']}s "
              f"tier={rec.get('tier')} srcs={len(rec.get('sources') or [])} "
              f"err={1 if rec.get('error') else 0}", flush=True)
    print(f"[{username}] FINISHED", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:3000")
    ap.add_argument("--sets-dir", default=os.path.join(HERE, "qa_sets"))
    ap.add_argument("--out", default=os.path.join(HERE, "results_concurrent"))
    ap.add_argument("--users", type=int, default=50)
    ap.add_argument("--shuffle", action="store_true",
                    help="用户内用例随机打乱(固定种子,可复现)")
    ap.add_argument("--limit", type=int, default=0,
                    help="每用户只取前 N 题(0=全量)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    users = USERS[:args.users]
    tokens = {}                       # 串行拿 token,避免并发注册/登录限流
    for username, password in users:
        tokens[username] = ensure_user(args.base, username, password)
        print(f"[auth] {username} ok", flush=True)
        time.sleep(2)
    ths = []
    for i, (username, password) in enumerate(users, 1):
        qa_set = json.load(open(os.path.join(args.sets_dir, f"user{i}.json"),
                                encoding="utf-8"))
        if args.shuffle:              # 每用户独立乱序,种子含用户号保证可复现
            import random
            random.Random(2026 + i).shuffle(qa_set)
        if args.limit > 0:            # 每用户只取前 N 题(评测口径:前 50)
            qa_set = qa_set[:args.limit]
        th = threading.Thread(
            target=run_user,
            args=(args.base, username, tokens[username], qa_set,
                  os.path.join(args.out, f"{username}.jsonl")),
            daemon=True)
        th.start()
        ths.append(th)
        time.sleep(3)   # 错开首题,平滑并发
    for th in ths:
        th.join()
    print("CONCURRENT EVAL DONE")


if __name__ == "__main__":
    main()
