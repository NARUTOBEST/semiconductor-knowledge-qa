# -*- coding: utf-8 -*-
"""20 用户并发评测执行器(仅标准库,在项目机上运行)。

与 run_http.py(单用户串行)同口径:走真实 HTTP 链路——N 个线程各模拟 1 个用户,
按各自 qa_set 逐题请求 POST /api/chat(SSE)。每题收集:最终答案、全部 sources
事件条目(含分数,按分数降序=检索名次)、tier、meta、错误。结果逐题追加
results/<user>.jsonl,断点续跑。轮次13 起与 run_http.py 同步:检索冲突反问
自动应答(选项按金文档 stem 匹配点选,复用 run_http.pick_option_event)。

用法: python3 run_concurrent.py --base http://127.0.0.1:3000
"""
import argparse
import json
import os
import threading
import time
import urllib.error
import urllib.request

from run_http import pick_option_event

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))  # eval/(脚本在 pipeline/dev/pipeline_* 内,4 层 dirname)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # pipeline_50x5/(runs 在此)
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
    for attempt in range(20):
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
            if e.code == 429 or e.code >= 500:  # 限流/nginx 5xx:退避后重试
                time.sleep(min(15 * (attempt + 1), 60))
                continue
            raise
        except (urllib.error.URLError, OSError) as e:  # 网络抖动也退避重试
            last = e
            time.sleep(min(15 * (attempt + 1), 60))
            continue
    raise RuntimeError(f"{username}: register/login 重试耗尽 ({last})")


def ask_once(base, token, question, source="", timeout_s=600):
    """一次完整提问(含检索冲突反问自动应答,口径同 run_http.ask_once)。"""
    out = {"answer": "", "sources": [], "tier": None, "meta": None,
           "error": None, "done": False, "clarify_shown": [],
           "clarify_picked": None, "clarify_matched": None}
    seen = {}
    done_seen = False
    rounds = 0
    msg, history = question, []
    while rounds < 3:
        rounds += 1
        req = urllib.request.Request(
            base + "/api/chat",
            data=json.dumps({"message": msg, "history": history}).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {token}"},
            method="POST")
        last_clarify = None
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
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
                    done_seen = True
                elif t == "clarify":
                    last_clarify = ev
                    out["clarify_shown"].append(
                        {"question": ev.get("question"),
                         "options": ev.get("options") or []})
        if out["error"] or not last_clarify:
            break
        opt, ratio = pick_option_event(last_clarify, source or "")
        out["clarify_picked"] = opt
        out["clarify_matched"] = ratio
        if not opt:
            out["error"] = out["error"] or "clarify without options"
            break
        history = history + [
            {"role": "user", "content": msg},
            {"role": "assistant", "content": last_clarify.get("question") or ""},
        ]
        msg = opt
    out["done"] = done_seen
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
        # 并发限流退避:per-user 锁被慢流占用时后端 429(错误事件或 HTTP 429),
        # 退避后重试;流中途断掉(无 done 且无答案)也重试。
        for attempt in range(8):
            try:
                r = ask_once(base, token, it["q"], it.get("source") or "")
                rec.update(r)
            except Exception as e:
                code = getattr(e, "code", None)
                if code == 429 and attempt < 7:
                    time.sleep(min(15 * (attempt + 1), 60))
                    continue
                rec["error"] = repr(e)
                break
            msg = rec.get("error") or ""
            if ("进行中的请求" in msg or "繁忙" in msg) and attempt < 7:
                rec.update({"answer": "", "sources": [], "tier": None,
                            "meta": None, "error": None, "done": False,
                            "clarify_shown": [], "clarify_picked": None,
                            "clarify_matched": None})
                time.sleep(min(15 * (attempt + 1), 60))
                continue
            if not rec.get("done") and not rec.get("answer") and not msg:
                if attempt < 7:            # 流中途断掉且无内容:重试
                    rec.update({"answer": "", "sources": [], "tier": None,
                                "meta": None, "error": None, "done": False})
                    time.sleep(5)
                    continue
                rec["error"] = "stream ended without done and empty answer"
            break
        rec["wall_s"] = round(time.time() - t0, 1)
        with LOCK:
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n += 1
        print(f"[{username}] {n}/{len(qa_set)} id={it['id']} {rec['wall_s']}s "
              f"tier={rec.get('tier')} srcs={len(rec.get('sources') or [])} "
              f"clarify={len(rec.get('clarify_shown') or [])} "
              f"err={1 if rec.get('error') else 0}", flush=True)
    print(f"[{username}] FINISHED", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:3000")
    ap.add_argument("--sets-dir", default=os.path.join(BASE_DIR, "runs", "dev200_baseline", "qa_sets"))
    ap.add_argument("--out", default=os.path.join(HERE, "runs", "adhoc"))
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
