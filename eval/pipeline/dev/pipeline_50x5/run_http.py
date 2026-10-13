# -*- coding: utf-8 -*-
"""内网纯 HTTP 评测器(单用户串行,反问自动应答)。

与 run_concurrent.py 同口径(直连 POST /api/chat,LAN),轮次9 增量:clarify 事件
自动应答——检索冲突反问的选项按题目金文档 stem 归一化模糊匹配(difflib)点选,
等价于前端 ClarifyCard onClick(option → onSend):选项文本作为新消息、原问题与
反问句入 history 重发,scope 锁按 history 首条用户消息匹配。无浏览器依赖。

用法: python run_http.py --ids-file runs/playwright_badcase_ids.txt --name badcase_http
"""
import argparse
import difflib
import json
import os
import re
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
SETS_DIR = os.path.join(HERE, "runs", "dev200_baseline", "qa_sets")

_NORM_RE = re.compile(r"[\s_\-()（）\[\]【】.。·]")


def _norm(s):
    return _NORM_RE.sub("", (s or "").lower())


def pick_option(options, source):
    """按金文档 stem 与选项文本的归一化模糊匹配挑最像的。"""
    if not options:
        return None, 0.0
    src = _norm(source)
    best, best_r = None, -1.0
    for opt in options:
        r = difflib.SequenceMatcher(None, _norm(opt), src).ratio()
        if r > best_r:
            best, best_r = opt, r
    return best, round(best_r, 3)


def http_json(base, path, payload, timeout=30):
    req = urllib.request.Request(
        base + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def ensure_user(base, username, password):
    """注册(409→登录),限流退避重试。"""
    last = None
    for attempt in range(20):
        try:
            try:
                return http_json(base, "/api/auth/register",
                                 {"username": username,
                                  "password": password})["token"]
            except urllib.error.HTTPError as e:
                if e.code != 409:
                    raise
                return http_json(base, "/api/auth/login",
                                 {"username": username,
                                  "password": password})["token"]
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (429,) or e.code >= 500:
                time.sleep(min(15 * (attempt + 1), 60))
                continue
            raise
        except (urllib.error.URLError, OSError):
            time.sleep(min(15 * (attempt + 1), 60))
    raise RuntimeError(f"{username}: register/login 重试耗尽 ({last})")


def chat_once(base, token, message, history, timeout_s=600):
    """单轮 POST /api/chat,返回该轮 SSE 事件列表。"""
    req = urllib.request.Request(
        base + "/api/chat",
        data=json.dumps({"message": message, "history": history}).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {token}"},
        method="POST")
    events = []
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            try:
                events.append(json.loads(line[6:]))
            except Exception:
                continue
    return events


def pick_option_event(ev, source):
    """从 clarify 事件挑选项。轮次12 起选项是候选**对象名**(非文档名),
    事件附带 option_stems(对象所属文档族)——评测按 stem 对金文档模糊匹配
    (等价于知道金文档的用户);旧口径(无 option_stems)退回选项文本匹配。"""
    opts = ev.get("options") or []
    stems = ev.get("option_stems") or []
    if stems and len(stems) == len(opts):
        best_stem, ratio = pick_option(stems, source or "")
        if best_stem:
            return opts[stems.index(best_stem)], ratio
        return None, 0.0
    return pick_option(opts, source or "")


def derive_clarify_reply(source):
    """反问无选项化(轮次20):预置用户应答的明确指代——取金文档族名原文,
    去掉入库侧分片尾缀(_NN)。等价于"知道自己要问哪台设备的用户"。"""
    return re.sub(r"[_\s]*\d+$", "", (source or "").strip()).strip()


def ask_once(base, token, question, source, timeout_s=600,
             clarify_reply=None):
    """一次完整提问(含反问自动应答)。字段口径同 run_concurrent。
    轮次20:无选项反问时回预置明确指代(clarify_reply,缺省从 source 派生);
    有选项事件(旧口径后端)仍走 pick_option_event 点选。"""
    if clarify_reply is None:
        clarify_reply = derive_clarify_reply(source)
    out = {"answer": "", "sources": [], "tier": None, "meta": None,
           "error": None, "done": False, "clarify_shown": [],
           "clarify_picked": None, "clarify_matched": None}
    seen = {}
    done_seen = False
    rounds = 0
    msg, history = question, []
    while rounds < 3:
        rounds += 1
        events = chat_once(base, token, msg, history, timeout_s)
        last_clarify = None
        for ev in events:
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
        # 轮次20:无选项反问 → 回预置明确指代;有选项(旧口径)→ 按金文档点选
        if not (last_clarify.get("options") or []):
            opt, ratio = (clarify_reply or ""), None
            if not opt:
                out["error"] = out["error"] or "clarify without options/reply"
                break
        else:
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://192.168.88.138:3000")
    ap.add_argument("--user", default="evalhttp1")
    ap.add_argument("--password", default="Eval#pass1")
    ap.add_argument("--sets-dir", default=SETS_DIR)
    ap.add_argument("--ids", default="")
    ap.add_argument("--ids-file", default="")
    ap.add_argument("--out", default=os.path.join(HERE, "runs", "http_badcase"))
    ap.add_argument("--name", default="results")
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()

    ids = [x.strip() for x in args.ids.split(",") if x.strip()]
    if args.ids_file:
        with open(args.ids_file, encoding="utf-8") as f:
            ids += [x.strip() for x in f if x.strip() and not x.startswith("#")]
    if not ids:
        raise SystemExit("no ids given (--ids / --ids-file)")

    pool = {}
    for fn in os.listdir(args.sets_dir):
        if fn.endswith(".json"):
            for it in json.load(open(os.path.join(args.sets_dir, fn),
                                     encoding="utf-8")):
                pool[it["id"]] = it
    items = []
    for i in ids:
        it = pool.get(i)
        if not it:
            raise SystemExit(f"unknown id: {i}")
        items.append(it)
    print(f"[plan] {len(items)} 题: {[it['id'] for it in items]}", flush=True)

    token = ensure_user(args.base, args.user, args.password)
    print(f"[auth] {args.user} ok", flush=True)

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"{args.name}.jsonl")
    done_ids = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if not r.get("error") and r.get("done"):
                        done_ids.add(r["id"])
                except Exception:
                    pass

    for it in items:
        if it["id"] in done_ids:
            print(f"[skip] {it['id']} (已跑)", flush=True)
            continue
        t0 = time.time()
        rec = {"id": it["id"], "q": it["q"], "source": it.get("source"),
               "facts": it.get("facts"), "tier_expected": it.get("tier"),
               "user": args.user}
        # 重试:撞用户锁退避(后端锁 TTL 600s);流断(无 done 无答案)重发
        for attempt in range(6):
            if attempt:
                rec.update({"answer": "", "sources": [], "tier": None,
                            "meta": None, "error": None, "done": False,
                            "clarify_shown": [], "clarify_picked": None,
                            "clarify_matched": None})
                time.sleep(min(60 * (attempt + 1), 300))
            try:
                r = ask_once(args.base, token, it["q"], it.get("source") or "",
                             timeout_s=args.timeout)
                rec.update(r)
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
            msg_txt = rec.get("error") or ""
            busy = "进行中的请求" in msg_txt or "繁忙" in msg_txt
            broken = not rec.get("done") and not rec.get("answer") and not msg_txt
            if (busy or broken) and attempt < 5:
                continue
            if busy and attempt >= 5:
                rec["error"] = rec["error"] or "busy after retries"
            break
        rec["wall_s"] = round(time.time() - t0, 1)
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[done] {it['id']} {rec['wall_s']}s tier={rec.get('tier')} "
              f"srcs={len(rec.get('sources') or [])} "
              f"clarify={len(rec.get('clarify_shown') or [])} "
              f"err={1 if rec.get('error') else 0}", flush=True)
    print("HTTP EVAL DONE", flush=True)


if __name__ == "__main__":
    main()
