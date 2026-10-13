# -*- coding: utf-8 -*-
"""Playwright 浏览器现场交互评测器(单用户串行)。

与 run_concurrent.py(白盒 HTTP POST /api/chat)不同:本脚本驱动真实浏览器走
用户路径——登录态注入 → 新对话 → 输入题目 → 发送 → (若出现澄清反问卡)按
题目金文档自动点选选项 → 等 SSE done。数据捕获不走 DOM:page.on("response")
拦截 /api/chat 的 SSE 响应体,按 run_concurrent.ask_once 同一套解析
(sources 去重保最高分 / assistant_message / tier / meta / done),
输出 JSONL 与 run_concurrent 完全兼容,judge_50x5.py 可直接判分。

反问应答策略(用户已确认):模拟"知道自己问什么的用户"——按该题 source
(金文档 stem)与选项文本做归一化模糊匹配(difflib),点最像的;匹配不到
点第一个。额外记录 clarify_shown / clarify_picked / clarify_matched。

用法:
  python run_playwright.py --ids-file ids_a.txt --name subset_a
  python run_playwright.py --ids dev_067,dev_084,dev_171 --name smoke
"""
import argparse
import difflib
import json
import os
import re
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))  # pipeline_50x5/(runs/ 也在此)
SETS_DIR = os.path.join(HERE, "runs", "dev200_baseline", "qa_sets")


def http_json(base, path, payload, timeout=30):
    req = urllib.request.Request(
        base + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def ensure_user(base, username, password):
    """注册(409→登录)。"""
    try:
        return http_json(base, "/api/auth/register",
                         {"username": username, "password": password})["token"]
    except urllib.error.HTTPError as e:
        if e.code != 409:
            raise
        return http_json(base, "/api/auth/login",
                         {"username": username, "password": password})["token"]


def parse_sse(body: bytes):
    """SSE 响应体 → 事件列表(与 run_concurrent.ask_once 同一套字段口径)。"""
    events = []
    for raw in body.decode("utf-8", "replace").splitlines():
        line = raw.strip()
        if not line.startswith("data: "):
            continue
        try:
            events.append(json.loads(line[6:]))
        except Exception:
            continue
    return events


def merge_events(rec, events):
    """把一轮 SSE 事件并入记录:sources 跨轮去重保最高分,其余取最后值。"""
    seen = rec.setdefault("_seen", {})
    for ev in events:
        t = ev.get("type")
        if t == "sources":
            for s in ev.get("items", []):
                key = s.get("chunk_id") or (s.get("source_stem", "") + s.get("page", ""))
                if not key:
                    continue
                prev = seen.get(key)
                if prev is None or float(s.get("score") or 0) > float(prev.get("score") or 0):
                    seen[key] = s
        elif t == "assistant_message":
            rec["answer"] = ev.get("content", "")
        elif t == "tier":
            rec["tier"] = ev.get("tier")
        elif t == "meta":
            rec["meta"] = ev
        elif t == "error":
            rec["error"] = ev.get("message")
        elif t == "done":
            rec["done"] = True
        elif t == "clarify":
            rec.setdefault("clarify_shown", []).append(
                {"question": ev.get("question"), "options": ev.get("options") or []})


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


def ask_in_browser(page, question, source, timeout_s=600):
    """一轮完整提问(含多轮反问应答)。返回记录 dict。"""
    rec = {"answer": "", "sources": [], "tier": None, "meta": None,
           "error": None, "done": False, "clarify_shown": [],
           "clarify_picked": None, "clarify_matched": None, "turns": 0}
    page.evaluate("() => { window.__chat_resps = 0 }")

    # 拦截 /api/chat 响应:每轮一个 Response,流结束才能取 body
    resps = []

    def on_response(resp):
        if "/api/chat" in resp.url:
            resps.append(resp)

    page.on("response", on_response)
    try:
        ta = page.locator("textarea")
        ta.click()
        ta.fill(question)
        page.locator('button[title="发送"]').click()

        deadline = time.time() + timeout_s
        processed = 0            # 已取 body 的响应数
        clarify_rounds = 0
        while time.time() < deadline:
            if processed >= len(resps):
                page.wait_for_timeout(500)
                continue
            resp = resps[processed]
            body = resp.body()   # 阻塞到该轮流结束(SSE done),天然等待点
            processed += 1
            rec["turns"] += 1
            events = parse_sse(body)
            merge_events(rec, events)
            if rec["done"] or rec["error"]:
                break
            # 反问卡:点选项后进入下一轮(新一轮 response 会出现在 resps)
            last_clarify = next((e for e in reversed(events) if e.get("type") == "clarify"), None)
            if last_clarify:
                clarify_rounds += 1
                if clarify_rounds > 3:
                    rec["error"] = "clarify loop >3"
                    break
                opt, ratio = pick_option(last_clarify.get("options") or [], source)
                rec["clarify_picked"] = opt
                rec["clarify_matched"] = ratio
                clicked = False
                if opt:
                    try:
                        page.get_by_role("button", name=opt, exact=True).first.click(timeout=10000)
                        clicked = True
                    except Exception:
                        pass
                if not clicked:  # 兜底:澄清卡里第一个按钮
                    try:
                        card_btn = page.locator(
                            'div.rounded-xl.border-\\[\\#3a3a6e\\] button').first
                        card_btn.click(timeout=10000)
                        rec["clarify_picked"] = rec["clarify_picked"] or "(fallback first)"
                    except Exception as e:
                        rec["error"] = f"clarify click failed: {type(e).__name__}"
                        break
                continue
            # 没有反问也没有 done:流异常结束,记错误退出
            if not rec["done"]:
                rec["error"] = rec["error"] or "turn ended without done/clarify"
            break
        else:
            rec["error"] = rec["error"] or f"timeout {timeout_s}s"
    finally:
        page.remove_listener("response", on_response)

    rec["sources"] = sorted(rec.pop("_seen", {}).values(),
                            key=lambda s: float(s.get("score") or 0),
                            reverse=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://192.168.88.138:3000")
    ap.add_argument("--user", default="evalpw1")
    ap.add_argument("--password", default="Eval#pass1")
    ap.add_argument("--sets-dir", default=SETS_DIR)
    ap.add_argument("--ids", default="", help="逗号分隔题目 id(如 dev_067)")
    ap.add_argument("--q", default="", help="自由提问(冒烟用);--source 配合金文档 stem")
    ap.add_argument("--source", default="", help="--q 模式下的金文档 stem")
    ap.add_argument("--ids-file", default="", help="每行一个题目 id 的文件")
    ap.add_argument("--out", default=os.path.join(HERE, "runs", "adhoc_playwright"))
    ap.add_argument("--name", default="results", help="输出文件名(不含扩展名)")
    ap.add_argument("--timeout", type=int, default=600, help="单题超时秒")
    ap.add_argument("--headed", action="store_true", help="有头模式(可视化 watching)")
    args = ap.parse_args()

    ids = [x.strip() for x in args.ids.split(",") if x.strip()]
    if args.ids_file:
        with open(args.ids_file, encoding="utf-8") as f:
            ids += [x.strip() for x in f if x.strip() and not x.startswith("#")]
    if not ids and not args.q:
        raise SystemExit("no ids given (--ids / --ids-file / --q)")

    # 合并全部 qa_sets 分片建 id→题目映射(--q 冒烟模式跳过)
    pool = {}
    if not args.q:
        for fn in os.listdir(args.sets_dir):
            if fn.endswith(".json"):
                for it in json.load(open(os.path.join(args.sets_dir, fn), encoding="utf-8")):
                    pool[it["id"]] = it
    items = []
    if args.q:
        items.append({"id": "adhoc", "q": args.q, "source": args.source,
                      "facts": None, "tier": None})
    for i in ids:
        it = pool.get(i)
        if not it:
            raise SystemExit(f"unknown id: {i}")
        items.append(it)
    print(f"[plan] {len(items)} 题: {[it['id'] for it in items]}", flush=True)

    token = ensure_user(args.base, args.user, args.password)
    print(f"[auth] {args.user} ok", flush=True)

    from playwright.sync_api import sync_playwright

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"{args.name}.jsonl")
    done_ids = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if not r.get("error") and r.get("done"):
                        done_ids.add(r["id"])   # 只有成功记录才算已完成
                except Exception:
                    pass

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=not args.headed)
        ctx = browser.new_context(viewport={"width": 1440, "height": 900})
        ctx.set_default_timeout(600000)
        # 登录态注入:在任何页面脚本运行前写 localStorage(绕过登录页)
        user_obj = json.dumps({"username": args.user}, ensure_ascii=False)
        ctx.add_init_script(
            f"window.localStorage.setItem('token', {json.dumps(token)});"
            f"window.localStorage.setItem('user', {json.dumps(user_obj)});")
        page = ctx.new_page()
        page.goto(args.base, wait_until="domcontentloaded")
        page.locator('button[title="新对话"]').first.wait_for(timeout=30000)
        print("[browser] ready", flush=True)

        for it in items:
            if it["id"] in done_ids:
                print(f"[skip] {it['id']} (已跑)", flush=True)
                continue
            t0 = time.time()
            page.locator('button[title="新对话"]').first.click()
            page.wait_for_timeout(500)
            rec = {"id": it["id"], "q": it["q"], "source": it.get("source"),
                   "facts": it.get("facts"), "tier_expected": it.get("tier"),
                   "user": args.user}
            # 重试:撞用户锁(前序请求滞留)退避后重发;流断(无 done 无答案)重发。
            # 后端用户锁 TTL 600s,退避上限 300s 内足够等到自愈。
            for attempt in range(6):
                if attempt:
                    page.locator('button[title="新对话"]').first.click()
                    page.wait_for_timeout(500)
                    rec.update({"answer": "", "sources": [], "tier": None,
                                "meta": None, "error": None, "done": False,
                                "clarify_shown": [], "clarify_picked": None,
                                "clarify_matched": None, "turns": 0})
                try:
                    rec.update(ask_in_browser(page, it["q"], it.get("source") or "",
                                              timeout_s=args.timeout))
                except Exception as e:
                    rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
                msg = rec.get("error") or ""
                retryable = ("进行中的请求" in msg or "繁忙" in msg
                             or not rec.get("done"))
                if retryable and attempt < 5:
                    time.sleep(min(60 * (attempt + 1), 300))
                    continue
                break
            rec.pop("_seen", None)
            rec["wall_s"] = round(time.time() - t0, 1)
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"[done] {it['id']} {rec['wall_s']}s tier={rec.get('tier')} "
                  f"srcs={len(rec.get('sources') or [])} "
                  f"clarify={len(rec.get('clarify_shown') or [])} "
                  f"err={1 if rec.get('error') else 0}", flush=True)
        browser.close()
    print("PLAYWRIGHT EVAL DONE", flush=True)


if __name__ == "__main__":
    main()
