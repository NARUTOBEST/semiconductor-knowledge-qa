# -*- coding: utf-8 -*-
"""采集 dev200 全部题目在当前检索配置下的完整重排分数(ranked),供 τ/K 离线扫描。

用法: python collect_ranked.py --out runs/dev200_gemma_ranked.jsonl
每行: {id, q, source, items(生产截断后), ranked(截断前全量 gemma 分)}
"""
import argparse
import glob
import json
import os
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
TOKEN = ""
BASE = "http://127.0.0.1:8002"
LOCK = threading.Lock()


def search(query):
    body = json.dumps({"query": query, "k": 8, "score_ratio": 0.01,
                       "with_pool": True, "with_ranked": True}).encode()
    req = urllib.request.Request(
        BASE + "/search_text", data=body,
        headers={"Content-Type": "application/json",
                 "X-Internal-Token": TOKEN})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


def job(rec, out_path):
    try:
        d = search(rec["q"])
        rec["items"] = d.get("items") or []
        rec["ranked"] = d.get("ranked") or []
    except Exception as e:
        rec["error"] = repr(e)
    with LOCK:
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec.get("id")


def main():
    global TOKEN
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "runs", "ranked.jsonl"))
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    TOKEN = _token()
    done = set()
    if os.path.exists(args.out):
        for line in open(args.out, encoding="utf-8"):
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                pass
    recs = []
    for f in sorted(glob.glob(os.path.join(
            HERE, "runs", "dev200_baseline", "qa_sets", "user*.json"))):
        for q in json.load(open(f, encoding="utf-8")):
            if q["id"] not in done:
                recs.append({"id": q["id"], "q": q["q"],
                             "source": q.get("source")})
    print(f"待采集 {len(recs)} 题", flush=True)
    with ThreadPoolExecutor(args.workers) as ex:
        for i, cid in enumerate(ex.map(lambda r: job(r, args.out), recs), 1):
            if i % 20 == 0 or i == len(recs):
                print(f"  {i}/{len(recs)}", flush=True)
    print("== saved ->", args.out)


def _token():
    env = os.path.join(HERE, "..", "..", "..", "..", "env", "env.env")
    env = os.path.normpath(env)
    for line in open(env, encoding="utf-8"):
        if line.startswith("RETRIEVAL_INTERNAL_TOKEN="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("token not found")


if __name__ == "__main__":
    main()
