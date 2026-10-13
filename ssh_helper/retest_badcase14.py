# -*- coding: utf-8 -*-
"""badcase14 重测:混合重排(Qwen3+m3)后 14 题检索 top-5 对比。
在 GPU 机上跑(直连 127.0.0.1:8002),输出 ssh_helper 可比对的 JSON。
用法: python retest_badcase14.py /root/autodl-tmp/badcase14_baseline.json > out.json
"""
import json
import sys
import urllib.request

BASE = "http://127.0.0.1:8002"
TOKEN = "rt-7f3a9c2e5b1d486fa0e4d3c2b1a09f87"


def search(query, k=6):
    body = json.dumps({"query": query, "k": k, "with_pool": True}).encode()
    req = urllib.request.Request(
        BASE + "/search_text", data=body,
        headers={"Content-Type": "application/json",
                 "X-Internal-Token": TOKEN})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def main():
    baseline = json.load(open(sys.argv[1], encoding="utf-8"))
    out = {}
    for qid, rec in sorted(baseline.items()):
        q = rec["q"]
        res = search(q, k=6)
        items = res if isinstance(res, list) else res.get("items", [])
        pool = res.get("pool", []) if isinstance(res, dict) else []
        out[qid] = {
            "q": q,
            "baseline_top5": [
                {"chunk_id": s["chunk_id"], "stem": s.get("source_stem", ""),
                 "content": s.get("content", "")} for s in rec.get("sources", [])],
            "new_top6": [
                {"chunk_id": it.get("chunk_id"), "stem": it.get("source_stem", ""),
                 "score": it.get("score"), "content": it.get("content", "")}
                for it in items],
            "new_pool_ids": [p.get("chunk_id") for p in pool],
        }
        print(f"{qid}: pool={len(pool)} top6={[i.get('chunk_id') for i in items]}",
              flush=True)
    json.dump(out, open("/root/autodl-tmp/badcase14_new.json", "w",
                        encoding="utf-8"), ensure_ascii=False, indent=1)
    print("== saved /root/autodl-tmp/badcase14_new.json")


if __name__ == "__main__":
    main()
