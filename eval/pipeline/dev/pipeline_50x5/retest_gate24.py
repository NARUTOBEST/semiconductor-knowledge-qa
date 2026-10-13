# -*- coding: utf-8 -*-
"""轮次13 伪引用守门重测:串行重答 round13d 无 sources 的伪引用题,验证 finalize
守门("⚠️ 说明:本轮知识库检索未返回可用资料…")是否生效。仅标准库。

用法: python retest_gate24.py --base http://192.168.88.138:3000
"""
import argparse
import json
import os
import sys

from run_concurrent import ask_once, ensure_user

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "runs", "20261009_round13d_20u200")
OUT_DIR = os.path.join(HERE, "runs", "20261009_gate24_retest")
OUT = os.path.join(OUT_DIR, "gate24.jsonl")


def load_fab():
    """round13d 无 sources 的题(id 去重,保留原题干/金文档)。"""
    import glob
    import re
    fab_ids = [
        "dev_041", "dev_061", "dev_121", "dev_181", "dev_070", "dev_090",
        "dev_071", "dev_152", "dev_056", "dev_156", "dev_077", "dev_137",
        "dev_177", "dev_019", "dev_082", "dev_020", "dev_003", "dev_043",
        "dev_063", "dev_004", "dev_085", "dev_125", "dev_106", "dev_007",
        "dev_027"]
    recs = {}
    for f in glob.glob(os.path.join(SRC, "eval*.jsonl")):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            if r["id"] in fab_ids and r["id"] not in recs:
                recs[r["id"]] = {"id": r["id"], "q": r["q"],
                                 "source": r.get("source", ""),
                                 "old_answer": r.get("answer", ""),
                                 "old_sources": r.get("sources", [])}
    missing = [i for i in fab_ids if i not in recs]
    if missing:
        print("WARN missing:", missing)
    return [recs[i] for i in fab_ids if i in recs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://192.168.88.138:3000")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT, encoding="utf-8"):
            done.add(json.loads(line)["id"])
    recs = [r for r in load_fab() if r["id"] not in done]
    print(f"todo={len(recs)} (done={len(done)})", flush=True)
    token = ensure_user(args.base, "eval1", "Eval#pass1")
    for i, r in enumerate(recs):
        t0 = __import__("time").time()
        try:
            res = ask_once(args.base, token, r["q"], r["source"])
        except Exception as e:
            res = {"answer": "", "sources": [], "tier": None, "meta": None,
                   "error": f"{type(e).__name__}: {e}", "done": False,
                   "clarify_shown": [], "clarify_picked": None,
                   "clarify_matched": None}
        r["new_answer"] = res.get("answer", "")
        r["new_sources"] = [
            {k: s.get(k) for k in ("chunk_id", "source_stem", "page",
                                   "score")}
            for s in res.get("sources", [])]
        r["new_error"] = res.get("error")
        r["tier"] = res.get("tier")
        r["gate_declared"] = res.get("answer", "").startswith("⚠️")
        with open(OUT, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[{i+1}/{len(recs)}] {r['id']} src={len(r['new_sources'])} "
              f"gate={r['gate_declared']} {__import__('time').time()-t0:.0f}s",
              flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
