# -*- coding: utf-8 -*-
"""补测并发轮次中检索故障的题(react 但 sources 为空),单用户串行。

读取 runs/dev200_round8 找出废样本,按 id 从 qa_sets 取题目重答,
写入新目录(断点续跑安全)。合并由 merge_round8.py 完成。

用法: python rerun_broken.py
"""
import glob
import json
import os

import run_concurrent as rc

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "runs", "dev200_round8")
SETS = os.path.join(HERE, "runs", "dev200_baseline", "qa_sets")
OUT = os.path.join(HERE, "runs", "dev200_round8_fix")
BASE = "http://192.168.88.138:3000"
USER, PWD = "evalfix", "evalfix123"


def main():
    os.makedirs(OUT, exist_ok=True)
    broken = set()
    for f in glob.glob(os.path.join(SRC, "*.jsonl")):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            if r.get("tier") == "react" and not (r.get("sources") or []):
                broken.add(r["id"])
    print(f"broken: {len(broken)}")

    qa = []
    for f in sorted(glob.glob(os.path.join(SETS, "user*.json"))):
        for it in json.load(open(f, encoding="utf-8")):
            if it["id"] in broken:
                qa.append(it)
    print(f"matched: {len(qa)}")

    token = rc.ensure_user(BASE, USER, PWD)
    print("[auth] ok", flush=True)
    rc.run_user(BASE, USER, token, qa, os.path.join(OUT, f"{USER}.jsonl"))
    print("RERUN BROKEN DONE")


if __name__ == "__main__":
    main()
