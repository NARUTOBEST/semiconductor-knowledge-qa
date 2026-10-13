# -*- coding: utf-8 -*-
"""单用户串行评测(项目规范:不并发,见 C:\\project3\\CLAUDE.md)。

把 qa_sets 全部分片(user*.json)合并为单一队列,一个账号顺序问完。
复用 run_concurrent 的 ask_once/run_user 重试与落盘逻辑。

用法: python run_single.py --base http://192.168.88.138:3000 \
        --out runs/dev200_round8_single
"""
import argparse
import glob
import json
import os

import run_concurrent as rc

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:3000")
    ap.add_argument("--sets-dir",
                    default=os.path.join(HERE, "runs", "dev200_baseline",
                                         "qa_sets"))
    ap.add_argument("--out", default=os.path.join(HERE, "runs", "adhoc_single"))
    ap.add_argument("--username", default="evalsingle")
    ap.add_argument("--password", default="evalsingle123")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    qa_set = []
    for f in sorted(glob.glob(os.path.join(args.sets_dir, "user*.json")),
                    key=lambda p: int("".join(ch for ch in
                                              os.path.basename(p)
                                              if ch.isdigit()))):
        qa_set.extend(json.load(open(f, encoding="utf-8")))
    print(f"merged {len(qa_set)} questions from {args.sets_dir}", flush=True)

    token = rc.ensure_user(args.base, args.username, args.password)
    print("[auth] ok", flush=True)
    rc.run_user(args.base, args.username, token, qa_set,
                os.path.join(args.out, f"{args.username}.jsonl"))
    print("SINGLE-USER EVAL DONE")


if __name__ == "__main__":
    main()
