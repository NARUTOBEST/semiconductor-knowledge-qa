# -*- coding: utf-8 -*-
"""20 并发跑批补跑驱动:清掉坏记录(error/空答案/无来源),按 5 并发分波
顺序补跑,直到无缺口(最多 3 轮)。复用 run_concurrent 的 run_user 断点去重。
用法: python tmp_retry_bad.py --out runs/20261007_round10_20u200
"""
import argparse, glob, json, os, sys, threading, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_concurrent import ensure_user, run_user, USERS  # noqa: E402

BASE = "http://192.168.88.138:3000"
SETS_DIR = os.path.join(HERE, "runs", "dev200_20u", "qa_sets")


def clean_bad(out_dir):
    """删掉 error/空答案/无来源 的记录,返回删除数。"""
    removed = 0
    for f in glob.glob(os.path.join(out_dir, "eval*.jsonl")):
        good = []
        for line in open(f, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("error") or not (r.get("answer") or "").strip() \
                    or not r.get("sources"):
                removed += 1
                continue
            good.append(line)
        with open(f, "w", encoding="utf-8") as fh:
            fh.write("\n".join(good) + ("\n" if good else ""))
    return removed


def count_all(out_dir):
    n = 0
    for f in glob.glob(os.path.join(out_dir, "eval*.jsonl")):
        n += sum(1 for l in open(f, encoding="utf-8") if l.strip())
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--wave", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    removed = clean_bad(args.out)
    print(f"cleaned bad records: {removed}", flush=True)
    for rnd in range(1, args.rounds + 1):
        missing = 200 - count_all(args.out)
        print(f"== round {rnd}: missing {missing}", flush=True)
        if missing <= 0:
            break
        users = USERS[:20]
        for w in range(0, 20, args.wave):
            wave = users[w:w + args.wave]
            ths = []
            for i, (username, password) in enumerate(wave):
                uidx = w + i + 1
                token = ensure_user(BASE, username, password)
                qa_set = json.load(open(
                    os.path.join(SETS_DIR, f"user{uidx}.json"),
                    encoding="utf-8"))
                th = threading.Thread(
                    target=run_user,
                    args=(BASE, username, token, qa_set,
                          os.path.join(args.out, f"{username}.jsonl")),
                    daemon=True)
                th.start()
                ths.append(th)
                time.sleep(3)
            for th in ths:
                th.join()
            print(f"   wave {w // args.wave + 1} done, total={count_all(args.out)}",
                  flush=True)
    print(f"RETRY DONE total={count_all(args.out)}/200", flush=True)


if __name__ == "__main__":
    main()
