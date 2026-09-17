# -*- coding: utf-8 -*-
"""50x5 并发回归测试跑批器(一键:跑测 → LLM 裁判 → 与基线对比)。

用法(项目根):
    python -m eval.regression_50x5                       # 全量 50 用户
    python -m eval.regression_50x5 --users 5             # 冒烟(5 用户)
    python -m eval.regression_50x5 --skip-run            # 只重判/重比对现有结果
    python -m eval.regression_50x5 --update-baseline     # 用本次结果刷新基线

流程:
  1. run_concurrent.py    真实 HTTP 链路 50 用户 x 5 题(qa_sets_50x5,断点续跑)
  2. judge_50x5.py        RAGAS 式 LLM 裁判(faithfulness/precision/recall/切题)
  3. concurrent_metrics.py 确定性硬指标(文档级 P/R/F1/NDCG/MRR + tier 准确率)
  4. 与 eval/regression_baseline.json 逐项对比,任一指标退化超过容差 -> 退出码 1

基线首次由 2026-09-17 全量运行固化(方舟按量 ep-8vvns 主 / ep-44sxb 副,
AutoDL 3080Ti 检索 + SSH 隧道,VM docker 部署)。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BASELINE = os.path.join(HERE, "regression_baseline.json")

# 退化容差(绝对值):质量指标允许 ±TOL 浮动,超出判为回归
TOL = 0.05
# 对比口径: {key: (来源, 方向)} 方向 up=越高越好 / down=越低越好
JUDGE_KEYS = {
    "avg_faithfulness": "up",
    "avg_context_precision": "up",
    "avg_context_recall": "up",
    "avg_answer_relevancy": "up",
    "hallucination_rate": "down",
}
RUN_KEYS = {
    "completion_rate": "up",
    "sources_hit_rate": "up",
    "error_rate": "down",
}
HARD_KEYS = {  # 文档级检索硬指标(metrics_report.json overall)
    "mrr": "up",
    "P@5": "up",
    "R@5": "up",
    "F1@5": "up",
    "NDCG@5": "up",
    "tier_ok": "up",
}


def sh(args, env=None):
    print(f"\n== $ {' '.join(args)}", flush=True)
    e = dict(os.environ)
    e.setdefault("NO_PROXY", "192.168.88.138,127.0.0.1,localhost,.volces.com")
    e.setdefault("no_proxy", e["NO_PROXY"])
    if env:
        e.update(env)
    r = subprocess.run([sys.executable] + args, cwd=ROOT, env=e)
    if r.returncode != 0:
        sys.exit(f"步骤失败(退出码 {r.returncode}): {args}")


def load_latest(results_dir):
    """按 (user,id) 去重取最后一次写入。"""
    latest = {}
    for f in glob.glob(os.path.join(results_dir, "eval*.jsonl")):
        for line in open(f, encoding="utf-8"):
            try:
                r = json.loads(line)
                if str(r.get("id", "")).startswith("u") and r.get("user"):
                    latest[(r["user"], r["id"])] = r
            except Exception:
                pass
    return list(latest.values())


def run_stats(recs):
    n = len(recs)
    wall = [r["wall_s"] for r in recs if isinstance(r.get("wall_s"), (int, float))]
    return {
        "n": n,
        "completion_rate": round(sum(1 for r in recs if r.get("done")) / n, 4) if n else None,
        "error_rate": round(sum(1 for r in recs if r.get("error")) / n, 4) if n else None,
        "sources_hit_rate": round(sum(1 for r in recs if r.get("sources")) / n, 4) if n else None,
        "latency_median_s": round(st.median(wall), 1) if wall else None,
        "latency_p95_s": round(sorted(wall)[int(len(wall) * .95) - 1], 1) if wall else None,
    }


def judge_stats(recs):
    judged = [r for r in recs if r.get("judge_done")]
    # 知识题=有参考要点的题;闲聊题(无 facts,无可校验主张)剔除出幻觉/忠实度口径
    kn = [r for r in judged if not r.get("judge_chitchat")]

    def avg(k, pool):
        # 统计名 avg_faithfulness -> 记录字段 judge_faithfulness
        field = "judge_" + k.replace("avg_", "", 1)
        vals = [r[field] for r in pool if r.get(field) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None
    hr = round(sum(1 for r in kn if r.get("has_hallucination")) / len(kn), 4) \
        if kn else None
    return {"judged": len(judged), "hallucination_rate": hr,
            **{k: avg(k, kn if k == "avg_faithfulness" else judged)
               for k in JUDGE_KEYS if k.startswith("avg_")}}


def hard_stats(results_dir):
    p = os.path.join(results_dir, "metrics_report.json")
    if not os.path.exists(p):
        return {}
    overall = json.load(open(p, encoding="utf-8")).get("overall", {})
    return {k: overall[k] for k in HARD_KEYS if k in overall}


def compare(current, baseline):
    rows = []
    regressed = False
    for section, keys in (("judge", JUDGE_KEYS), ("run", RUN_KEYS),
                          ("hard", HARD_KEYS)):
        cur, base = current.get(section) or {}, baseline.get(section) or {}
        for k, direction in keys.items():
            c, b = cur.get(k), base.get(k)
            if c is None or b is None:
                rows.append((f"{section}.{k}", b, c, "skip(缺值)"))
                continue
            delta = round(c - b, 4)
            bad = (delta < -TOL) if direction == "up" else (delta > TOL)
            regressed |= bad
            rows.append((f"{section}.{k}", b, c,
                         f"{'REGRESS ' if bad else ''}{delta:+.4f}"))
    return rows, regressed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://192.168.88.138:3000")
    ap.add_argument("--sets-dir", default=os.path.join(HERE, "qa_sets_50x5"))
    ap.add_argument("--users", type=int, default=50)
    ap.add_argument("--out", default=None, help="结果目录(默认 eval/results_regress)")
    ap.add_argument("--skip-run", action="store_true", help="跳过跑测,复用现有结果")
    ap.add_argument("--no-qdrant", action="store_true", default=True,
                    help="硬指标不取 qdrant 全文(默认;事实级 Rfact 不可用时跳过)")
    ap.add_argument("--update-baseline", action="store_true")
    args = ap.parse_args()

    out = args.out or os.path.join(HERE, "results_regress")
    os.makedirs(out, exist_ok=True)

    if not args.skip_run:
        sh(["-m", "eval.run_concurrent", "--base", args.base,
            "--sets-dir", args.sets_dir, "--users", str(args.users),
            "--out", out])
    sh(["-m", "eval.judge_50x5", "--results-dir", out, "--out-dir", out])
    sh(["-m", "eval.concurrent_metrics", "--results-dir", out,
        "--sets-dir", args.sets_dir, "--users", str(args.users),
        "--no-qdrant"])

    recs = load_latest(out)
    current = {
        "run": run_stats(recs),
        "judge": judge_stats(recs),
        "hard": hard_stats(out),
    }
    print("\n===== 本次结果 =====")
    print(json.dumps(current, ensure_ascii=False, indent=1))

    if args.update_baseline:
        current["date"] = __import__("datetime").date.today().isoformat()
        current["config"] = {
            "base": args.base, "users": args.users,
            "sets_dir": os.path.basename(args.sets_dir),
        }
        json.dump(current, open(BASELINE, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print(f"== 基线已更新 -> {BASELINE}")
        return

    if not os.path.exists(BASELINE):
        sys.exit(f"未找到基线 {BASELINE};首次请加 --update-baseline 固化基线")
    baseline = json.load(open(BASELINE, encoding="utf-8"))
    rows, regressed = compare(current, baseline)
    print(f"\n===== 对比基线({baseline.get('date', '?')},容差 ±{TOL}) =====")
    print(f"{'指标':32s} {'基线':>8s} {'本次':>8s}  变化")
    for name, b, c, note in rows:
        print(f"{name:32s} {b!s:>8s} {c!s:>8s}  {note}")
    if regressed:
        print("\n结果: 存在退化(退出码 1)")
        sys.exit(1)
    print("\n结果: 通过")


if __name__ == "__main__":
    main()
