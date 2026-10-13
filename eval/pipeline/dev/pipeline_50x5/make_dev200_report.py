# -*- coding: utf-8 -*-
"""dev200_baseline 报告再生成:读新口径 judge_summary_50x5.json + 判后 jsonl,
输出 eval/runs/dev/20261002_dev200_baseline/{metrics_report.json,REPORT.md}。
含旧口径(首判 2026-10-02)对照、分题型/分层明细、剩余真幻觉清单。

每次运行自动把本轮指标追加到题集历史文件
eval/runs/dev/history/<题集名>_history.jsonl(指标与末条完全相同时跳过,防重复)。
"""
import json, os, glob, time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "runs", "dev200_baseline", "results")
META_PATH = os.path.normpath(os.path.join(HERE, "..", "..", "..", "qa_sets",
                                          "v2_20261001", "dev_200.json"))
OUT_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "..", "runs", "dev",
                                        "20261002_dev200_baseline"))
HISTORY_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "..", "runs",
                                            "dev", "history"))
OLD_SUMMARY = {  # 首判(旧口径,含判分冤案),仅作对照
    "avg_faithfulness": 0.7724, "avg_context_precision": 0.4662,
    "avg_context_recall": 0.5629, "avg_answer_relevancy": 0.9558,
    "hallucination_count": 46, "hallucination_rate": 0.2987,
}
_HIST_KEYS = ("judged", "avg_faithfulness", "avg_context_precision",
              "avg_context_recall", "avg_answer_relevancy",
              "hallucination_count", "hallucination_rate")


def append_history(summary, dataset_path, run, note=""):
    """把一轮判分指标追加进题集历史 jsonl;与末条指标完全相同则跳过。"""
    os.makedirs(HISTORY_DIR, exist_ok=True)
    stem = os.path.splitext(os.path.basename(dataset_path))[0]
    path = os.path.join(HISTORY_DIR, f"{stem}_history.jsonl")
    rec = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "run": run,
           "dataset": os.path.normpath(dataset_path),
           **{k: summary.get(k) for k in _HIST_KEYS},
           "refusal_bucket": summary.get("refusal_bucket"), "note": note}
    if os.path.exists(path):
        lines = open(path, encoding="utf-8").read().strip().splitlines()
        if lines:
            try:
                last = json.loads(lines[-1])
                if all(last.get(k) == rec[k] for k in _HIST_KEYS):
                    return path, False
            except Exception:
                pass
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return path, True


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=RESULTS,
                    help="本运行的判分结果目录(默认 dev200_baseline/results)")
    ap.add_argument("--out-dir", default=OUT_DIR,
                    help="报告输出目录(默认 runs/dev/20261002_dev200_baseline)")
    ap.add_argument("--run", default="dev200_baseline", help="历史记录里的 run 名")
    ap.add_argument("--note", default="", help="本轮备注(如 调参说明),写入历史记录")
    args = ap.parse_args()
    summary = json.load(open(os.path.join(args.results_dir,
                                          "judge_summary_50x5.json"),
                             encoding="utf-8"))
    meta = {q["query_id"]: q for q in json.load(open(META_PATH, encoding="utf-8"))}
    recs = {}
    for f in sorted(glob.glob(os.path.join(args.results_dir, "eval*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            try:
                r = json.loads(line)
                if r.get("judge_done"):
                    recs[r["id"]] = r   # 同 id 后写覆盖(重判结果在后)
            except Exception:
                pass
    kn = [r for r in recs.values()
          if not r.get("judge_chitchat") and not r.get("judge_refusal")]

    def bucket(rs, key):
        vals = [r[key] for r in rs if r.get(key) is not None]
        return {"n": len(rs), "avg": round(sum(vals) / len(vals), 4)} if vals else None

    by_type, by_tier = defaultdict(list), defaultdict(list)
    for r in kn:
        m = meta.get(r["id"], {})
        by_type[m.get("task_type", "?")].append(r)
        by_tier[str(m.get("expected_rounds"))].append(r)

    halluc = sorted((r for r in kn if r.get("has_hallucination")),
                    key=lambda r: r.get("judge_faithfulness") or 0)
    halluc_list = [{
        "id": r["id"], "task_type": meta.get(r["id"], {}).get("task_type"),
        "difficulty": meta.get(r["id"], {}).get("difficulty"),
        "expected_rounds": meta.get(r["id"], {}).get("expected_rounds"),
        "faithfulness": r.get("judge_faithfulness"),
        "q": (r.get("q") or "")[:60],
    } for r in halluc]

    rep = {
        "run": "dev200_baseline", "date": "2026-10-02",
        "dataset": "eval/qa_sets/v2_20261001/dev_200.json",
        "users": 10, "questions": 200, "errors": 0,
        "judge_methodology": "口径修正后重判(2026-10-02):算术推导算 supported、"
                             "定位用中间块算 relevant、拒答题(异常处理)剔除总口径单列",
        "metrics": summary,
        "old_metrics_first_judge": OLD_SUMMARY,
        "faithfulness_by_task_type": {t: bucket(rs, "judge_faithfulness")
                                      for t, rs in sorted(by_type.items())},
        "context_precision_by_task_type": {t: bucket(rs, "judge_context_precision")
                                           for t, rs in sorted(by_type.items())},
        "faithfulness_by_tier": {t: bucket(rs, "judge_faithfulness")
                                 for t, rs in sorted(by_tier.items())},
        "remaining_hallucinations": halluc_list,
    }
    os.makedirs(args.out_dir, exist_ok=True)
    json.dump(rep, open(os.path.join(args.out_dir, "metrics_report.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=2)

    om, nm = OLD_SUMMARY, summary
    lines = [
        "# dev-200 基线测评报告(2026-10-02,口径修正后重判)", "",
        "- 数据集: dev_200.json, 10 用户并发 × 20 题, **200/200 完成, 0 错误**",
        "- 判分口径修正: ① 数据计算的算术推导(如 总量=日用量×天数)视为资料可推出;"
        "② 定位答案的中间检索块(目录/导航/多跳跳板)计入 precision;"
        "③ 异常处理(期望拒答)题剔除总口径、单列 refusal_bucket。"
        "系统答案与检索结果未重跑,仅重判。", "",
        "## 旧口径 vs 干净口径", "",
        "| 指标 | 首判(旧口径) | 重判(干净口径) | Δ |",
        "|---|---|---|---|",
    ]
    for k, label in [("avg_faithfulness", "忠实度"),
                     ("avg_context_precision", "检索精确率"),
                     ("avg_context_recall", "检索召回率"),
                     ("avg_answer_relevancy", "切题度")]:
        o, n = om[k], nm[k]
        d = f"{n - o:+.4f}" if (o is not None and n is not None) else "-"
        lines.append(f"| {label} | {o} | {n} | {d} |")
    o, n = om["hallucination_rate"], nm["hallucination_rate"]
    d = f"{n - o:+.4f}" if (o is not None and n is not None) else "-"
    lines.append(f"| 幻觉率(阈值0.75) | {o} ({om['hallucination_count']}) | "
                 f"{n} ({nm['hallucination_count']}/{nm.get('refusal_excluded', 0)}拒答题已剔除) | {d} |")

    rb = nm.get("refusal_bucket") or {}
    if rb:
        lines += ["", f"> 拒答题单列(不计入总口径): n={rb.get('n')}, "
                      f"faithfulness={rb.get('avg_faithfulness')}, "
                      f"precision={rb.get('avg_context_precision')}"
                      f"(拒答场景下块\"无用\"是满分行为,低分为判分伪影)"]

    lines += ["", "## 分题型(干净口径)", "",
              "| 题型 | n | 忠实度 | 检索精确率 |", "|---|---|---|---|"]
    for t in sorted(by_type):
        f_, p_ = bucket(by_type[t], "judge_faithfulness"), bucket(by_type[t], "judge_context_precision")
        lines.append(f"| {t} | {f_['n']} | {f_['avg']} | {p_['avg'] if p_ else '-'} |")

    lines += ["", "## 分层忠实度(expected_rounds)", "",
              "| tier | n | avg |", "|---|---|---|"]
    for t in sorted(by_tier, key=lambda x: int(x)):
        b = bucket(by_tier[t], "judge_faithfulness")
        lines.append(f"| {t} | {b['n']} | {b['avg']} |")

    lines += ["", f"## 剩余真幻觉清单(干净口径,{len(halluc_list)} 条,待优化目标)", "",
              "| id | 题型 | 难度 | 轮数 | 忠实度 | 问题(截断) |",
              "|---|---|---|---|---|---|"]
    for h in halluc_list:
        lines.append(f"| {h['id']} | {h['task_type']} | {h['difficulty']} "
                     f"| {h['expected_rounds']} | {h['faithfulness']} | {h['q']} |")

    lines += ["", "> 说明: 忠实度/幻觉率/精确率分母已剔除闲聊题与拒答题;"
                 "检索召回率 context_recall 仍为真实短板(多跳题),优化优先级 recall >> precision。"]
    open(os.path.join(args.out_dir, "REPORT.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(f"== saved -> {os.path.normpath(os.path.join(args.out_dir, 'REPORT.md'))}")
    # 追加题集历史(与末条指标相同则自动跳过)
    hpath, appended = append_history(summary, META_PATH, args.run, args.note)
    print(("== history appended -> " if appended else "== history unchanged -> ")
          + hpath)
    print(json.dumps({"halluc_rate_new": nm["hallucination_rate"],
                      "faith_new": nm["avg_faithfulness"],
                      "halluc_new": nm["hallucination_count"],
                      "refusal_n": rb.get("n")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
