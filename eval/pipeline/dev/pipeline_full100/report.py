# -*- coding: utf-8 -*-
"""评测报告:per-question 明细 JSON(bad case 标注失败类型)+ 控制台汇总。"""
from __future__ import annotations

import json
import os

from . import metrics as M

# bad case 阈值(可按需调;先给经验值,用 eval 结果校准)
LOW_FAITHFULNESS = 0.7
LOW_FACT_COVERAGE = 0.6
LOW_ANSWER_RELEVANCY = 0.6
LOW_RETRIEVAL_SCORE = 0.5

_REFUSAL_MARKERS = ("未找到", "暂未找到", "没有找到", "未覆盖", "未收录", "暂无相关", "联系")


def tag_bad_case(r: dict) -> list[str]:
    """给单题打失败类型标签(可多个)。空列表=无明显问题。"""
    tags = []
    answer = r.get("answer") or ""
    sources = r.get("sources") or []
    fact_cov = r.get("fact_coverage", 1.0)
    max_score = r.get("retrieval_max_score", 0.0)
    searched = r.get("search_count", 0)

    if r.get("clarified"):
        tags.append("反问澄清")          # 系统没直接答,而是反问(未必是坏,单列)
    if r.get("final_reason") in ("error", "timeout"):
        tags.append(f"运行异常({r.get('final_reason')})")
    if r.get("expect_tier") and r.get("tier") != r.get("expect_tier"):
        tags.append(f"路由错(期望{r.get('expect_tier')}/实得{r.get('tier')})")

    is_react = (r.get("tier") == "react" or r.get("expect_tier") == "react")
    if is_react:
        if searched > 0 and not sources:
            tags.append("检索无来源")
        elif searched > 0 and max_score < LOW_RETRIEVAL_SCORE:
            tags.append(f"检索低置信(max_score={max_score:.2f})")
        if not answer.strip():
            tags.append("空答案")
        elif any(mk in answer for mk in _REFUSAL_MARKERS):
            tags.append("拒答/未覆盖")

    if fact_cov < LOW_FACT_COVERAGE:
        tags.append(f"事实点覆盖不足({fact_cov:.0%})")
    if r.get("faithfulness") is not None and r["faithfulness"] < LOW_FAITHFULNESS:
        tags.append(f"不忠实({r['faithfulness']:.2f})")
    if r.get("answer_relevancy") is not None and r["answer_relevancy"] < LOW_ANSWER_RELEVANCY:
        tags.append(f"答非所问({r['answer_relevancy']:.2f})")
    if r.get("expect_source") and not r.get("source_hit"):
        tags.append("来源书未命中")
    return tags


def write_reports(records: list[dict], out_dir: str, stamp: str) -> dict:
    """写 per-question 明细 + 汇总 JSON,打印控制台汇总。返回汇总 dict。"""
    os.makedirs(out_dir, exist_ok=True)
    for r in records:
        r["bad_tags"] = tag_bad_case(r)
        r["is_bad"] = bool(r["bad_tags"])

    summary = M.aggregate(records)
    bad = [r for r in records if r["is_bad"]]
    summary["bad_case_count"] = len(bad)
    summary["bad_case_rate"] = round(len(bad) / max(1, len(records)), 4)

    detail_path = os.path.join(out_dir, f"eval-detail-{stamp}.json")
    summary_path = os.path.join(out_dir, f"eval-summary-{stamp}.json")
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    _print_console(summary, bad, detail_path, summary_path)
    return summary


def _print_console(summary: dict, bad: list[dict], detail_path: str, summary_path: str):
    line = "=" * 64
    print("\n" + line)
    print("离线评测汇总(RAGAS 式 + 硬指标)")
    print(line)
    print(f"题量 n={summary['n']}  bad case={summary['bad_case_count']}"
          f"({summary['bad_case_rate']:.0%})")
    for k in ("ragas_faithfulness", "ragas_context_precision",
              "ragas_answer_relevancy", "ragas_context_recall"):
        if k in summary:
            print(f"  {k:28s}: {summary[k]}")
    print(f"  {'fact_coverage_avg':28s}: {summary['fact_coverage_avg']}")
    print(f"  {'source_hit_rate':28s}: {summary['source_hit_rate']}")
    if summary.get("tier_accuracy") is not None:
        print(f"  {'tier_accuracy':28s}: {summary['tier_accuracy']}")
    print(f"  {'latency p50/p95/avg (s)':28s}: "
          f"{summary['latency_p50_s']} / {summary['latency_p95_s']} / {summary['latency_avg_s']}")
    print(f"  {'redo/escalation rate':28s}: "
          f"{summary['redo_rate']} / {summary['escalation_rate']}")
    print(f"  {'avg_search_count':28s}: {summary['avg_search_count']}")
    print(f"  {'final_reason 分布':28s}: {summary['final_reason_dist']}")
    print(line)
    if bad:
        print(f"bad case 明细(共 {len(bad)} 条,前 20):")
        for r in bad[:20]:
            print(f"  [id={r.get('id')}] {r.get('q', '')[:34]}  ->  {'; '.join(r['bad_tags'])}")
    print(f"\n明细: {detail_path}\n汇总: {summary_path}\n")
