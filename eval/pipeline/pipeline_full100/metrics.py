# -*- coding: utf-8 -*-
"""确定性硬指标(不调 LLM):关键词事实命中、来源命中、tier 路由、延迟分位、
final_reason 分布、重做/升级率,以及 RAGAS 三指标的汇总均值。

facts 字段语义(见 web/public/eval-qa.json):facts 是若干"事实组"的列表,每个
事实组是一组近义词(组内任一命中即视为该事实覆盖),跨组取覆盖率。例:
  [["料盒", "输出系统"], ["更换", "满"]] -> 需覆盖 2 个事实点,每个点命中近义词之一即可。
"""
from __future__ import annotations

import math
from collections import Counter


def cosine(a, b) -> float:
    """两个向量的余弦相似度;优先用 numpy,无 numpy 时纯 Python 回退。"""
    try:
        import numpy as np
        va = np.asarray(a, dtype="float32")
        vb = np.asarray(b, dtype="float32")
        na = float(np.linalg.norm(va))
        nb = float(np.linalg.norm(vb))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return float(np.dot(va, vb) / (na * nb))
    except Exception:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return dot / (na * nb)


def fact_coverage(answer: str, facts: list) -> tuple[float, list[bool]]:
    """返回 (覆盖率, 各事实组是否命中)。facts 为空(如 simple 闲聊题)返回 (1.0, [])。"""
    if not facts:
        return 1.0, []
    hits = []
    for group in facts:
        kws = [str(k) for k in (group or []) if str(k).strip()]
        hits.append(any(k in (answer or "") for k in kws))
    return sum(hits) / len(hits), hits


def source_hit(sources: list[dict], expect_source: str) -> bool:
    """期望来源名是否被某条检索来源的 source_stem 覆盖。

    期望标签常是简写(如"BESI固晶机"),而书名含型号/后缀(如
    "BESI Datacon 2200固晶机操作手册"),整串非连续、直接子串匹配会漏。
    故把期望名拆成拉丁/数字段与中文段(如 ["besi","固晶机"]),要求每段都能在
    归一化书名中找到(分段包含,容忍中间夹型号/空格/标点)。
    """
    if not expect_source:
        return True  # 该题未标注期望来源,不判负
    import re
    toks = re.findall(r"[a-z0-9]+|[一-鿿]+", str(expect_source).lower())
    if not toks:
        return True
    for s in sources or []:
        stem = _norm(str(s.get("source_stem", "")))
        if all(t in stem for t in toks):
            return True
    return False


def _norm(s: str) -> str:
    import re
    return re.sub(r"[^0-9a-z一-鿿]", "", str(s or "").lower())


def percentile(sorted_vals: list[float], q: float) -> float:
    """q∈[0,1];sorted_vals 需已升序。空列表返回 0.0。"""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    idx = q * (len(sorted_vals) - 1)
    lo = int(math.floor(idx))
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return float(sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac)


def aggregate(records: list[dict]) -> dict:
    """把每题的明细聚合成控制台汇总。record 字段见 run_eval.py。"""
    n = len(records)
    if n == 0:
        return {"n": 0}

    def _avg(key):
        vals = [r.get(key) for r in records if isinstance(r.get(key), (int, float))]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    lat = sorted(r.get("latency_s", 0.0) for r in records)
    tier_correct = sum(
        1 for r in records
        if not r.get("expect_tier") or r.get("tier") == r.get("expect_tier"))
    tier_total = sum(1 for r in records if r.get("expect_tier"))

    ragas_keys = ["faithfulness", "context_precision", "answer_relevancy", "context_recall"]
    ragas = {}
    for k in ragas_keys:
        vals = [r.get(k) for r in records if isinstance(r.get(k), (int, float))]
        if vals:
            ragas[k] = round(sum(vals) / len(vals), 4)

    return {
        "n": n,
        # RAGAS 式三指标(+可选 context_recall),只在跑了 judge 时存在
        **{f"ragas_{k}": v for k, v in ragas.items()},
        # 硬指标
        "fact_coverage_avg": _avg("fact_coverage"),
        "source_hit_rate": round(
            sum(1 for r in records if r.get("source_hit")) /
            max(1, sum(1 for r in records if r.get("expect_source"))), 4),
        "tier_accuracy": round(tier_correct / tier_total, 4) if tier_total else None,
        "latency_avg_s": round(sum(lat) / len(lat), 3),
        "latency_p50_s": round(percentile(lat, 0.50), 3),
        "latency_p95_s": round(percentile(lat, 0.95), 3),
        "redo_rate": round(sum(1 for r in records if r.get("redos")) / n, 4),
        "escalation_rate": round(sum(1 for r in records if r.get("escalations")) / n, 4),
        "avg_search_count": round(_avg("search_count"), 2),
        "final_reason_dist": dict(Counter(
            r.get("final_reason") or "unknown" for r in records)),
        "clarify_count": sum(1 for r in records if r.get("clarified")),
    }
