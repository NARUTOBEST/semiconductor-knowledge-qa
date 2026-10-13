# -*- coding: utf-8 -*-
"""离线批量评测入口:白盒直调 react_stream 跑 eval-qa.json,收集答案/来源/延迟/
重做升级,可选跑 RAGAS 式 LLM 裁判,产出明细 + 汇总报告。

前置(需服务在跑;judge 与 react 检索都依赖):
  - 检索微服务 :8002(BGE-m3 / rerank;judge 的 answer_relevancy 也用它)
  - LLM 网关 :4000 或云端 ARK(judge 用 MODEL_LIGHT)
  - Qdrant :6333
主后端 :8001 / Next 不需要(白盒直调,不走 HTTP/JWT/限流)。

用法(项目根,注意旁路代理):
  set NO_PROXY=127.0.0.1,localhost,.volces.com,.hf-mirror.com
  set HTTP_PROXY= & set HTTPS_PROXY=
  python -m eval.pipeline.pipeline_full100.run_eval                 # 全量 100 题 + judge
  python -m eval.pipeline.pipeline_full100.run_eval --limit 10      # 只跑前 10 题
  python -m eval.pipeline.pipeline_full100.run_eval --no-judge      # 只跑硬指标(不调裁判 LLM,最省)
  python -m eval.pipeline.pipeline_full100.run_eval --ids 1,27,60   # 指定题
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from eval import _bootstrap  # noqa: F401  路径/代理/环境引导,必须先于业务 import

import chat.service as service  # noqa: E402
from eval.pipeline.pipeline_full100 import judge as judge_mod  # noqa: E402
from eval.pipeline.pipeline_full100 import report as report_mod  # noqa: E402
from eval.pipeline.pipeline_full100 import metrics as M  # noqa: E402


def _load_questions(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    items = data if isinstance(data, list) else (data.get("items") or data.get("data"))
    return items or []


def _run_one(item: dict, *, use_judge: bool) -> dict:
    q = item["q"]
    tid = f"eval-{item['id']}-{int(time.time()*1000) % 100000}"
    t0 = time.time()

    answer = ""
    sources: list[dict] = []
    tier = None
    redos = 0
    escalations = 0
    clarified = False
    held_done = None
    path_errored = False

    for ev in service.react_stream(q, [], thread_id=tid, username="eval"):
        et = ev.get("type")
        if et == "assistant_message":
            answer = ev.get("content", "") or answer
        elif et == "sources":
            sources = ev.get("items") or sources      # 留最后一次(finalize 引用卡片)
        elif et == "tier":
            tier = ev.get("tier")
        elif et == "reflect":
            redos += 1
        elif et == "escalation":
            escalations += 1
        elif et == "clarify":
            clarified = True
        elif et == "error":
            path_errored = True
        elif et == "done":
            held_done = ev

    latency = round(time.time() - t0, 3)
    trace = (held_done or {}).get("trace") or {}
    retrieval_max_score = float((held_done or {}).get("retrieval_max_score") or 0.0)
    search_count = int((held_done or {}).get("search_count") or 0)
    final_reason = trace.get("final_reason") or ("error" if path_errored else "answer")

    fact_cov, fact_hits = M.fact_coverage(answer, item.get("facts") or [])
    record = {
        "id": item.get("id"),
        "q": q,
        "expect_tier": item.get("tier"),
        "expect_source": item.get("source") or "",
        "ground_truth": item.get("answer") or "",
        "tier": tier,
        "answer": answer,
        "sources": [{"source_stem": s.get("source_stem", ""),
                     "page": s.get("page", ""),
                     "score": s.get("score", 0.0),
                     "content": s.get("content", "")} for s in sources],
        "latency_s": latency,
        "redos": redos,
        "escalations": escalations,
        "clarified": clarified,
        "final_reason": final_reason,
        "retrieval_max_score": retrieval_max_score,
        "search_count": search_count,
        "fact_coverage": round(fact_cov, 4),
        "fact_hits": fact_hits,
        "source_hit": M.source_hit(sources, item.get("source") or ""),
    }

    if use_judge and answer.strip() and not clarified:
        scores = judge_mod.judge_all(
            q, answer, sources, ground_truth=record["ground_truth"])
        record.update(scores)
    return record


def main():
    ap = argparse.ArgumentParser(description="离线 RAGAS 式评测")
    ap.add_argument("--qa", default=_bootstrap.EVAL_QA_PATH, help="题库 JSON 路径")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 题(0=全部)")
    ap.add_argument("--ids", default="", help="逗号分隔的题目 id,如 1,27,60")
    ap.add_argument("--tier", default="", choices=["", "simple", "react"],
                    help="只跑该 tier 的题")
    ap.add_argument("--no-judge", action="store_true", help="跳过 LLM 裁判(只跑硬指标)")
    args = ap.parse_args()

    items = _load_questions(args.qa)
    if args.ids:
        want = {int(x) for x in args.ids.split(",") if x.strip()}
        items = [it for it in items if it.get("id") in want]
    if args.tier:
        items = [it for it in items if it.get("tier") == args.tier]
    if args.limit > 0:
        items = items[:args.limit]
    if not items:
        print("题库为空或筛选无结果。", file=sys.stderr)
        return 1

    print(f"开始评测:共 {len(items)} 题,judge={'关' if args.no_judge else '开'}")
    records = []
    for i, it in enumerate(items, 1):
        try:
            rec = _run_one(it, use_judge=not args.no_judge)
        except Exception as e:
            rec = {"id": it.get("id"), "q": it.get("q"), "answer": "",
                   "sources": [], "final_reason": "error", "fact_coverage": 0.0,
                   "fact_hits": [], "source_hit": False, "latency_s": 0.0,
                   "redos": 0, "escalations": 0, "clarified": False,
                   "tier": None, "retrieval_max_score": 0.0, "search_count": 0,
                   "expect_tier": it.get("tier"), "expect_source": it.get("source") or "",
                   "ground_truth": it.get("answer") or "", "run_error": str(e)[:200]}
            print(f"  [{i}/{len(items)}] id={it.get('id')} 运行异常: {str(e)[:120]}")
        records.append(rec)
        print(f"  [{i}/{len(items)}] id={rec.get('id')} tier={rec.get('tier')} "
              f"reason={rec.get('final_reason')} fact={rec.get('fact_coverage')} "
              f"maxscore={rec.get('retrieval_max_score'):.2f} "
              f"{rec.get('latency_s')}s redo={rec.get('redos')}")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    report_mod.write_reports(records, _bootstrap.RESULTS_DIR, stamp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
