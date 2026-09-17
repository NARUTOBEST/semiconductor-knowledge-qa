# -*- coding: utf-8 -*-
"""并发评测结果的 LLM-as-Judge 驱动(配合 run_concurrent.py 的 JSONL)。

用法(项目根,需可访问 LLM 网关 :4000 与检索服务 :8002):
    python -m eval.judge_concurrent --results-dir results_concurrent

对每条有答案的记录跑 judge.judge_all(faithfulness / context_precision /
answer_relevancy),faithfulness < 0.75 记 has_hallucination=True。
逐条追加写回 JSONL 字段(judge_*),汇总写入 judge_summary.json。
断点续跑:已有 judge 字段的记录跳过。
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # 项目根(eval 作为包)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "config"))

from eval import _bootstrap  # noqa: E402,F401
from eval import judge as judge_mod  # noqa: E402
import config as C  # noqa: E402

HALLUC_THRESHOLD = 0.75
JUDGE_KEYS = ("faithfulness", "context_precision", "answer_relevancy")


def _has_judge(rec: dict) -> bool:
    return all(rec.get(f"judge_{k}") is not None for k in JUDGE_KEYS) or rec.get("judge_done")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=os.path.join(HERE, "results_concurrent"))
    ap.add_argument("--sets-dir", default=os.path.join(HERE, "qa_sets"))
    ap.add_argument("--limit", type=int, default=0, help="每用户最多判 N 条(0=全部)")
    args = ap.parse_args()

    summary = {}
    for u in sorted(os.listdir(args.results_dir)):
        if not (u.startswith("eval") and u.endswith(".jsonl")):
            continue
        path = os.path.join(args.results_dir, u)
        with open(path, encoding="utf-8") as f:
            recs = [json.loads(l) for l in f if l.strip()]
        n_done = 0
        # 全文缓存:eval 记录里 sources.content 是 160 字符流式预览,
        # 裁判必须用块全文(经检索服务 /get_chunk 按 chunk_id 取回),否则
        # 忠实度/上下文精确率会被预览截断系统性压低。
        full_cache: dict[str, str] = {}

        def _full_content(chunk_id: str) -> str:
            if chunk_id in full_cache:
                return full_cache[chunk_id]
            text = ""
            try:
                import httpx
                base = getattr(C, "RETRIEVAL_SERVICE_URL",
                               "http://127.0.0.1:8002").rstrip("/")
                tok = getattr(C, "RETRIEVAL_INTERNAL_TOKEN", "")
                headers = {"X-Internal-Token": tok} if tok else None
                r = httpx.post(base + "/get_chunk", json={"chunk_id": chunk_id},
                               headers=headers, timeout=30)
                r.raise_for_status()
                text = str((r.json() or {}).get("content") or "")
            except Exception:
                # 检索服务不可达(如 GPU 下线)时回退 VM 本地 qdrant 直取:
                # point id 与入库侧一致为 uuid5(NAMESPACE_DNS, chunk_id)
                try:
                    import httpx, uuid as _uuid
                    pid = str(_uuid.uuid5(_uuid.NAMESPACE_DNS, str(chunk_id)))
                    q = getattr(C, "QDRANT_URL", "http://127.0.0.1:6333").rstrip("/")
                    r2 = httpx.post(
                        f"{q}/collections/ald_text/points",
                        json={"ids": [pid], "with_payload": True},
                        timeout=30)
                    r2.raise_for_status()
                    pts = (r2.json() or {}).get("result") or []
                    if pts:
                        text = str(((pts[0] or {}).get("payload") or {})
                                   .get("content") or "")
                except Exception:
                    text = ""
            full_cache[chunk_id] = text
            return text

        for rec in recs:
            if args.limit and n_done >= args.limit:
                break
            answer = (rec.get("answer") or "").strip()
            sources = rec.get("sources") or []
            if not answer or not sources or _has_judge(rec):
                continue
            # 用全文替换预览(拿不到全文时保留预览,裁判仍可部分判定)
            for s in sources:
                if not isinstance(s, dict):
                    continue
                cid = s.get("chunk_id")
                if cid:
                    full = _full_content(str(cid))
                    if full:
                        s["content"] = full
            gt = ""
            # ground_truth:题库有 facts 时用其拼参考要点(context_recall 用)
            facts = rec.get("facts") or []
            if isinstance(facts, list) and facts:
                gt = "；".join(str(x) for x in facts)
            j = judge_mod.judge_all(rec.get("q", ""), answer, sources, ground_truth=gt)
            for k, v in j.items():
                rec[f"judge_{k}"] = v
            f_val = j.get("faithfulness")
            rec["has_hallucination"] = bool(
                f_val is not None and f_val < HALLUC_THRESHOLD)
            rec["judge_done"] = True
            n_done += 1
        # 重写整个文件(记录不多,100 条/用户)
        with open(path, "w", encoding="utf-8") as f:
            for rec in recs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        vals = {k: [r[f"judge_{k}"] for r in recs if r.get(f"judge_{k}") is not None]
                for k in JUDGE_KEYS}
        judged = [r for r in recs if r.get("judge_done")]
        summary[u] = {
            "total": len(recs),
            "judged": len(judged),
            "avg_faithfulness": round(sum(vals["faithfulness"]) / len(vals["faithfulness"]), 4) if vals["faithfulness"] else None,
            "avg_context_precision": round(sum(vals["context_precision"]) / len(vals["context_precision"]), 4) if vals["context_precision"] else None,
            "avg_answer_relevancy": round(sum(vals["answer_relevancy"]) / len(vals["answer_relevancy"]), 4) if vals["answer_relevancy"] else None,
            "hallucination_count": sum(1 for r in judged if r.get("has_hallucination")),
            "hallucination_rate": round(sum(1 for r in judged if r.get("has_hallucination")) / len(judged), 4) if judged else None,
        }
        print(f"[{u}] judged={len(judged)} summary={summary[u]}", flush=True)

    out = os.path.join(args.results_dir, "judge_summary.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"== judge summary -> {out}")


if __name__ == "__main__":
    main()
