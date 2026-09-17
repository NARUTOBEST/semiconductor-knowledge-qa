# -*- coding: utf-8 -*-
"""生成参考答案(ground_truth)草稿供人工校对。

对高置信检索题,白盒跑一遍 react_stream 拿到检索来源,再让 judge LLM **严格基于
检索资料**写一份参考答案,输出到 eval/results/groundtruth-draft-<ts>.json。
人工校对后,把认可的 answer 抄回 web/public/eval-qa.json 的 "answer" 字段,
run_eval 的 context_recall 指标即会启用。

用法(项目根,旁路代理同 run_eval):
  python -m eval.make_groundtruth --limit 20
  python -m eval.make_groundtruth --ids 1,27
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from eval import _bootstrap  # noqa: F401

import chat.service as service  # noqa: E402
from eval import judge as judge_mod  # noqa: E402

# 只给检索足够可信的题生成草稿(max rerank 分 >= 此阈值),避免拿无关资料编参考答案。
_MIN_CONF = 0.5


def _draft_answer(question: str, sources: list[dict]) -> str:
    ctx = "\n\n".join(
        f"[资料{i+1}]({s.get('source_stem','')} {s.get('page','')}) {s.get('content','')}"
        for i, s in enumerate(sources))
    data = judge_mod._chat_json(
        "你是半导体设备资料编辑。只输出 JSON。",
        "请严格、仅依据下面的内部资料,为问题撰写一份准确、简洁的中文参考答案;"
        "资料不足处不要编造。\n"
        f"问题:{question}\n\n资料:\n{ctx}\n\n"
        '输出格式:{"answer":"参考答案正文"}')
    if isinstance(data, dict) and data.get("answer"):
        return str(data["answer"]).strip()
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", default=_bootstrap.EVAL_QA_PATH)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids", default="")
    ap.add_argument("--min-conf", type=float, default=_MIN_CONF)
    args = ap.parse_args()

    with open(args.qa, encoding="utf-8") as f:
        items = json.load(f)
    items = items if isinstance(items, list) else (items.get("items") or [])
    if args.ids:
        want = {int(x) for x in args.ids.split(",") if x.strip()}
        items = [it for it in items if it.get("id") in want]
    items = [it for it in items if it.get("tier") == "react" and not it.get("answer")]
    if args.limit > 0:
        items = items[:args.limit]

    drafts = []
    for i, it in enumerate(items, 1):
        q = it["q"]
        sources, max_score, done = [], 0.0, None
        for ev in service.react_stream(
                q, [], thread_id=f"gt-{it['id']}-{int(time.time()*1000)%100000}",
                username="eval"):
            if ev.get("type") == "sources":
                sources = ev.get("items") or sources
            elif ev.get("type") == "done":
                done = ev
        max_score = float((done or {}).get("retrieval_max_score") or 0.0)
        if max_score < args.min_conf or not sources:
            print(f"  [{i}/{len(items)}] id={it['id']} 跳过(置信 {max_score:.2f})")
            continue
        ans = _draft_answer(q, sources)
        drafts.append({"id": it["id"], "q": q, "answer": ans,
                       "max_score": max_score,
                       "sources": [f"{s.get('source_stem','')} {s.get('page','')}".strip()
                                   for s in sources[:6]]})
        print(f"  [{i}/{len(items)}] id={it['id']} 已生成草稿({len(ans)}字)")

    os.makedirs(_bootstrap.RESULTS_DIR, exist_ok=True)
    out = os.path.join(_bootstrap.RESULTS_DIR,
                       f"groundtruth-draft-{time.strftime('%Y%m%d-%H%M%S')}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(drafts, f, ensure_ascii=False, indent=2)
    print(f"\n生成 {len(drafts)} 条草稿 -> {out}\n请人工校对后把 answer 抄回 eval-qa.json。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
