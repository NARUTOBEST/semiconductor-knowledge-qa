# -*- coding: utf-8 -*-
"""异常处理子集(16题拒答型为主)的 E2E 答案判分:good/partial/bad。

判分口径 = 该题 success_criteria(拒答型题:明确说手册未涉及+不编造+给出替代建议)。
用法: python eval/pipeline/dev/grade_subset16.py --results-dir <run目录> [--label A]
需要环境变量 OPENAI_API_KEY(ARK)。
"""
import argparse
import glob
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", "..", ".."))

ARK_KEY = os.getenv("OPENAI_API_KEY", "")
ARK_BASE = "https://ark.cn-beijing.volces.com/api/v3"
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "ep-20260917012501-44sxb")

PROMPT = """你是评测裁判。给定的"成功标准"描述了正确答案应满足的要求(很多题是"手册未涉及X,
应明确拒答该未知项并给出替代建议,不得编造数值")。

请对照候选答案,输出三选一:
- good: 完全符合成功标准(该拒答的明确拒答且未编造;该给值的给对了)
- partial: 方向对但有缺漏/表述含糊,或拒答了但夹带未经证实的内容
- bad: 编造了数值/章节,或答非所问,或错误断言手册里有/没有某内容

只输出一个 JSON 对象: {{"verdict": "good|partial|bad", "reason": "一句话中文理由"}}


=== 成功标准 ===
{criteria}

=== 候选答案 ===
{answer}"""


def judge_one(qid, criteria, answer):
    body = json.dumps({
        "model": JUDGE_MODEL,
        "messages": [{"role": "user",
                      "content": PROMPT.format(criteria=criteria,
                                               answer=answer[:3000])}],
        "temperature": 0.1,
    }, ensure_ascii=False).encode()
    for attempt in range(4):
        try:
            r = requests.post(ARK_BASE + "/chat/completions", data=body,
                              headers={"Content-Type": "application/json",
                                       "Authorization": f"Bearer {ARK_KEY}"},
                              timeout=120)
            r.raise_for_status()
            txt = r.json()["choices"][0]["message"]["content"]
            m = re.search(r'\{[^}]+\}', txt, re.S)
            d = json.loads(m.group(0))
            return qid, d.get("verdict", "?"), d.get("reason", "")[:120]
        except Exception as e:
            if attempt == 3:
                return qid, "err", str(e)[:120]
            time.sleep(5 * (attempt + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    meta = {q["query_id"]: q for q in json.load(
        open(os.path.join(ROOT, "eval/qa_sets/v2_20261001/dev_200.json"),
             encoding="utf-8"))}
    rows = {}
    for f in glob.glob(os.path.join(args.results_dir, "eval*.jsonl")):
        for line in open(f, encoding="utf-8"):
            d = json.loads(line)
            rows[d["id"]] = d

    todo = [(qid, meta[qid]["success_criteria"], str(d.get("answer") or ""))
            for qid, d in sorted(rows.items())]
    with ThreadPoolExecutor(max_workers=4) as ex:
        res = list(ex.map(lambda t: judge_one(*t), todo))

    score = {"good": 0, "partial": 0, "bad": 0, "err": 0}
    print(f"== 判分 {args.label or args.results_dir} ({len(res)} 题)")
    for qid, v, reason in res:
        score[v] = score.get(v, 0) + 1
        print(f"{qid} [{v}] {reason}")
    n = len(res)
    print(f"== 汇总: good={score['good']} partial={score['partial']} "
          f"bad={score['bad']} err={score['err']} | "
          f"good率={score['good']/max(1,n):.2%} "
          f"good+partial率={(score['good']+score['partial'])/max(1,n):.2%}")
    out = os.path.join(args.results_dir, "subset16_grade.json")
    json.dump({"score": score, "detail": res}, open(out, "w",
               encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
