# -*- coding: utf-8 -*-
"""清除 run21b 结果里评测期写入的 judge_* 字段(它们全是 None/预览口径),
让 judge_concurrent 真正重判。"""
import glob
import json

d = "/home/lly/eval/results_concurrent/run21b_postfix"
for p in sorted(glob.glob(d + "/eval*.jsonl")):
    out = []
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        for k in list(r):
            if k.startswith("judge_") or k == "has_hallucination":
                r.pop(k, None)
        out.append(json.dumps(r, ensure_ascii=False))
    open(p, "w", encoding="utf-8").write("\n".join(out) + "\n")
    print("purged", p)
