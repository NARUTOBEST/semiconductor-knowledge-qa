# -*- coding: utf-8 -*-
"""合并 dev200_round8(149 有效)+ 补测结果(dev200_round8_fix),输出可判分目录。

按 id 用补测记录替换原废样本(检索故障、sources 为空的 react 记录),
其余原样保留。输出: runs/dev200_round8_final/<user>.jsonl

用法: python merge_round8.py
"""
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "runs", "dev200_round8")
FIX = os.path.join(HERE, "runs", "dev200_round8_fix")
OUT = os.path.join(HERE, "runs", "dev200_round8_final")


def main():
    os.makedirs(OUT, exist_ok=True)
    fixed = {}
    for f in glob.glob(os.path.join(FIX, "*.jsonl")):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            if (r.get("sources") or []) and not r.get("error"):
                fixed[r["id"]] = r
    print(f"fixed usable: {len(fixed)}")

    n_replaced = n_kept = n_still_bad = 0
    for f in glob.glob(os.path.join(SRC, "*.jsonl")):
        out_path = os.path.join(OUT, os.path.basename(f))
        with open(out_path, "w", encoding="utf-8") as w:
            for line in open(f, encoding="utf-8"):
                r = json.loads(line)
                if r["id"] in fixed:
                    w.write(json.dumps(fixed[r["id"]], ensure_ascii=False) + "\n")
                    n_replaced += 1
                else:
                    if r.get("tier") == "react" and not (r.get("sources") or []):
                        n_still_bad += 1
                    w.write(line if line.endswith("\n") else line + "\n")
                    n_kept += 1
    print(f"replaced={n_replaced} kept={n_kept} still_broken={n_still_bad}")


if __name__ == "__main__":
    main()
