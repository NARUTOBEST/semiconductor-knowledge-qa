# -*- coding: utf-8 -*-
"""汇总 draft_material/draft_P01..P80.json(80×25 题)→ eval/qa_sets/supplement_pool_2000.json

校验:每份 25 题、9 字段齐全、id 前缀、精确难度/类型分布;全池 query_id 唯一、
query 去重(完全相同文本剔除)。输出统计摘要。
"""
import json, os, glob
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
DRAFT = os.path.join(HERE, "draft_material")
OUT = os.path.join(HERE, "..", "..", "..", "..", "qa_sets", "supplement_pool_2000.json")
REQ = ("query", "task_type", "difficulty", "source_id", "required_tools",
       "expected_rounds", "gold_standard", "success_criteria", "evidence")
DIST_D = {"easy": 8, "medium": 12, "hard": 5}
DIST_T = {"信息查询": 15, "数据计算": 5, "工具编排": 3, "异常处理": 2}


def main():
    pool, problems = [], []
    for f in sorted(glob.glob(os.path.join(DRAFT, "draft_P*.json"))):
        tag = os.path.basename(f)[6:9]
        try:
            arr = json.load(open(f, encoding="utf-8"))
            assert len(arr) == 25, f"{len(arr)} 题"
            for q in arr:
                assert all(q.get(k) for k in REQ), f"缺字段 {q.get('query_id')}"
                assert q["query_id"].startswith(tag), q["query_id"]
            assert dict(Counter(q["difficulty"] for q in arr)) == DIST_D
            assert dict(Counter(q["task_type"] for q in arr)) == DIST_T
            pool.extend(arr)
        except Exception as e:
            problems.append(f"{tag}: {e}")
    # 全池唯一性
    ids = [q["query_id"] for q in pool]
    dup_ids = [i for i, c in Counter(ids).items() if c > 1]
    seen, uniq = set(), []
    for q in pool:
        key = q["query"].strip()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(q)
    print(f"文档数: {80 - len(problems)}/80  总题数: {len(pool)}")
    print(f"重复 query_id: {dup_ids[:5] or '无'}  重复 query 剔除: {len(pool) - len(uniq)}")
    if problems:
        print("问题文档:")
        for p in problems:
            print("  " + p)
        raise SystemExit(1)
    assert len(uniq) == 2000, f"去重后 {len(uniq)} != 2000"
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(uniq, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("难度:", dict(Counter(q["difficulty"] for q in uniq)))
    print("类型:", dict(Counter(q["task_type"] for q in uniq)))
    print("轮数:", dict(Counter(q["expected_rounds"] for q in uniq)))
    print(f"== saved -> {os.path.normpath(OUT)}")


if __name__ == "__main__":
    main()
