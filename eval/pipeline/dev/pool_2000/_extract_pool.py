# -*- coding: utf-8 -*-
"""从会话 transcript 批量提取所有已完成备选池出题结果(P01..P80)并落盘。

与 dev/_extract_result.py 同源逻辑,区别:
- 一次扫全量,按 query_id 前缀(P\\d\\d)分组,每组取最新通知;
- 允许超产(len>25):按精确分布 {easy:8,medium:12,hard:5} 裁剪子集(优先丢弃靠后 id);
- len<25 或分布凑不齐 → 记 FAIL,待重跑。

用法: python _extract_pool.py [P01 P02 ...]   (无参数=全部已完成的)
"""
import json, os, re, sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "draft_material")
SESSION = r"C:\Users\33423\.claude\projects\C--project3\d0356e8b-1380-4d0a-99ac-1cb9a77669d3.jsonl"
REQ = ("query", "task_type", "difficulty", "source_id", "required_tools",
       "expected_rounds", "gold_standard", "success_criteria", "evidence")
DIST_D = {"easy": 8, "medium": 12, "hard": 5}


def walk(x, out):
    if isinstance(x, str):
        if "<task-notification>" in x:
            out.append(x)
    elif isinstance(x, dict):
        for v in x.values():
            walk(v, out)
    elif isinstance(x, list):
        for v in x:
            walk(v, out)


def parse_arr(body):
    body = re.sub(r'"\s*\.replace\([^)]*\)', '"', body)   # 误写的 Python .replace 后缀
    j = body.find("[")
    k = body.rfind("]")          # 剥掉 ```json 围栏等首尾杂质
    if j < 0 or (0 <= body.find("{") < j):
        j = body.find("{") - 1
    seg = body[j:k + 1].strip() if k > j else body[j:].strip()
    if not seg.endswith("]"):
        seg += "]"
    variants = [
        seg,
        re.sub(r'"\)(\s*[,}\]])', r'"\1', seg),           # 引号后游离的 ')'
        re.sub(r'"\)(\s*,)', r'"}\1', seg),               # ')' 实为 '}' 笔误(缺对象闭合)
    ]
    for v in variants:
        for s in (v, _repair_quotes(v)):
            try:
                arr = json.loads(s, strict=False)
                if isinstance(arr, list):
                    return arr
            except (json.JSONDecodeError, ValueError):
                continue
    raise ValueError("所有解析方案均失败")


def trim_dist(arr):
    """len>25 时按精确难度分布裁剪;返回 (子集或 None, 原因)。"""
    cnt = Counter(q.get("difficulty") for q in arr)
    if all(cnt[k] >= v for k, v in DIST_D.items()):
        keep = {k: v for k, v in DIST_D.items()}
        sel = []
        for q in reversed(arr):          # 从后往前丢超产的
            d = q.get("difficulty")
            if keep.get(d, 0) > 0:
                keep[d] -= 1
                sel.append(q)
        return list(reversed(sel)), None
    return None, f"分布凑不齐 {dict(cnt)}"


def _repair_quotes(s):
    out, i, n, instr = [], 0, len(s), False
    while i < n:
        c = s[i]
        if not instr:
            instr = c == '"'
            out.append(c); i += 1
        elif c == "\\":
            nxt = s[i + 1] if i + 1 < n else ""
            if nxt in '"\\/\bfnrtu' or nxt == "u":
                out.append(s[i:i + 2]); i += 2
            else:
                out.append("\\\\" + nxt); i += 2
        elif c == '"':
            k = i + 1
            while k < n and s[k] in " \t\r\n":
                k += 1
            if k < n and s[k] in ",:}]":
                instr = False
                out.append(c)
            else:
                out.append('\\"')
            i += 1
        else:
            out.append(c); i += 1
    return "".join(out)


def main():
    only = set(a.upper() for a in sys.argv[1:])
    data = open(SESSION, encoding="utf-8").read()
    best = {}                              # tag -> (arr, err)
    for line in data.splitlines():
        if "<task-notification>" not in line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        notes = []
        walk(obj, notes)
        for s in notes:
            m = re.search(r"<result>(.*)</result>", s, re.S)
            if not m:
                continue
            body = m.group(1)
            mid = re.search(r'"query_id"\s*:\s*"(P\d{2})-', body)
            if not mid:
                continue
            tag = mid.group(1)
            if only and tag not in only:
                continue
            try:
                arr = parse_arr(body)
            except Exception as e:
                best[tag] = (None, f"JSON解析失败: {str(e)[:80]}")
                continue
            if not isinstance(arr, list):
                best[tag] = (None, "结果不是数组")
                continue
            for q in arr:                # P08-13 源名误写修正
                if 'Richardson' in str(q.get("source_id", "")):
                    q["source_id"] = q["source_id"].replace(
                        "Richardson", "Richard Lawson ")
            bad = [q.get("query_id") for q in arr
                   if not all(q.get(k) for k in REQ)]
            if bad:
                best[tag] = (None, f"缺字段: {bad[:3]}")
                continue
            if len(arr) == 25:
                best[tag] = (arr, None)
            elif len(arr) > 25:
                sel, err = trim_dist(arr)
                best[tag] = (sel, err)
            else:
                best[tag] = (None, f"仅 {len(arr)} 题 <25")
    os.makedirs(OUT, exist_ok=True)
    ok, fail = [], []
    for tag in sorted(best):
        if only and tag not in only:
            continue
        arr, err = best[tag]
        if arr is None:
            fail.append(f"{tag}: {err}")
            continue
        ids = [q["query_id"] for q in arr]
        assert all(i.startswith(tag) for i in ids), f"{tag} id 前缀异常"
        d = Counter(q["difficulty"] for q in arr)
        t = Counter(q["task_type"] for q in arr)
        if dict(d) != DIST_D or dict(t) != {"信息查询": 15, "数据计算": 5,
                                            "工具编排": 3, "异常处理": 2}:
            fail.append(f"{tag}: 裁剪后分布异常 d={dict(d)} t={dict(t)}")
            continue
        fp = os.path.join(OUT, f"draft_{tag}.json")
        json.dump(arr, open(fp, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        ok.append(f"{tag}: 25题 {dict(t)}")
    print(f"OK {len(ok)}: {' '.join(t.split(':')[0] for t in ok)}")
    if fail:
        print(f"FAIL {len(fail)}:")
        for f in fail:
            print("  " + f)


if __name__ == "__main__":
    main()
