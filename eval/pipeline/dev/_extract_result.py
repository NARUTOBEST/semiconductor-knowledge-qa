# -*- coding: utf-8 -*-
"""从会话 transcript 的 task-notification 提取某批出题结果并落盘。

用法: python _extract_result.py <query_id前缀如CL8-> <draft_tag如08_清洗技术_VIII>
在 transcript 中找含该前缀的最新 <task-notification>,从 <result> 里恢复 JSON 数组,
校验字段完整与 25 题分布后写 draft_material/draft_<tag>.json。
"""
import json, os, re, sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
SESSION = r"C:\Users\33423\.claude\projects\C--project3\d0356e8b-1380-4d0a-99ac-1cb9a77669d3.jsonl"
REQ = ("query", "task_type", "difficulty", "source_id", "required_tools",
       "expected_rounds", "gold_standard", "success_criteria", "evidence")


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


def main():
    prefix, tag = sys.argv[1], sys.argv[2]
    data = open(SESSION, encoding="utf-8").read()
    cands = []
    for line in data.splitlines():
        if prefix not in line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        res = []
        walk(obj, res)
        cands.extend(res)
    if not cands:
        sys.exit(f"未找到含 {prefix} 的通知")
    body = None
    for s in reversed(cands):  # 取最新
        m = re.search(r"<result>(.*)</result>", s, re.S)
        if m and prefix in m.group(1):
            body = m.group(1)
            break
    if body is None:
        sys.exit("通知里没有 <result> 含 " + prefix)
    j = body.find("[")
    if j < 0 or body.find("{") < j:
        j = body.find("{") - 1  # 开头 '[' 缺损时补
    seg = body[j:].strip()
    if not seg.endswith("]"):
        seg += "]"  # 尾部 ']' 截断时补
    try:
        arr = json.loads(seg)
    except json.JSONDecodeError:
        arr = json.loads(_repair_quotes(seg))
    assert isinstance(arr, list) and len(arr) == 25, f"题数 {len(arr)} != 25"
    bad = [q.get("query_id") for q in arr if not all(q.get(k) for k in REQ)]
    assert not bad, f"缺字段: {bad}"
    ids = [q["query_id"] for q in arr]
    assert all(i.startswith(prefix.rstrip("-")) for i in ids), ids
    d, t = Counter(q["difficulty"] for q in arr), Counter(q["task_type"] for q in arr)
    assert dict(d) == {"easy": 8, "medium": 12, "hard": 5}, dict(d)
    assert dict(t) == {"信息查询": 15, "数据计算": 5, "工具编排": 3, "异常处理": 2}, dict(t)
    out = os.path.join(HERE, "draft_material", f"draft_{tag}.json")
    json.dump(arr, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("saved", out, dict(d), dict(t))


def _repair_quotes(s):
    """字符串内未转义 '"' 的容错修复:仅当其后(跳过空白)是 , : } ] 才视为结束符。"""
    out, i, n, instr = [], 0, len(s), False
    while i < n:
        c = s[i]
        if not instr:
            instr = c == '"'
            out.append(c); i += 1
        elif c == "\\":
            nxt = s[i+1] if i + 1 < n else ""
            if nxt in '"\\/\bfnrtu' or nxt == "u":
                out.append(s[i:i+2]); i += 2
            else:
                out.append("\\\\" + nxt); i += 2  # 非法转义(如 \_) → 字面反斜杠
        elif c == '"':
            k = i + 1
            while k < n and s[k] in " \t\r\n":
                k += 1
            if k < n and s[k] in ",:}]":
                instr = False
                out.append(c)
            else:
                out.append('\\"')  # 字符串内裸引号 → 转义
            i += 1
        else:
            out.append(c); i += 1
    return "".join(out)


if __name__ == "__main__":
    main()
