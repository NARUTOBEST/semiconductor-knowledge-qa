# -*- coding: utf-8 -*-
"""纯 LLM 翻译路离线验证:盲区题逐题 翻译→检索→金文档/金块进池检查。

用法: python test_translate25.py
"""
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
for _line in open(os.path.join(ROOT, "env", "env.env"), encoding="utf-8"):
    _line = _line.strip()
    if _line and not _line.startswith("#") and "=" in _line:
        _k, _, _v = _line.partition("=")
        os.environ.setdefault(_k.strip(), _v.strip())

ARK_KEY = os.getenv("OPENAI_API_KEY", "")
ARK_BASE = "https://ark.cn-beijing.volces.com/api/coding/v1"
MODEL = "doubao-seed-2-0-lite-260428"
PROMPT = ("把用户的中文问题翻译成一句用于检索英文半导体设备手册的英文查询。"
          "保留型号/数字/单位原样,输出英文查询本身,不要解释。\n\n用户问题:")

import collect_ranked
from collect_ranked import search  # noqa: E402

collect_ranked.TOKEN = collect_ranked._token()


def chat(user):
    import urllib.request
    body = json.dumps({"model": MODEL,
                       "messages": [{"role": "user", "content": user}],
                       "temperature": 0, "max_tokens": 120}).encode()
    req = urllib.request.Request(
        ARK_BASE + "/chat/completions", data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + ARK_KEY})
    with urllib.request.urlopen(req, timeout=60) as r:
        d = json.load(r)
    return (d["choices"][0]["message"]["content"] or "").strip()


def main():
    gold = json.load(open(os.path.join(HERE, "runs", "gold_cache.json"),
                          encoding="utf-8"))
    r8 = {}
    for f in glob.glob(os.path.join(HERE, "runs", "dev200_round8_final_judge",
                                    "*.jsonl")):
        for l in open(f, encoding="utf-8"):
            r = json.loads(l)
            if r.get("judge_done"):
                r8[r["id"]] = r
    print("judged recs:", len(r8))
    IDS = ["dev_185", "dev_195", "dev_198", "dev_030", "dev_041", "dev_067",
           "dev_075", "dev_076", "dev_080", "dev_105", "dev_111", "dev_130",
           "dev_168", "dev_169", "dev_028", "dev_043", "dev_088", "dev_101",
           "dev_124", "dev_163", "dev_036", "dev_084", "dev_171"]
    doc_in = chunk_in10 = chunk_inpool = 0
    n = 0
    for i in IDS:
        v = r8.get(i)
        if v is None:
            print(i, "SKIP(不在判分集)")
            continue
        stem = v.get("source") or ""
        g = set(gold.get(stem, {}).get("gold") or [])
        try:
            en = chat(PROMPT + v["q"])
        except Exception as e:  # noqa: BLE001
            print(i, "TRANS_ERR", repr(e)[:120])
            continue
        d = search(en)
        ranked = d.get("ranked") or []
        ids = [c.get("chunk_id") for c in ranked]
        pool_stems = {c.get("source_stem") for c in ranked}
        pos = next((x + 1 for x, c in enumerate(ids) if c in g), None)
        n += 1
        doc_in += stem in pool_stems
        chunk_inpool += pos is not None
        chunk_in10 += pos is not None and pos <= 10
        print(f"{i} pos={pos} | {en[:90]}")
    print(f"\n{n} 题: 金文档进池 {doc_in},金块进池 {chunk_inpool},"
          f"金块前10 {chunk_in10}")


if __name__ == "__main__":
    main()
