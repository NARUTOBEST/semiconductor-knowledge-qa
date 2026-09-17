# -*- coding: utf-8 -*-
"""从 user1-5.json 各抽 10 题,用云 LLM 改写成"中等难度"问法:
事实/答案完全不变,但问法间接化(不直接点名关键术语,用场景描述/同义转述),
使问题相较于原题不容易理解。输出 eval/qa_sets_medium/user{i}.json。
在 VM 上运行:python3 gen_medium_sets.py"""
import json
import os
import re
import time
import urllib.request

BASE = os.environ.get("OPENAI_BASE_URL",
                      "https://ark.cn-beijing.volces.com/api/coding/v1")
KEY = os.environ["OPENAI_API_KEY"]
MODEL = "doubao-seed-2.0-lite-260428"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "qa_sets_medium")

PROMPT = """你是测试题改写器。把下面的设备知识问题改写成"中等难度"版本:
1. 问题的答案/事实必须完全不变(同一台设备、同一个参数、同一个结论),只是问法变绕;
2. 【不要直接使用】原题中的关键术语字面(型号代码、报警号、部件名可保留其中之一作锚点,
   但参数名/指标名/动作要换成场景化描述、同义词或间接指代,让人不能一眼看出在问什么);
3. 可以包装成现场情景(如"现场反映某现象,想知道标准值是多少");
4. 保持中文,长度与原题相近或略长,仍然是一个可以从内部手册资料回答的问题;
5. 只输出改写后的问题本身,不要解释。

原题:{q}"""


def chat(q, retry=3):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT.format(q=q)}],
        "temperature": 0.4, "max_tokens": 300,
        "thinking": {"type": "disabled"},
    }).encode()
    for i in range(retry):
        try:
            req = urllib.request.Request(
                BASE + "/chat/completions", data=body,
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {KEY}"}, method="POST")
            with urllib.request.urlopen(req, timeout=30) as r:
                d = json.load(r)
            txt = (d["choices"][0]["message"].get("content") or "").strip()
            txt = re.sub(r"^(改写[后版]?\s*[:：]\s*)", "", txt).strip().strip('"“”')
            if txt and len(txt) >= 8:
                return txt
        except Exception as e:
            print("  retry", i, type(e).__name__, str(e)[:80], flush=True)
            time.sleep(2 * (i + 1))
    return None


def main():
    os.makedirs(OUT, exist_ok=True)
    for u in range(1, 6):
        src = json.load(open(os.path.join(HERE, "qa_sets", f"user{u}.json"),
                             encoding="utf-8"))
        # 每 10 题取 1 题,分散覆盖题库
        picked = src[::10][:10]
        out = []
        for it in picked:
            newq = chat(it["q"])
            if newq is None:
                newq = it["q"]          # 改写失败保底用原题
                print(f"  user{u} id{it['id']}: KEEP-ORIGINAL", flush=True)
            out.append({**it, "q": newq, "orig_q": it["q"],
                        "difficulty": "medium"})
            print(f"user{u} id{it['id']}: {newq}", flush=True)
            time.sleep(0.5)
        with open(os.path.join(OUT, f"user{u}.json"), "w",
                  encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print(f"== user{u} done ({len(out)} questions)", flush=True)


if __name__ == "__main__":
    main()
