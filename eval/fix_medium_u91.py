# -*- coding: utf-8 -*-
"""把 qa_sets_medium 里无 facts 的题替换为原题库中有 facts 的未用题(从尾部找)的改写版。"""
import json, os, re, time, urllib.request

BASE = os.environ.get("OPENAI_BASE_URL", "https://ark.cn-beijing.volces.com/api/coding/v1")
KEY = os.environ["OPENAI_API_KEY"]
MODEL = "doubao-seed-2.0-lite-260428"
PROMPT = """你是测试题改写器。把下面的设备知识问题改写成"中等难度"版本:
1. 答案/事实完全不变,只是问法变绕;
2. 不要直接使用原题的关键参数名/指标名,换成场景化描述或间接指代;
3. 保持中文,长度与原题相近或略长;
4. 只输出改写后的问题本身,不要解释、不要思考过程。

原题:{q}"""

def chat(q, retry=3):
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": PROMPT.format(q=q)}],
                       "temperature": 0.3, "max_tokens": 250, "thinking": {"type": "disabled"}}).encode()
    for i in range(retry):
        try:
            req = urllib.request.Request(BASE + "/chat/completions", data=body,
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"}, method="POST")
            with urllib.request.urlopen(req, timeout=30) as r:
                txt = (json.load(r)["choices"][0]["message"].get("content") or "").strip()
            txt = re.sub(r"^(改写[后版]?\s*[:：]\s*)", "", txt).strip().strip('"“”')
            if 8 <= len(txt) <= 150 and "原题" not in txt and "无法回答" not in txt:
                return txt
        except Exception as e:
            print("  retry", i, type(e).__name__, str(e)[:60], flush=True)
            time.sleep(2 * (i + 1))
    return None

HERE = os.path.dirname(os.path.abspath(__file__))
for u in range(1, 6):
    p = os.path.join(HERE, "qa_sets_medium", f"user{u}.json")
    qa = json.load(open(p, encoding="utf-8"))
    src = json.load(open(os.path.join(HERE, "qa_sets", f"user{u}.json"), encoding="utf-8"))
    used = {x["id"] for x in qa}
    # 从尾部找有 facts 且未用的题
    cand = None
    for it in reversed(src):
        if it.get("facts") and it["id"] not in used:
            cand = it
            break
    assert cand, u
    newq = chat(cand["q"])
    if not newq:
        newq = cand["q"]
        print(f"user{u}: KEEP-ORIGINAL")
    for i, x in enumerate(qa):
        if not x.get("facts"):
            qa[i] = {**cand, "q": newq, "orig_q": cand["q"], "difficulty": "medium"}
            break
    json.dump(qa, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"user{u} id{cand['id']}: {newq}", flush=True)
    time.sleep(0.5)
print("FIX DONE")
