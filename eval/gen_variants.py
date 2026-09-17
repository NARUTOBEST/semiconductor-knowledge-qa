# -*- coding: utf-8 -*-
"""从 web/public/eval-qa.json 生成 5 套互不相同的 100 题测试集。

user1 = 原题;user2..user5 = ARK 云端模型改写(保持语义/数字/型号/报警码不变,
facts/source 原样保留)。输出 eval/qa_sets/userN.json。
"""
import json
import os
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "web", "public", "eval-qa.json")
OUT_DIR = os.path.join(HERE, "qa_sets")

API_KEY = "REDACTED-API-KEY"
BASE = "https://ark.cn-beijing.volces.com/api/coding/v1"
MODEL = "deepseek-v4-flash"

STYLES = [
    "用更口语化的方式重新表述问题",
    "换一种问法,把疑问句改成请求说明的形式(如'请说明…''请介绍…')",
    "调整语序并适当补充'在设备维护场景下'等背景,但不得添加新事实",
    "用更简洁精炼的方式重新表述问题",
]


def chat(messages, retry=3):
    body = json.dumps({
        "model": MODEL, "messages": messages, "temperature": 0.3,
    }).encode()
    req = urllib.request.Request(
        f"{BASE}/chat/completions", data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {API_KEY}"})
    for i in range(retry):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                d = json.load(r)
            return d["choices"][0]["message"]["content"]
        except Exception as e:
            if i == retry - 1:
                raise
            time.sleep(3 * (i + 1))


def parse_items(text):
    """从模型输出中提取 JSON 数组。"""
    m = re.search(r"\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]", text, re.S)
    if not m:
        m = re.search(r"\[.*\]", text, re.S)
    arr = json.loads(m.group(0))
    return [str(x) for x in arr]


def rewrite_batch(items, style_idx):
    """一批 10 题改写;返回与输入等长的改写列表。"""
    qs = [it["q"] for it in items]
    prompt = (
        f"你是测试数据构造助手。对下列半导体设备问答测试题逐条改写:{STYLES[style_idx]}。"
        "硬性要求:1) 保持原意与提问焦点完全不变;2) 设备型号/报警码/数字/参数名"
        "必须原样保留,一字不改;3) 不得增加或臆造新事实;4) 每题只输出改写后的"
        "问句本身。\n"
        "输出:仅输出一个 JSON 字符串数组,长度必须等于输入题数,顺序一一对应。\n\n"
        + "\n".join(f"{i+1}. {q}" for i, q in enumerate(qs))
    )
    out = parse_items(chat([{"role": "user", "content": prompt}]))
    if len(out) != len(qs):
        raise ValueError(f"batch len mismatch: got {len(out)} want {len(qs)}")
    return out


def main():
    base = json.load(open(SRC, encoding="utf-8"))
    assert len(base) == 100, f"expect 100 questions, got {len(base)}"
    os.makedirs(OUT_DIR, exist_ok=True)

    # user1 = 原题
    json.dump(base, open(os.path.join(OUT_DIR, "user1.json"), "w",
                         encoding="utf-8"), ensure_ascii=False, indent=1)
    print("user1: 原题 (100)")

    # 4 套改写:按 10 题一批、每套 10 批,批次间并行
    for style_idx, uname in [(0, "user2"), (1, "user3"), (2, "user4"), (3, "user5")]:
        variants = [None] * 100
        batches = [(s, base[s:s + 10], i // 10)
                   for i, s in enumerate(range(0, 100, 10))]
        with ThreadPoolExecutor(max_workers=5) as ex:
            futs = {ex.submit(rewrite_batch, b, style_idx): s
                    for s, b, _ in batches}
            for fut in futs:
                start = futs[fut]
                for j, q in enumerate(fut.result()):
                    variants[start + j] = q
        out = []
        for it, q in zip(base, variants):
            nit = dict(it)
            nit["q"] = q
            nit["q_orig"] = it["q"]
            out.append(nit)
        path = os.path.join(OUT_DIR, f"{uname}.json")
        json.dump(out, open(path, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print(f"{uname}: 改写完成 (100) -> {path}")
    print("ALL SETS DONE")


if __name__ == "__main__":
    main()
