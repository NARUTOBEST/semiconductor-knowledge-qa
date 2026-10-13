# -*- coding: utf-8 -*-
"""轮次19 术语表构建:tmp_terms_raw.json( extract_terms.py 产物)经 LLM 批译,
产出 mcp_servers/retrieval/glossary.json。
- 走 coding plan 接入点(ep 按量欠费,2026-10-05 起)
- 每术语:drop 判定 + 中文叫法 1~3 个 + 英文别名(变体/缩写,用于 en 查询归一)
- 断点续跑(CKPT),与 build_glossary.py 同款流程
"""
import json
import os
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ENV_FILE = os.path.join(ROOT, "env", "env.env")
if os.path.exists(ENV_FILE):
    for _line in open(ENV_FILE, encoding="utf-8"):
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())
# ret.env 是检索服务的 key 来源(与生产翻译路同 key),env.env 没有它
RET_ENV = os.path.join(HERE, "ret.env")
if os.path.exists(RET_ENV):
    for _line in open(RET_ENV, encoding="utf-8"):
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

BASE = os.getenv("TRANSLATE_BASE", "https://ark.cn-beijing.volces.com/api/coding/v1")
KEY = os.getenv("TRANSLATE_KEY", "")
MODEL = os.getenv("TRANSLATE_MODEL", "doubao-seed-2-0-lite-260428")

SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "tmp_terms_raw.json")
OUT = os.path.join(ROOT, "mcp_servers", "retrieval", "glossary.json")
CKPT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "tmp_glossary2_ckpt.json")
B = 50

PROMPT = (
    "你是半导体设备领域的术语专家。下面给你一批英文术语(来自半导体设备技术手册语料)。\n"
    "对每个术语:\n"
    "1. 判断它是否为\"技术名词\"(设备/部件/工艺/材料/参数/指标/品牌型号)。纯泛用词"
    "(如 function, addition, NOTE, Message Header)标记 drop=true;\n"
    "2. 技术名词给出中文叫法 1~3 个:第一个是最常用的规范译名,后面可补工程口语/常见别名"
    "(如 megasonic cleaning -> [\"兆声清洗\",\"兆声波清洗\"]);品牌/型号(如 6DS-SP)保留"
    "原文不译,zh 给空数组;\n"
    "3. 给出英文别名 0~3 个(手册里的常见变体/缩写/全称,en2 字段,不含术语本身;"
    "如 buried oxide -> [\"BOX\",\"buried oxide layer\"],没有就给空数组);\n"
    "4. 严禁臆造不存在的叫法。\n"
    "只输出 JSON:{\"items\":[{\"en\":\"...\",\"drop\":false,\"zh\":[\"...\"],"
    "\"en2\":[\"...\"]}]},items 顺序与输入一致。\n\n术语列表:\n{terms}"
)


def batch(terms):
    tl = "\n".join(f"- {t['en']}" for t in terms)
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user",
                      "content": PROMPT.replace("{terms}", tl)}],
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "thinking": {"type": "disabled"},
    }).encode()
    for attempt in range(4):
        try:
            req = urllib.request.Request(
                BASE + "/chat/completions", data=body,
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {KEY}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                txt = json.load(r)["choices"][0]["message"]["content"].strip()
            s, e = txt.find("{"), txt.rfind("}")
            obj = json.loads(txt[s:e + 1])
            items = obj.get("items") or []
            if len(items) >= len(terms) * 0.9:
                return items
            print(f"  short return {len(items)}/{len(terms)}", flush=True)
        except Exception as e:
            print(f"  batch retry {attempt}: {str(e)[:100]}", flush=True)
            time.sleep(3 * (attempt + 1))
    return []


def main():
    terms = json.load(open(SRC, encoding="utf-8"))
    kept, zh2en, en2en, dropped, start = [], {}, {}, 0, 0
    if os.path.exists(CKPT):
        ck = json.load(open(CKPT, encoding="utf-8"))
        kept, zh2en, en2en = ck["kept"], ck["zh2en"], ck.get("en2en", {})
        dropped, start = ck["dropped"], ck["next"]
        print(f"resume from batch at {start}", flush=True)
    t0 = time.time()
    for i in range(start, len(terms), B):
        chunk = terms[i:i + B]
        items = batch(chunk)
        got = {it.get("en", "").strip().lower(): it for it in items
               if isinstance(it, dict)}
        for t in chunk:
            it = got.get(t["en"].strip().lower())
            if not it or it.get("drop"):
                dropped += 1
                continue
            zhs = [z.strip() for z in (it.get("zh") or []) if z and z.strip()]
            e2s = [x.strip() for x in (it.get("en2") or [])
                   if x and x.strip() and x.strip().lower() != t["en"].strip().lower()]
            kept.append({"en": t["en"], "tf": t["tf"], "zh": zhs, "en2": e2s})
            for z in zhs:
                zh2en.setdefault(z, t["en"])    # 冲突保留先出现(tf 更高)
            for x in e2s:
                en2en.setdefault(x.lower(), t["en"])
        done = i + len(chunk)
        print(f"[{done}/{len(terms)}] kept={len(kept)} dropped={dropped} "
              f"{time.time()-t0:.0f}s", flush=True)
        json.dump({"kept": kept, "zh2en": zh2en, "en2en": en2en,
                   "dropped": dropped, "next": done}, open(CKPT, "w",
                                                           encoding="utf-8"),
                  ensure_ascii=False)
        time.sleep(0.5)
    glossary = {
        "version": "2026-10-11",
        "source": f"全量 chunk 词频提取 + LLM 判定翻译(coding plan);{SRC}",
        "en_terms": kept,
        "zh2en": zh2en,
        "en2en": en2en,
    }
    # 合并模式:CKPT 文件名含 'incr' 时,并入已有 glossary.json(同键优选更长英文)
    if "incr" in os.path.basename(CKPT) and os.path.exists(OUT):
        old = json.load(open(OUT, encoding="utf-8"))
        merged = {it["en"].strip().lower(): it for it in old.get("en_terms", [])}
        for it in kept:
            k = it["en"].strip().lower()
            if k in merged:
                o = merged[k]
                o["zh"] = list(dict.fromkeys((o.get("zh") or []) + it["zh"]))
                o["en2"] = list(dict.fromkeys((o.get("en2") or []) + it["en2"]))
            else:
                merged[k] = it
        terms_all = sorted(merged.values(), key=lambda x: -x.get("tf", 0))
        z2, e2 = {}, {}
        for it in terms_all:
            en = it["en"].strip()
            for z in it["zh"]:
                if not z.strip():
                    continue
                if z not in z2 or len(en) > len(z2[z]):
                    z2[z] = en
            for x in it["en2"]:
                xl = x.strip().lower()
                if xl and xl != en.lower():
                    e2.setdefault(xl, en)
        glossary = {
            "version": "2026-10-11-r2",
            "source": "全量 chunk 词频提取(大写词组+缩写+型号+小写词组频带),LLM 判定翻译",
            "en_terms": terms_all,
            "zh2en": z2,
            "en2en": e2,
        }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(glossary, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"== 术语 {len(kept)} 条 zh映射 {len(zh2en)} en别名 {len(en2en)} -> {OUT}")


if __name__ == "__main__":
    main()
