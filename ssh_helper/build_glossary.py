# -*- coding: utf-8 -*-
"""用 ARK(Doubao)把英文技术术语批量翻译成中文常用/口语叫法,产出术语表。
输入 D:/tmp_terms_raw.json;输出 mcp_servers/retrieval/glossary.json。
"""
import json, os, time
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("ARK_API_KEY", ""),
    base_url="https://ark.cn-beijing.volces.com/api/v3")
MODEL = "ep-20260917012501-44sxb"

PROMPT = (
    "你是半导体设备领域的术语专家。下面给你一批英文术语(来自半导体设备技术手册语料)。\n"
    "对每个术语:\n"
    "1. 判断它是否为\"技术名词\"(设备/部件/工艺/材料/参数/指标)。纯泛用词(如 function, "
    "addition, For example)标记 drop=true;\n"
    "2. 技术名词给出中文叫法 1~3 个:第一个是最常用的规范译名,后面可补工程口语/常见别名"
    "(如 megasonic cleaning -> [\"兆声清洗\",\"兆声波清洗\"]);单晶圆设备品牌/型号(如 Mobilith "
    "SHC460WT)保留原文不译,zh 给空数组;\n"
    "3. 严禁臆造不存在的中文叫法。\n"
    "只输出 JSON:{\"items\":[{\"en\":\"...\",\"drop\":false,\"zh\":[\"...\",\"...\"]}]},"
    "items 顺序与输入一致。\n\n术语列表:\n{terms}"
)


def batch(terms):
    tl = "\n".join(f"- {t['en']}" for t in terms)
    for attempt in range(3):
        try:
            r = client.chat.completions.create(
                model=MODEL, temperature=0,
                messages=[{"role": "user", "content": PROMPT.replace("{terms}", tl)}],
                response_format={"type": "json_object"})
            txt = r.choices[0].message.content or ""
            s, e = txt.find("{"), txt.rfind("}")
            obj = json.loads(txt[s:e + 1])
            items = obj.get("items") or []
            if len(items) >= len(terms) * 0.9:
                return items
        except Exception as e:
            print(f"  batch retry {attempt}: {str(e)[:80]}", flush=True)
            time.sleep(3)
    return []


CKPT = "D:/tmp_glossary_ckpt.json"


def main():
    terms = json.load(open("D:/tmp_terms_raw.json", encoding="utf-8"))
    B = 50
    kept, zh2en, dropped, start = [], {}, 0, 0
    if os.path.exists(CKPT):           # 断点续跑
        ck = json.load(open(CKPT, encoding="utf-8"))
        kept, zh2en, dropped, start = ck["kept"], ck["zh2en"], ck["dropped"], ck["next"]
        print(f"resume from batch at {start}", flush=True)
    for i in range(0, len(terms), B):
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
            if not zhs:
                continue            # 保留原文型号,不做中→英映射
            kept.append({"en": t["en"], "tf": t["tf"], "zh": zhs})
            for z in zhs:
                zh2en.setdefault(z, t["en"])   # 冲突时保留先出现的(tf 更高)
        print(f"[{i + len(chunk)}/{len(terms)}] kept={len(kept)} dropped={dropped}",
              flush=True)
        json.dump({"kept": kept, "zh2en": zh2en, "dropped": dropped,
                   "next": i + B}, open(CKPT, "w", encoding="utf-8"),
                  ensure_ascii=False)
        time.sleep(1)
    glossary = {
        "version": "2026-10-04",
        "source": "pool_2000+dev200+heldout200 evidence 语料挖掘,ARK 判定翻译",
        "en_terms": kept,
        "zh2en": zh2en,
    }
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "mcp_servers/retrieval/glossary.json")
    json.dump(glossary, open(out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"== 术语 {len(kept)} 条,中文映射 {len(zh2en)} 条 -> {out}")


if __name__ == "__main__":
    main()
