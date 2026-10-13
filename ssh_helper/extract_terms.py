# -*- coding: utf-8 -*-
"""从全量 chunk 文本统计提炼英文专有术语(轮次19 术语表)。
方法:正则抽取候选(大写词组/缩写/型号) + 全库词频,频次过滤 + 去重。
输出:tmp_terms_raw.json  [{"en": ..., "tf": ...}] 供 build_glossary.py 批译。
"""
import json
import re
import sys
from collections import Counter

SRC = sys.argv[1] if len(sys.argv) > 1 else "ssh_helper/all_chunks.jsonl"
OUT = sys.argv[2] if len(sys.argv) > 2 else "ssh_helper/tmp_terms_raw.json"
TOP = int(sys.argv[3]) if len(sys.argv) > 3 else 4000
MIN_TF = 4

# 通用词/排版词黑名单(大写也常出现,非术语)
STOP = {"the", "and", "for", "with", "note", "caution", "warning", "figure",
        "table", "chapter", "section", "page", "manual", "contents",
        "appendix", "revision", " specification", "continued", "example",
        "default", "description", "parameter", "parameters", "setting",
        "settings", "function", "display", "menu", "screen", "status",
        "system", "control", "operation", "installation", "maintenance",
        "troubleshooting", "overview", "introduction", "general", "item",
        "items", "step", "steps", "no", "yes", "on", "off", "ok", "id",
        "part", "name", "type", "size", "mode", "data", "time", "date",
        "led", "lcd", "usb", "ac", "dc", "rpm", "psi", "kpa", "mpa",
        "circuit", "diagram", "wiring", "power", "supply", "voltage",
        "current", "signal", "output", "input", "error", "alarm", "code",
        "press", "touch", "select", "enter", "set", "clear", "start",
        "stop", "open", "close", "this", "that", "these", "those"}

RE_MULTI = re.compile(
    r"\b[A-Z][A-Za-z0-9]*(?:[ ](?:of|for|and|the|in|with|to)?[ ]?"
    r"[A-Z][A-Za-z0-9]*){1,3}\b")
RE_ACRONYM = re.compile(r"\b[A-Z]{2,12}\b")
RE_MODEL = re.compile(r"\b[A-Z][A-Za-z0-9]{0,10}(?:-[A-Z0-9][A-Za-z0-9]*)+\b")
# 小写多词术语(buried oxide / megasonic cleaning / end point detector):
# 首版提取器只抓大写开头词组,漏掉语料里大量小写技术词组——二版补抓
RE_LOW = re.compile(r"\b[a-z]{3,}(?: [a-z]{3,}){1,2}\b")
EN_STOP = {"the", "and", "for", "with", "this", "that", "from", "are", "can",
           "will", "when", "where", "which", "into", "onto", "than", "then",
           "not", "but", "all", "any", "each", "per", "via", "use", "used",
           "using", "have", "has", "had", "was", "were", "been", "must",
           "should", "may", "might", "after", "before", "during", "between",
           "such", "same", "other", "more", "most", "less", "least", "only",
           "also", "both", "over", "under", "out", "off", "its", "his",
           "her", "our", "their", "there", "here", "what", "how", "why"}
# 文档样板词(法律声明/排版套话,高频但非术语)
BOILER = {"notice", "copyright", "copyrights", "patent", "patents", "rights",
          "reserved", "asserted", "covered", "mentioned", "approved",
          "edition", "labels", "sheets", "liability", "warranty", "declined",
          "responsibility", "assumes", "disclaims", "permission", "without",
          "except", "provided", "reproduction", "transmission", "retrieval",
          "photocopying", "recording", "scanning", "information", "material",
          "materials", "products", "manufacturer", "supplier", "reference",
          "consult", "contact", "address", "establish", "ensure", "ensures",
          "appropriate", "applicable", "relevant", "following", "above",
          "below", "given", "shown", "listed", "described", "discussed"}


def ok_multi(t):
    words = t.split()
    if not (2 <= len(words) <= 4):
        return False
    for w in words:
        lw = w.lower()
        if lw in STOP and w[0].isupper() and len(words) > 1 and lw in (
                "the", "and", "of", "for"):
            continue
        if lw in STOP and lw not in ("the", "and", "of", "for"):
            return False
    return True


def ok_acronym(t):
    return t not in STOP and len(t) >= 2


def ok_model(t):
    return len(t) >= 3 and not t.isdigit()


def main():
    cnt_multi, cnt_acro, cnt_model = Counter(), Counter(), Counter()
    cnt_low = Counter()
    n = 0
    for line in open(SRC, encoding="utf-8"):
        try:
            c = json.loads(line).get("content") or ""
        except Exception:
            continue
        if not c:
            continue
        n += 1
        for m in RE_MULTI.findall(c):
            m = m.strip()
            if ok_multi(m):
                cnt_multi[m] += 1
        for m in RE_ACRONYM.findall(c):
            if ok_acronym(m):
                cnt_acro[m] += 1
        for m in RE_MODEL.findall(c):
            if ok_model(m):
                cnt_model[m.lower()] += 1     # 型号统一小写去重
        for m in RE_LOW.findall(c):
            words = m.split()
            if any(w in EN_STOP or w in BOILER for w in words):
                continue
            cnt_low[m] += 1
            # 贪婪匹配会把短词组吃进长词组(buried oxide 层),短词组计零——
            # 给各连续子词组也计一次
            if len(words) == 3:
                cnt_low[words[0] + " " + words[1]] += 1
                cnt_low[words[1] + " " + words[2]] += 1
    out = []
    for t, tf in cnt_multi.most_common(TOP):
        if tf >= MIN_TF:
            out.append({"en": t, "tf": tf})
    for t, tf in cnt_acro.most_common(TOP):
        if tf >= MIN_TF * 2:                  # 缩写噪声大,阈值加倍
            out.append({"en": t, "tf": tf})
    for t, tf in cnt_model.most_common(TOP):
        if tf >= MIN_TF:
            out.append({"en": t, "tf": tf})
    for t, tf in cnt_low.most_common(TOP):
        if tf >= MIN_TF * 2:                  # 小写词组噪声大,阈值加倍
            out.append({"en": t, "tf": tf})
    # 按词面去重(大小写不敏感),保 tf 高的写法
    seen = {}
    for it in sorted(out, key=lambda x: -x["tf"]):
        k = it["en"].lower()
        if k not in seen:
            seen[k] = it
    out = list(seen.values())[:TOP]
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"chunks={n} 多词术语候选={len(cnt_multi)} 缩写={len(cnt_acro)} "
          f"型号={len(cnt_model)} 小写词组={len(cnt_low)} "
          f"-> 保留 {len(out)} 条 -> {OUT}")


if __name__ == "__main__":
    main()
