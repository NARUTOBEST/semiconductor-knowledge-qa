# -*- coding: utf-8 -*-
"""池内 badcase 三臂对照:金块在候选池却进不了 top-4 的题,离线比较
  ZH  现状:中文 query 重排(ranked 序前4)
  EN  翻译:英文 query 重排
  MIX 分区:英文块用英文分、中文块用中文分,合成后排序

用法: python test_inpool4.py
输出: runs/inpool4_result.json + 控制台表
"""
import glob
import json
import os
import re
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", "..", "..", ".."))
for _line in open(os.path.join(ROOT, "env", "env.env"), encoding="utf-8"):
    _line = _line.strip()
    if _line and not _line.startswith("#") and "=" in _line:
        _k, _, _v = _line.partition("=")
        os.environ.setdefault(_k.strip(), _v.strip())

ARK_KEY = os.getenv("OPENAI_API_KEY", "")
ARK_BASE = "https://ark.cn-beijing.volces.com/api/coding/v1"
MODEL = "doubao-seed-2-0-lite-260428"
TRAN_PROMPT = ("把用户的中文问题翻译成一句用于检索英文半导体设备手册的英文查询。"
               "保留型号/数字/单位原样,输出英文查询本身,不要解释。\n\n用户问题:")

BASE = "http://127.0.0.1:8002"
TOKEN = ""
CJK = re.compile(r"[一-鿿]")
LAT = re.compile(r"[a-zA-Z]")


def search(query, k=96):
    body = json.dumps({"query": query, "k": k, "score_ratio": 0.01,
                       "with_pool": True, "with_ranked": True}).encode()
    req = urllib.request.Request(
        BASE + "/search_text", data=body,
        headers={"Content-Type": "application/json",
                 "X-Internal-Token": TOKEN})
    last = None
    for _ in range(3):  # 中文 query 偶发 500,重试
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            last = e
            if e.code >= 500:
                time.sleep(2)
                continue
            raise
    raise last


def translate(q):
    body = json.dumps({"model": MODEL,
                       "messages": [{"role": "user",
                                     "content": TRAN_PROMPT + q}],
                       "temperature": 0, "max_tokens": 120}).encode()
    req = urllib.request.Request(
        ARK_BASE + "/chat/completions", data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + ARK_KEY})
    with urllib.request.urlopen(req, timeout=60) as r:
        d = json.load(r)
    return (d["choices"][0]["message"]["content"] or "").strip()


def lang_of(content):
    zh = len(CJK.findall(content))
    en = len(LAT.findall(content))
    return "zh" if zh > en else "en"


def load_targets():
    """低 recall 题里金块在中文池内的(ranked 含金块即算池内)。"""
    gold = json.load(open(os.path.join(HERE, "runs", "gold_cache.json"),
                          encoding="utf-8"))
    recs = {}
    for f in glob.glob(os.path.join(HERE, "runs",
                                    "dev200_round8_final_judge", "*.jsonl")):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            if r.get("judge_done") and r.get("judge_context_recall") is not None:
                recs[r["id"]] = r
    targets = []
    for qid, v in recs.items():
        if v["judge_context_recall"] >= 0.5:
            continue
        stem = v.get("source") or ""
        gl = set(gold.get(stem, {}).get("gold") or [])
        if not gl:
            continue
        targets.append({"id": qid, "q": v["q"], "stem": stem, "gold": gl})
    return targets


def main():
    global TOKEN
    TOKEN = collect_token()
    targets = load_targets()
    print(f"低 recall 候选 {len(targets)} 题,逐题采池…", flush=True)

    rows = []
    for i, t in enumerate(targets, 1):
        row = {"id": t["id"], "arms": {}}
        try:
            zh = search(t["q"])
            time.sleep(0.3)
            en = search(translate(t["q"]))
        except Exception as e:  # noqa: BLE001
            print(f"  [{i}/{len(targets)}] {t['id']} ERR {repr(e)[:80]}",
                  flush=True)
            row["error"] = repr(e)[:120]
            rows.append(row)
            continue

        # 每臂:ranked 过滤到 items(k=96,有 content)后的前4
        def top4(d):
            items = {it["chunk_id"]: it for it in (d.get("items") or [])}
            out = []
            for c in (d.get("ranked") or []):
                if c["chunk_id"] in items:
                    out.append(c)
                if len(out) >= 4:
                    break
            return [c["chunk_id"] for c in out], items

        zh_top, zh_items = top4(zh)
        en_top, en_items = top4(en)

        gold = t["gold"]
        zh_rank = [c["chunk_id"] for c in zh["ranked"]]
        pos = next((p + 1 for p, c in enumerate(zh_rank) if c in gold), None)
        row["zh_pool_pos"] = pos
        if pos is None:
            row["arms"]["note"] = "金块不在中文池(盲区题,非本测对象)"
            rows.append(row)
            print(f"  [{i}/{len(targets)}] {t['id']} 盲区 pos=None",
                  flush=True)
            continue

        # MIX:并集池,英文块取英文分、中文块取中文分
        lang = {}
        for it in list(zh_items.values()) + list(en_items.values()):
            lang[it["chunk_id"]] = lang_of(it.get("content") or "")
        zsc = {c["chunk_id"]: c["score"] for c in zh["ranked"]}
        esc = {c["chunk_id"]: c["score"] for c in en["ranked"]}
        universe = set(zsc) | set(esc)
        mixed = []
        for cid in universe:
            s = esc.get(cid) if lang.get(cid) == "en" else zsc.get(cid)
            if s is None:  # 无该语言路分数,用另一路补
                s = esc.get(cid, zsc.get(cid))
            if s is not None:
                mixed.append((s, cid))
        mixed.sort(reverse=True)
        mix_top = [c for _, c in mixed[:4]]

        hit = lambda ids: any(c in gold for c in ids)  # noqa: E731
        row["arms"] = {"ZH": hit(zh_top), "EN": hit(en_top), "MIX": hit(mix_top)}
        row["zh_top4"] = zh_top[:4]
        row["en_translated"] = en.get("query") or ""
        rows.append(row)
        print(f"  [{i}/{len(targets)}] {t['id']} pos={pos} "
              f"ZH={row['arms']['ZH']:d} EN={row['arms']['EN']:d} "
              f"MIX={row['arms']['MIX']:d}", flush=True)
        time.sleep(0.3)

    with open(os.path.join(HERE, "runs", "inpool4_result.json"), "w",
              encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)

    ok = [r for r in rows if r["arms"].get("ZH") is not None
          or r["arms"].get("note")]
    pool = [r for r in rows if "ZH" in r.get("arms", {})]
    print(f"\n池内 badcase {len(pool)} 题,三臂命中 top-4:")
    for arm in ("ZH", "EN", "MIX"):
        print(f"  {arm}: {sum(r['arms'].get(arm, False) for r in pool)}"
              f"/{len(pool)}")
    blind = [r["id"] for r in rows if r["arms"].get("note")]
    if blind:
        print("盲区题(金块不在中文池,不计入):", len(blind))


def collect_token():
    for line in open(os.path.join(ROOT, "env", "env.env"), encoding="utf-8"):
        if line.startswith("RETRIEVAL_INTERNAL_TOKEN="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("token not found")


if __name__ == "__main__":
    main()
