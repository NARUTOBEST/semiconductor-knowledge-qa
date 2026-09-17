# -*- coding: utf-8 -*-
"""qdrant ald_text 乱码块扫描(在 VM 上跑,连 localhost:6333)。

背景:日文手册 PDF 提取的 chunk 里混有"䄺ਞ"类乱码字符(CJK 扩展A/
其他语系字母),污染 dense 向量。本脚本量化:按文档统计含可疑字符的
chunk 数与典型样本,产出清洗清单(重嵌需 GPU 在线)。

判定:字符不在 [ASCII / CJK统一表意 U+4E00-9FFF / 常用中文标点 /
日文假名 / 全角形式] 即可疑;chunk 可疑字符占比 > 0.5% 记为乱码块。
用法: python3 mojibake_scan.py [--limit 20000]
"""
import argparse
import collections
import json
import urllib.request

BASE = "http://127.0.0.1:6333"
COLL = "ald_text"


def _ok_char(ch: str) -> bool:
    o = ord(ch)
    if ch.isspace() or o < 128:                      # ASCII/空白
        return True
    if 0x4E00 <= o <= 0x9FFF:                        # CJK 统一表意
        return True
    if 0x3000 <= o <= 0x30FF:                        # 中文标点+假名
        return True
    if 0xFF00 <= o <= 0xFFEF:                        # 全角形式
        return True
    if ch in "—–…·“”‘’《》〈〉！？｡｢｣":
        return True
    return False


def scroll_all(batch=256, limit=None):
    offset = None
    n = 0
    while True:
        body = {"limit": batch, "with_payload": True,
                "with_vector": False, "order_by": None}
        if offset is not None:
            body["offset"] = offset
        req = urllib.request.Request(
            f"{BASE}/collections/{COLL}/points/scroll",
            data=json.dumps(body).encode(), method="POST",
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.loads(r.read())
        pts = data["result"]["points"]
        if not pts:
            break
        for p in pts:
            yield p
            n += 1
            if limit and n >= limit:
                return
        offset = data["result"].get("next_page_offset")
        if offset is None:
            break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--samples", type=int, default=2, help="每文档打印样本数")
    args = ap.parse_args()

    total = 0
    bad_by_doc = collections.Counter()
    total_by_doc = collections.Counter()
    samples = collections.defaultdict(list)
    char_counter = collections.Counter()

    for p in scroll_all(limit=args.limit):
        pl = p.get("payload") or {}
        stem = str(pl.get("source_stem", "?"))
        content = str(pl.get("content", ""))
        total += 1
        total_by_doc[stem] += 1
        weird = [c for c in content if not _ok_char(c)]
        if weird:
            ratio = len(weird) / max(len(content), 1)
            if ratio > 0.005:
                bad_by_doc[stem] += 1
                char_counter.update(weird)
                if len(samples[stem]) < args.samples:
                    cid = pl.get("chunk_id", p["id"])
                    samples[stem].append(
                        (cid, round(ratio, 4), "".join(weird[:12]),
                         content[:60].replace("\n", " ")))

    print(f"总 chunk 数: {total}")
    print(f"乱码块总数: {sum(bad_by_doc.values())} "
          f"({sum(bad_by_doc.values()) / max(total,1) * 100:.1f}%)")
    print("\n按文档统计(乱码块数/该文档块数):")
    for stem, bad in bad_by_doc.most_common(30):
        print(f"  {stem}: {bad}/{total_by_doc[stem]}")
    print("\nTop 可疑字符(码点/字符/次数):")
    for ch, cnt in char_counter.most_common(25):
        print(f"  U+{ord(ch):04X} {ch!r} ×{cnt}")
    print("\n样本:")
    for stem, ss in list(samples.items())[:15]:
        for cid, ratio, weird, head in ss:
            print(f"  [{stem}] {cid} ratio={ratio} weird={weird!r}")
            print(f"      {head}")


if __name__ == "__main__":
    main()
