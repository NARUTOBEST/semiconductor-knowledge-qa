# -*- coding: utf-8 -*-
"""qdrant ald_text 乱码块定向修复:检测 -> 恢复/消毒 -> 重嵌 -> 回写。

背景(mojibake_scan.py 量化):约 1.2% chunk 含三类乱码——
  A. 编码误解码(GBK/UTF-8 链)产生的不可读 CJK 扩展区 junk(最伤 dense 向量)
  B. C1 控制符 + U+FFFD(PDF 提取丢字,局部损坏)
  C. PUA 私用区字体符号(外观问题,连带给 embedding 添噪声)

修复策略(保守,宁可不改也不改坏):
  1) 先试编码恢复链 text.encode(X).decode(Y);恢复结果必须通过
     "可疑字符占比下降且长度不缩水超 40%" 校验才采纳;
  2) 恢复失败则消毒:仅删除 C1/FFFD/PUA 字符;删完仍超阈值则放弃该块;
  3) 只对内容发生变化的块重嵌(embed_text = heading_path + "\n" + content,
     与 chunker.py:719 一致)并整点回写(payload 保留原字段+打标)。

用法(VM 上跑): python3 mojibake_fix.py [--dry-run] [--limit N]
依赖:localhost:6333(qdrant)、localhost:8002(检索 /embed_text,VM 隧道)。
"""
import argparse
import json
import time
import urllib.request

BASE_Q = "http://127.0.0.1:6333"
BASE_R = "http://127.0.0.1:8002"
COLL = "ald_text"
TOKEN = "REDACTED-TOKEN"  # 与 ssh_helper/ret.env 一致
BATCH = 32


# ---------- 字符分类 ----------

def _suspicious(ch):
    """真乱码类字符:C1 控制符 / U+FFFD / PUA / CJK 扩展A(罕见,误解码高发)。"""
    o = ord(ch)
    if 0x80 <= o <= 0x9F:          # C1 控制符
        return True
    if o == 0xFFFD:                # 替换符
        return True
    if 0xE000 <= o <= 0xF8FF:      # PUA 私用区
        return True
    if 0x3400 <= o <= 0x4DBF:      # CJK 扩展A
        return True
    return False


def susp_ratio(text):
    if not text:
        return 0.0
    n = sum(1 for c in text if _suspicious(c))
    return n / len(text)


THRESH = 0.005  # 与 scan 口径一致:可疑占比 >0.5% 记为乱码块


# ---------- 恢复 / 消毒 ----------

RECOVERY_CHAINS = [
    ("gbk", "utf-8"),      # UTF-8 字节被当 GBK 解
    ("latin-1", "utf-8"),  # UTF-8 字节被当 Latin-1 解
]


def try_recover(text):
    """编码链恢复;通过校验返回新文本,否则 None。"""
    best = None
    for enc, dec in RECOVERY_CHAINS:
        try:
            cand = text.encode(enc, errors="ignore").decode(dec, errors="strict")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
        if not cand or len(cand) < 0.6 * len(text):
            continue
        if susp_ratio(cand) < susp_ratio(text) * 0.5:
            if best is None or susp_ratio(cand) < susp_ratio(best):
                best = cand
    return best


def sanitize(text):
    """兜底:仅删除确定坏类的字符(C1/FFFD/PUA),保留 CJK 扩展A 以防误删真罕字。"""
    return "".join(
        ch for ch in text
        if not (0x80 <= ord(ch) <= 0x9F or ord(ch) == 0xFFFD
                or 0xE000 <= ord(ch) <= 0xF8FF))


# ---------- qdrant / 检索服务 ----------

def _post(url, body, headers=None, timeout=120):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json", **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def scroll_all(batch=256):
    offset = None
    while True:
        body = {"limit": batch, "with_payload": True, "with_vector": False}
        if offset is not None:
            body["offset"] = offset
        data = _post(f"{BASE_Q}/collections/{COLL}/points/scroll", body)
        pts = data["result"]["points"]
        if not pts:
            return
        yield from pts
        offset = data["result"].get("next_page_offset")
        if offset is None:
            return


def embed_texts(texts):
    data = _post(f"{BASE_R}/embed_text", {"texts": texts},
                 headers={"X-Internal-Token": TOKEN}, timeout=300)
    out = []
    for d, sp in zip(data["dense"], data["sparse"]):
        idx = [int(k) for k in (sp or {}).keys()]
        val = [float(v) for v in (sp or {}).values()]
        out.append((d, {"indices": idx, "values": val}))
    return out


def upsert_points(points):
    req = urllib.request.Request(
        f"{BASE_Q}/collections/{COLL}/points",
        data=json.dumps({"points": points}).encode(), method="PUT",
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        json.loads(r.read())


# ---------- 主流程 ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import uuid
    def pid(chunk_id):
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))

    bad = []
    total = 0
    t0 = time.time()
    for p in scroll_all():
        total += 1
        pl = p.get("payload") or {}
        content = str(pl.get("content", ""))
        if content and susp_ratio(content) > THRESH:
            bad.append((p["id"], pl))
            if args.limit and len(bad) >= args.limit:
                break
    print(f"扫描 {total} 块,坏块 {len(bad)} "
          f"({len(bad)/max(total,1)*100:.2f}%),耗时 {time.time()-t0:.0f}s")

    fixed = []
    stats = {"recovered": 0, "sanitized": 0, "skipped": 0}
    for qid, pl in bad:
        content = pl.get("content", "")
        cand = try_recover(content)
        if cand is not None:
            stats["recovered"] += 1
        else:
            cand = sanitize(content)
            if cand and susp_ratio(cand) <= THRESH and cand != content:
                stats["sanitized"] += 1
            else:
                stats["skipped"] += 1
                continue
        fixed.append((qid, pl, cand))
    print(f"恢复 {stats['recovered']},消毒 {stats['sanitized']},"
          f"放弃 {stats['skipped']},待回写 {len(fixed)}")

    if args.dry_run:
        for qid, pl, cand in fixed[:8]:
            print("---", pl.get("chunk_id"))
            print("  旧:", str(pl.get("content"))[:80].replace("\n", " "))
            print("  新:", cand[:80].replace("\n", " "))
        return

    t1 = time.time()
    n_up = 0
    for i in range(0, len(fixed), BATCH):
        chunk_batch = fixed[i:i + BATCH]
        texts = [f"{pl.get('heading_path','')}\n{cand}" for _, pl, cand in chunk_batch]
        vecs = embed_texts(texts)
        pts = []
        for (qid, pl, cand), (dense, sparse) in zip(chunk_batch, vecs):
            payload = dict(pl)
            payload["content"] = cand
            payload["char_count"] = len(cand)
            payload["mojibake_fixed_at"] = time.time()
            pts.append({"id": qid, "vector": {"dense": dense, "sparse": sparse},
                        "payload": payload})
        upsert_points(pts)
        n_up += len(pts)
        print(f"  已回写 {n_up}/{len(fixed)}")
    print(f"完成:回写 {n_up} 块,耗时 {time.time()-t1:.0f}s")


if __name__ == "__main__":
    main()
