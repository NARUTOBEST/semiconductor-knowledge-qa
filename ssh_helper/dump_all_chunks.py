# -*- coding: utf-8 -*-
"""全量 dump ald_text 块文本(经 6333 转发 -> VM qdrant),供术语表提取。
断点续跑:按 offset 增量,输出 append 到 jsonl。
"""
import json
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:6333/collections/ald_text/points/scroll"
OUT = sys.argv[1] if len(sys.argv) > 1 else "ssh_helper/all_chunks.jsonl"
PAGE = 4000


def main():
    offset = None
    n = 0
    t0 = time.time()
    with open(OUT, "a", encoding="utf-8") as f:
        while True:
            body = {"limit": PAGE, "with_payload": True,
                    "with_vector": False}
            if offset:
                body["offset"] = offset
            req = urllib.request.Request(
                BASE, data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"})
            for attempt in range(5):
                try:
                    with urllib.request.urlopen(req, timeout=300) as r:
                        res = json.load(r)["result"]
                    break
                except Exception as e:
                    print(f"retry {attempt}: {e}", flush=True)
                    time.sleep(5 * (attempt + 1))
            else:
                raise SystemExit("scroll failed after retries")
            pts = res.get("points", [])
            if not pts:
                break
            for p in pts:
                pl = p.get("payload", {})
                f.write(json.dumps({
                    "chunk_id": pl.get("chunk_id"),
                    "stem": pl.get("source_stem"),
                    "content": pl.get("content"),
                }, ensure_ascii=False) + "\n")
            n += len(pts)
            offset = res.get("next_page_offset")
            if n % 20000 < PAGE:
                print(f"{n} pts, {time.time()-t0:.0f}s", flush=True)
            if not offset:
                break
    print(f"DONE {n} pts in {time.time()-t0:.0f}s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
