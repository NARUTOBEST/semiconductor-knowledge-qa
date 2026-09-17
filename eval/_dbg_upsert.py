# -*- coding: utf-8 -*-
"""upsert 400 最小复现:单坏块走完整修复链,打印 qdrant 错误体。"""
import json
import sys
import urllib.request

sys.path.insert(0, "/home/lly/eval")
from mojibake_fix import (scroll_all, susp_ratio, THRESH, try_recover,
                          sanitize, embed_texts, BASE_Q)

qid = None
for p in scroll_all():
    pl = p.get("payload") or {}
    c = str(pl.get("content", ""))
    if c and susp_ratio(c) > THRESH:
        qid = p["id"]
        break

cand = try_recover(c) or sanitize(c)
hp = pl.get("heading_path", "") or ""
dense, sparse = embed_texts([hp + "\n" + cand])[0]
nan = any(x != x for x in dense)
print("dense len:", len(dense), "sparse n:", len(sparse["indices"]),
      "nan in dense:", nan)

payload = dict(pl)
payload["content"] = cand
payload["char_count"] = len(cand)
payload["mojibake_fixed_at"] = 0.0
pt = {"id": qid, "vector": {"dense": dense, "sparse": sparse},
      "payload": payload}
body = json.dumps({"points": [pt]}).encode()
print("body bytes:", len(body))
req = urllib.request.Request(
    BASE_Q + "/collections/ald_text/points", data=body, method="POST",
    headers={"Content-Type": "application/json"})
try:
    with urllib.request.urlopen(req, timeout=60) as r:
        print("UPSERT OK:", r.read()[:200])
except urllib.error.HTTPError as e:
    print("ERR", e.code, e.read()[:800])
