# -*- coding: utf-8 -*-
"""给存量 Qdrant 块打设备型号标签(payload.devices),配合检索侧双路融合。

配套 mcp_servers/retrieval/devices.py(同一套 token 规则)与 query.py 的
devices 过滤召回路。只写 payload,不动向量 -> 无需重新嵌入,秒级生效。

用法:
  python RAG/RAG_tools/backfill_devices.py            # dry-run:只统计不写入
  python RAG/RAG_tools/backfill_devices.py --apply    # 实际写 payload

连接:读 config(QDRANT_URL 优先)。对 VM 上的 qdrant(绑 127.0.0.1)先开隧道:
  ssh -N -L 16333:127.0.0.1:6333 <vm>,再 QDRANT_URL=http://127.0.0.1:16333 运行。

断点安全:标签按"算出的 tags"整体覆盖写,重复运行幂等;写入走 batch_update_points
的 SetPayloadOperation,每组一批,中断重跑即可。
"""
import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))          # RAG/RAG_tools
_RAG = os.path.dirname(_HERE)                              # RAG
_PROJECT = os.path.dirname(_RAG)
for _p in (os.path.join(_PROJECT, "config"), _RAG):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from qdrant_client import models                           # noqa: E402

import config as C                                         # noqa: E402
import ingest_core                                         # noqa: E402
sys.path.insert(0, os.path.join(_PROJECT, "mcp_servers", "retrieval"))
import devices                                             # noqa: E402

# 每块正文/描述字段(文本库 content,图像库 description+caption)
_TEXT_FIELD = {"ald_text": ("content", "heading_path"),
               "ald_image": ("description", "")}


def iter_points(client, coll, batch=256):
    offset = None
    while True:
        pts, offset = client.scroll(coll, limit=batch, offset=offset,
                                    with_payload=True, with_vectors=False)
        yield from pts
        if offset is None:
            break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="实际写 payload(默认 dry-run)")
    ap.add_argument("--ops-per-call", type=int, default=32, help="每次 batch_update 的操作数")
    ap.add_argument("--ids-per-op", type=int, default=256, help="每个 SetPayload 操作的点数上限")
    a = ap.parse_args()

    client = ingest_core.get_client()
    # 批量 payload 写入会触发 qdrant 异步 payload 索引优化器,期间写入可能被卡
    # 数十秒(默认 30s 客户端超时直接炸,踩坑实录)-> 显式放大超时
    if getattr(C, "QDRANT_URL", ""):
        from qdrant_client import QdrantClient
        client = QdrantClient(url=C.QDRANT_URL, timeout=600)
    for coll in (C.TEXT_COLLECTION, C.IMAGE_COLLECTION):
        try:  # KEYWORD 索引:MatchAny 过滤路必需,重复建无害
            client.create_payload_index(coll, field_name="devices",
                                        field_schema=models.PayloadSchemaType.KEYWORD)
        except Exception:
            pass

    for coll in (C.TEXT_COLLECTION, C.IMAGE_COLLECTION):
        content_f, heading_f = _TEXT_FIELD[coll]
        groups = {}                       # tags tuple -> [point_id]
        n_total = n_tagged = 0
        for p in iter_points(client, coll):
            n_total += 1
            pl = p.payload or {}
            tags = devices.chunk_device_tags(
                str(pl.get(content_f, "") or ""),
                source_stem=str(pl.get("source_stem", "") or ""),
                heading_path=str(pl.get(heading_f, "") or "") if heading_f else "")
            if tags:
                n_tagged += 1
                groups.setdefault(tuple(tags), []).append(p.id)
            if n_total % 50000 == 0:
                print(f"  [{coll}] scanned {n_total}, tagged {n_tagged}", flush=True)
        print(f"[{coll}] 共 {n_total} 块,打上标签 {n_tagged} "
              f"({n_tagged / max(n_total, 1):.1%}),不同标签组 {len(groups)}")

        if not a.apply or not groups:
            print(f"[{coll}] dry-run 结束(--apply 才写入)\n")
            continue

        ops = []
        for tags, ids in groups.items():
            for i in range(0, len(ids), a.ids_per_op):
                ops.append(models.SetPayloadOperation(set_payload=models.SetPayload(
                    payload={"devices": list(tags)},
                    points=ids[i:i + a.ids_per_op])))
        done = 0
        for i in range(0, len(ops), a.ops_per_call):
            chunk = ops[i:i + a.ops_per_call]
            for attempt in range(3):          # 优化器 stall 等瞬态失败重试;标签写入幂等
                try:
                    client.batch_update_points(coll, update_operations=chunk)
                    break
                except Exception as e:
                    wait = 10 * (attempt + 1)
                    print(f"  [{coll}] 批次失败重试{attempt + 1}({type(e).__name__}),"
                          f"{wait}s 后重试", flush=True)
                    time.sleep(wait)
            done = min(len(ops), i + a.ops_per_call)
            if done % 6400 < a.ops_per_call or done == len(ops):
                print(f"  [{coll}] 写入进度 {done}/{len(ops)} ops", flush=True)
        print(f"[{coll}] 写入完成\n")


if __name__ == "__main__":
    main()
