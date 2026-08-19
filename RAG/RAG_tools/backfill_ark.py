# -*- coding: utf-8 -*-
"""回填 LLM 图描述到已入库的图像点(不重编码、不重切分、不 --force)。

遍历 ald_image 中 description 为空的点 -> 按 PDF 分组 -> 调 LLM 模型(per-PDF 缓存,
复用 ark_client.describe_with_cache) -> Qdrant set_payload 回填 description。

适用场景:先 --no-desc 把 text+CLIP 便宜入库,再渐进补 LLM 图描述。
全库也用这个,不重跑 ingest。"""
import os, sys
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from qdrant_client import QdrantClient

_HERE = os.path.dirname(os.path.abspath(__file__))
_RAG = os.path.dirname(_HERE)
_PROJECT = os.path.dirname(_RAG)
_CONFIG = os.path.join(_PROJECT, "config")
for _p in (_CONFIG, _RAG):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import config as C
import ark_client


def _scroll_all(client):
    """返回所有图像点 payload 列表。"""
    out, off = [], None
    while True:
        res, off = client.scroll(
            collection_name=C.IMAGE_COLLECTION,
            limit=256, offset=off, with_payload=True, with_vectors=False)
        out.extend(res)
        if not off:
            break
    return out


def backfill():
    client = QdrantClient(path=C.QDRANT_PATH)
    pts = _scroll_all(client)

    # 按 auto_dir 分组(复用 per-PDF 缓存,每组只 load 一次)
    groups = {}          # auto_dir -> [(point_id, img_path, img_relname)]
    n_have = n_noimg = 0
    for p in pts:
        pl = p.payload or {}
        if (pl.get("description") or "").strip():
            n_have += 1
            continue
        img_path = pl.get("image_path")
        if not img_path or not os.path.exists(img_path):
            n_noimg += 1
            continue
        auto_dir = os.path.dirname(os.path.dirname(img_path))   # .../<stem>/auto
        rel = os.path.relpath(img_path, auto_dir)               # images/<hash>.jpg
        groups.setdefault(auto_dir, []).append((str(p.id), img_path, rel))

    todo = sum(len(v) for v in groups.values())
    print(f"总 {len(pts)} 图:已有描述 {n_have} | 无图文件 {n_noimg} | 待回填 {todo}"
          f" (分 {len(groups)} 个 PDF)\n")

    done = failed = 0
    for auto_dir, items in groups.items():
        stem = os.path.basename(os.path.dirname(auto_dir))
        cache = ark_client.load_desc_cache(auto_dir)
        print(f"== {stem} ({len(items)} 图) ==")
        updates = []
        for pid, img_path, rel in items:
            try:
                d, cache = ark_client.describe_with_cache(rel, img_path, auto_dir, cache)
                updates.append((pid, d))
                done += 1
                print(f"  [{done}/{todo}] {os.path.basename(img_path)[:20]}  {d[:50]}")
            except Exception as e:
                failed += 1
                print(f"  [fail] {os.path.basename(img_path)}: {str(e)[:70]}")
        for pid, d in updates:   # 批量回写
            client.set_payload(C.IMAGE_COLLECTION, payload={"description": d}, points=[pid])
    print(f"\n完成:回填 {done},失败 {failed}")


if __name__ == "__main__":
    backfill()
