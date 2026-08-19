# -*- coding: utf-8 -*-
"""编排:chunk -> 嵌入(BGE-m3 文本 / CLIP 图)-> [可选]LLM 图描述 -> Qdrant 双库入库。

断点续跑:_ingest_checkpoint.json 记已入库 stem,重跑跳过(--force 强制重做)。
LLM 图描述默认开但容错:单图失败记空描述、不中断;ep-xxx 未就绪时用 --no-desc 跳过。
"""
import os, sys, json, uuid, time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from qdrant_client import QdrantClient, models

_CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")
if _CONFIG not in sys.path:
    sys.path.insert(0, _CONFIG)
import config as C
import chunker, embed, ark_client

CKPT = os.path.join(C.RAG_DIR, "_ingest_checkpoint.json")
TE = None   # 懒加载,进程内复用
IE = None


def _pid(chunk_id):
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))


def get_client():
    if getattr(C, "QDRANT_URL", ""):
        return QdrantClient(url=C.QDRANT_URL)
    os.makedirs(C.QDRANT_PATH, exist_ok=True)
    return QdrantClient(path=C.QDRANT_PATH)


def ensure_collections(client):
    cols = {c.name for c in client.get_collections().collections}
    if C.TEXT_COLLECTION not in cols:
        client.create_collection(
            C.TEXT_COLLECTION,
            vectors_config={"dense": models.VectorParams(size=C.TEXT_DENSE_DIM,
                                                         distance=models.Distance.COSINE)},
            sparse_vectors_config={"sparse": models.SparseVectorParams(
                index=models.SparseIndexParams())},
        )
        print(f"  建集合 {C.TEXT_COLLECTION}(dense {C.TEXT_DENSE_DIM} + sparse)")
    if C.IMAGE_COLLECTION not in cols:
        client.create_collection(
            C.IMAGE_COLLECTION,
            vectors_config={
                "dense": models.VectorParams(size=C.IMAGE_DENSE_DIM,
                                             distance=models.Distance.COSINE),
                "desc_dense": models.VectorParams(size=C.TEXT_DENSE_DIM,
                                                  distance=models.Distance.COSINE),
            },
        )
        print(f"  建集合 {C.IMAGE_COLLECTION}(dense {C.IMAGE_DENSE_DIM} + desc_dense {C.TEXT_DENSE_DIM})")
    # 为 source_stem 创建索引(用于按文档名删除旧 chunk)
    for coll in [C.TEXT_COLLECTION, C.IMAGE_COLLECTION]:
        try:
            client.create_payload_index(
                collection_name=coll, field_name="source_stem",
                field_schema=models.PayloadSchemaType.KEYWORD)
        except Exception:
            pass  # 索引已存在


def _batch_upsert(client, coll, points, size=64):
    for i in range(0, len(points), size):
        client.upsert(coll, points=points[i:i + size])


def delete_by_source_stem(client, stem):
    """删除同一文档的旧 chunk(文本+图像),新版本覆盖旧版本。"""
    flt = models.Filter(must=[
        models.FieldCondition(key="source_stem", match=models.MatchValue(value=stem))
    ])
    try:
        client.delete(C.TEXT_COLLECTION, points_selector=flt)
        client.delete(C.IMAGE_COLLECTION, points_selector=flt)
    except Exception as e:
        print(f"    [warn] 删除旧 chunk 失败: {e}")


def _text_points(chunks, te):
    dense, sparse = te.encode([t["embed_text"] for t in chunks])
    pts = []
    for i, t in enumerate(chunks):
        sp = sparse[i]
        idx = [int(k) for k in sp.keys()]
        val = [float(v) for v in sp.values()]
        payload = {k: t[k] for k in (
            "chunk_id", "content", "source_path", "source_stem",
            "page_start", "page_end", "heading_path",
            "image_paths", "image_descriptions", "has_table", "table_html",
            "chunk_index", "char_count")}
        payload["ingested_at"] = time.time()
        pts.append(models.PointStruct(
            id=_pid(t["chunk_id"]),
            vector={"dense": dense[i].tolist(),
                    "sparse": models.SparseVector(indices=idx, values=val)},
            payload=payload))
    return pts


def _image_points(imgs, ie, te):
    """图像块 -> Qdrant 点(双路向量)。description 已在 chunker 阶段生成。"""
    clip_vecs = ie.encode([v["image_path"] for v in imgs])
    desc_texts = []
    for v in imgs:
        cap = (v.get("caption") or "").strip()
        d = (v.get("description") or "").strip()
        desc_texts.append(f"{cap}\n{d}".strip() or "[无描述]")
    desc_dense, _ = te.encode(desc_texts)
    pts = []
    for i, v in enumerate(imgs):
        if not clip_vecs[i].any():
            # 打不开的图:embed 用零向量占位,此处过滤不入库
            # (零向量在 cosine 集合里是相似度恒为 0 的死点)
            continue
        payload = {k: v[k] for k in (
            "chunk_id", "image_path", "description", "source_path", "source_stem",
            "page_num", "caption", "parent_text_chunk_id", "chunk_index", "item_type")}
        payload["ingested_at"] = time.time()
        pts.append(models.PointStruct(
            id=_pid(v["chunk_id"]),
            vector={
                "dense": clip_vecs[i].tolist(),
                "desc_dense": desc_dense[i].tolist(),
            },
            payload=payload))
    return pts


def process_pdf(stem, auto_dir, source_path, client, do_describe=True):
    # 先切块/编码,全部成功后再删旧 chunk、写入新版本:
    # 若先删后切,切块/编码阶段一旦失败(损坏的 content_list.json、
    # 模型加载失败等),旧数据已删而新数据未写,该文档在库中彻底消失
    tc, ic = chunker.chunk_one_pdf(stem, auto_dir, source_path,
                                   target=C.CHUNK_TARGET, max_size=C.CHUNK_MAX,
                                   min_size=C.CHUNK_MIN,
                                   do_describe=do_describe,
                                   llm_split=ark_client.llm_split_text if do_describe else None)
    # 删旧 chunk(同一文档的新版本覆盖旧版本)
    delete_by_source_stem(client, stem)
    if tc:
        _batch_upsert(client, C.TEXT_COLLECTION, _text_points(tc, TE))
    if ic:
        _batch_upsert(client, C.IMAGE_COLLECTION, _image_points(ic, IE, TE))
    return len(tc), len(ic)


def load_ckpt():
    if os.path.exists(CKPT):
        try:
            return set(json.load(open(CKPT, encoding="utf-8")))
        except Exception:
            return set()
    return set()


def save_ckpt(done):
    json.dump(sorted(done), open(CKPT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


def ingest_dir(clean_subdir, src_root, do_describe=True, force=False):
    """清洗产物某子目录 -> 对应源根,逐 PDF 入库。"""
    global TE, IE
    client = get_client()
    ensure_collections(client)
    TE = TE or embed.get_text_encoder()
    IE = IE or embed.get_image_encoder()
    done = load_ckpt()
    # 收集已清洗的 PDF(auto/<stem>_content_list.json)
    pdfs = []
    for dp, dn, fn in os.walk(clean_subdir):
        if os.path.basename(dp) != "auto":
            continue
        for f in fn:
            if f.endswith("_content_list.json") and "_v2" not in f:
                stem = f[:-len("_content_list.json")]
                pdfs.append((stem, dp, os.path.dirname(dp)))
    pdfs.sort()
    print(f"待入库 PDF: {len(pdfs)}  (已跳过 {len(done & {p[0] for p in pdfs})})")
    for i, (stem, auto, _) in enumerate(pdfs, 1):
        if stem in done and not force:
            continue
        # 源 PDF 路径:由 auto 反推镜像到 src_root
        rel = os.path.relpath(os.path.dirname(auto), C.CLEAN_ROOT)
        src_pdf = os.path.join(src_root, rel, stem + ".pdf")
        t0 = time.time()
        try:
            nt, ni = process_pdf(stem, auto, src_pdf, client, do_describe=do_describe)
            done.add(stem)
            save_ckpt(done)
            print(f"[{i}/{len(pdfs)}] OK {stem[:50]:50s} text={nt:4d} img={ni:3d}  {time.time()-t0:.0f}s")
        except Exception as e:
            print(f"[{i}/{len(pdfs)}] FAIL {stem[:50]:50s} {str(e)[:120]}")
    print(f"完成。累计入库 {len(done)} 个 PDF")


if __name__ == "__main__":
    CLEAN_SUB = r"D:\清洗文件\0001 半导体设备资料合集\ALD原子沉积资料\ALD原子沉积资料"
    SRC_ROOT  = r"C:\project3\180-半导体设备相关资料！"
    do_describe = "--no-desc" not in sys.argv
    force = "--force" in sys.argv
    print(f"LLM 图描述: {'开' if do_describe else '关(仅 CLIP)'}")
    ingest_dir(CLEAN_SUB, SRC_ROOT, do_describe=do_describe, force=force)
