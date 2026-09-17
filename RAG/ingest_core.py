# -*- coding: utf-8 -*-
"""入库通用核心(Qdrant 双库 + BGE-m3 文本 / CLIP 图)。

把 ``RAG/pdf/ingest.py`` 里与文件格式无关的机械抽到这里,供 pdf / docx / mp4
三套入库脚本共用:

  - Qdrant 客户端与集合(ald_text dense+sparse / ald_image dense+desc_dense);
  - chunk -> PointStruct 的文本/图像点构造(与 PDF 版逐字段一致);
  - 按 source_stem 删除旧 chunk 的覆盖式入库;
  - checklist.json 断点(已入库 stem)与 _md5.json 内容去重;
  - 一个批量驱动 ``run_ingest``,格式相关的切块/算指纹通过回调注入。

文本点默认白名单与 PDF 完全一致;格式专属字段(video 的 duration/original_name
等)通过 ``extra_text_fields`` 透传进 payload,不影响检索层既有读取。
"""
import os
import sys
import json
import uuid
import time
import hashlib

# config/ 是无 __init__.py 的目录,真正的模块是 config/config.py;必须把该目录
# 本身加入 sys.path,否则 Python 会把 config/ 当命名空间包导入(缺 SRC_ROOT 等)。
_HERE = os.path.dirname(os.path.abspath(__file__))            # RAG
_PROJECT = os.path.dirname(_HERE)
_CONFIG = os.path.join(_PROJECT, "config")
for _p in (_CONFIG, _PROJECT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from qdrant_client import QdrantClient, models  # noqa: E402

import config as C  # noqa: E402

TE = None   # 文本编码器,懒加载,进程内复用
IE = None   # 图像编码器,懒加载

# 文本块写入 payload 的标准字段(与 RAG/pdf/ingest.py 一致)
_TEXT_FIELDS = (
    "chunk_id", "content", "source_path", "source_stem",
    "page_start", "page_end", "heading_path",
    "image_paths", "image_descriptions", "has_table", "table_html",
    "chunk_index", "char_count")
# 图像块写入 payload 的标准字段
_IMAGE_FIELDS = (
    "chunk_id", "image_path", "description", "source_path", "source_stem",
    "page_num", "caption", "parent_text_chunk_id", "chunk_index", "item_type")


# ---------- 文件指纹 ----------
def md5_file(path, chunk=1 << 20):
    """文件 MD5;不存在或不可读返回 None。"""
    h = hashlib.md5()
    try:
        with open(path, "rb") as f:
            for blk in iter(lambda: f.read(chunk), b""):
                h.update(blk)
        return h.hexdigest()
    except OSError:
        return None


# ---------- Qdrant ----------
def point_id(chunk_id):
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
            vectors_config={"dense": models.VectorParams(
                size=C.TEXT_DENSE_DIM, distance=models.Distance.COSINE)},
            sparse_vectors_config={"sparse": models.SparseVectorParams(
                index=models.SparseIndexParams())})
        print(f"  建集合 {C.TEXT_COLLECTION}(dense {C.TEXT_DENSE_DIM} + sparse)")
    if C.IMAGE_COLLECTION not in cols:
        client.create_collection(
            C.IMAGE_COLLECTION,
            vectors_config={
                "dense": models.VectorParams(
                    size=C.IMAGE_DENSE_DIM, distance=models.Distance.COSINE),
                "desc_dense": models.VectorParams(
                    size=C.TEXT_DENSE_DIM, distance=models.Distance.COSINE)})
        print(f"  建集合 {C.IMAGE_COLLECTION}"
              f"(dense {C.IMAGE_DENSE_DIM} + desc_dense {C.TEXT_DENSE_DIM})")
    for coll in (C.TEXT_COLLECTION, C.IMAGE_COLLECTION):
        try:
            client.create_payload_index(
                collection_name=coll, field_name="source_stem",
                field_schema=models.PayloadSchemaType.KEYWORD)
        except Exception:
            pass  # 索引已存在


def batch_upsert(client, coll, points, size=64):
    for i in range(0, len(points), size):
        client.upsert(coll, points=points[i:i + size])


def delete_by_source_stem(client, stem):
    """删除同一文档的旧 chunk(文本+图像),新版本覆盖旧版本。"""
    flt = models.Filter(must=[
        models.FieldCondition(key="source_stem", match=models.MatchValue(value=stem))])
    try:
        client.delete(C.TEXT_COLLECTION, points_selector=flt)
        client.delete(C.IMAGE_COLLECTION, points_selector=flt)
    except Exception as e:
        print(f"    [warn] 删除旧 chunk 失败: {e}")


def text_points(chunks, te, extra_fields=()):
    """文本块 -> Qdrant 点(dense+sparse)。extra_fields 追加格式专属 payload 字段。"""
    dense, sparse = te.encode([t["embed_text"] for t in chunks])
    pts = []
    for i, t in enumerate(chunks):
        sp = sparse[i]
        idx = [int(k) for k in sp.keys()]
        val = [float(v) for v in sp.values()]
        payload = {k: t.get(k) for k in _TEXT_FIELDS}
        for k in extra_fields:
            if k in t:
                payload[k] = t[k]
        payload["ingested_at"] = time.time()
        pts.append(models.PointStruct(
            id=point_id(t["chunk_id"]),
            vector={"dense": dense[i].tolist(),
                    "sparse": models.SparseVector(indices=idx, values=val)},
            payload=payload))
    return pts


def image_points(imgs, ie, te):
    """图像块 -> Qdrant 点(CLIP dense + 描述 desc_dense)。打不开的图跳过。"""
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
            continue  # 打不开的图:零向量是 cosine 死点,不入库
        payload = {k: v.get(k) for k in _IMAGE_FIELDS}
        payload["ingested_at"] = time.time()
        pts.append(models.PointStruct(
            id=point_id(v["chunk_id"]),
            vector={"dense": clip_vecs[i].tolist(),
                    "desc_dense": desc_dense[i].tolist()},
            payload=payload))
    return pts


# ---------- 断点 / 去重持久化 ----------
def load_json_set(path):
    if os.path.exists(path):
        try:
            return set(json.load(open(path, encoding="utf-8")))
        except Exception:
            return set()
    return set()


def save_json_set(path, done):
    json.dump(sorted(done), open(path, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)


def load_md5map(path):
    if os.path.exists(path):
        try:
            return json.load(open(path, encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_md5map(path, m):
    json.dump(m, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


# ---------- 批量驱动 ----------
def run_ingest(jobs, *, chunk_one, ckpt_path, md5map_path,
               content_md5=None, do_describe=True, force=False,
               label="文档", extra_text_fields=(), with_images=True,
               describe_arg=True):
    """通用批量入库。

    jobs: 已排序的任务列表,每项第一个元素必须是 stem,其余原样传给 chunk_one。
    chunk_one(job_tuple, do_describe) -> (text_chunks, image_chunks)。
    content_md5(job_tuple) -> (md5|None, source_label):内容指纹用于跨副本去重;
        返回 None 表示不去重。
    describe_arg: 调用 chunk_one 时是否传 do_describe 关键字(mp4 描述已在清洗
        阶段生成,切块签名不接受该参数,置 False)。
    返回 (done_count, dup_count)。
    """
    global TE, IE
    client = get_client()
    ensure_collections(client)
    import embed
    TE = TE or embed.get_text_encoder()
    if with_images:
        IE = IE or embed.get_image_encoder()

    done = load_json_set(ckpt_path)
    md5map = load_md5map(md5map_path)
    stems = {job[0] for job in jobs}
    skipped_done = len(done & stems)
    print(f"待入库 {label}: {len(jobs)}  (已入库跳过 {skipped_done})")

    n_dup = 0
    for i, job in enumerate(jobs, 1):
        stem = job[0]
        if stem in done and not force:
            continue
        cm = None
        if content_md5 is not None:
            cm, _src = content_md5(job)
        if cm and not force:
            kept = md5map.get(cm)
            if kept and kept != stem:
                n_dup += 1
                done.add(stem)
                save_json_set(ckpt_path, done)
                print(f"[{i}/{len(jobs)}] DUP {stem[:50]:50s} "
                      f"与已入库的 {kept[:40]} 内容相同,跳过")
                continue
        t0 = time.time()
        try:
            if describe_arg:
                tc, ic = chunk_one(job, do_describe=do_describe)
            else:
                tc, ic = chunk_one(job)
            # 先切块/编码成功再删旧:避免切块失败导致库中该文档彻底消失
            delete_by_source_stem(client, stem)
            if tc:
                batch_upsert(client, C.TEXT_COLLECTION,
                             text_points(tc, TE, extra_text_fields))
            if ic and with_images:
                batch_upsert(client, C.IMAGE_COLLECTION,
                             image_points(ic, IE, TE))
            done.add(stem)
            if cm:
                md5map[cm] = stem
                save_md5map(md5map_path, md5map)
            save_json_set(ckpt_path, done)
            print(f"[{i}/{len(jobs)}] OK {stem[:50]:50s} "
                  f"text={len(tc):4d} img={len(ic):3d}  {time.time()-t0:.0f}s")
        except Exception as e:
            print(f"[{i}/{len(jobs)}] FAIL {stem[:50]:50s} {str(e)[:120]}")
    print(f"完成。累计入库 {len(done)} 个{label},本次跳过重复副本 {n_dup} 个")
    return len(done), n_dup
