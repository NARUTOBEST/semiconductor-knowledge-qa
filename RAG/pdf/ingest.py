# -*- coding: utf-8 -*-
"""编排:chunk -> 嵌入(BGE-m3 文本 / CLIP 图)-> [可选]LLM 图描述 -> Qdrant 双库入库。

断点续跑:checklist.json 记已入库 stem,重跑跳过(--force 强制重做)。
LLM 图描述默认开但容错:单图失败记空描述、不中断;ep-xxx 未就绪时用 --no-desc 跳过。
"""
import os, sys, json, uuid, time, hashlib

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from qdrant_client import QdrantClient, models

_HERE = os.path.dirname(os.path.abspath(__file__))            # RAG/pdf
_RAG = os.path.dirname(_HERE)                                  # RAG
_PROJECT = os.path.dirname(_RAG)                               # 项目根
_CONFIG = os.path.join(_PROJECT, "config")
for _p in (_HERE, _RAG, _CONFIG):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import config as C
import chunker, embed, ark_client

_PDF_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(_PDF_DIR, "checklist.json")
# 内容去重:源 PDF 的 MD5 -> 首个入库时使用的 stem。原资料库里有约 110 份逐字节相同的
# 副本散落在不同分类目录(同内容常不同名),按 MD5 跳过可避免重复块入库。
MD5MAP = os.path.join(_PDF_DIR, "_ingest_md5.json")
TE = None   # 懒加载,进程内复用
IE = None


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


def load_md5map():
    if os.path.exists(MD5MAP):
        try:
            return json.load(open(MD5MAP, encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_md5map(m):
    json.dump(m, open(MD5MAP, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


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


def content_md5(auto_dir, stem, src_pdf):
    """内容身份键:优先用源 PDF 的 MD5(最稳);源 PDF 缺失时回退到 content_list.json。

    返回 (md5, 来源)。无法计算返回 (None, None)——此时不去重,按原逻辑入库。
    """
    if src_pdf and os.path.exists(src_pdf):
        m = md5_file(src_pdf)
        if m:
            return m, "src"
    cl = os.path.join(auto_dir, stem + "_content_list.json")
    m = md5_file(cl)
    if m:
        return m, "cl"
    return None, None


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
    md5map = load_md5map()
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
    stems = {p[0] for p in pdfs}
    skipped_done = len(done & stems)
    print(f"待入库 PDF: {len(pdfs)}  (已入库跳过 {skipped_done})")
    n_dup = 0
    for i, (stem, auto, _) in enumerate(pdfs, 1):
        if stem in done and not force:
            continue
        # 源 PDF 路径:由 auto 反推镜像到 src_root
        rel = os.path.relpath(os.path.dirname(auto), C.CLEAN_ROOT)
        src_pdf = os.path.join(src_root, rel, stem + ".pdf")
        # 内容去重:同 MD5 已以另一个 stem 入库 -> 跳过,不产生重复块
        cm, _src = content_md5(auto, stem, src_pdf)
        if cm and not force:
            kept = md5map.get(cm)
            if kept and kept != stem:
                n_dup += 1
                done.add(stem)  # 记为已处理,重跑不再重复提示
                save_ckpt(done)
                print(f"[{i}/{len(pdfs)}] DUP {stem[:50]:50s} 与已入库的 {kept[:40]} 内容相同,跳过")
                continue
        t0 = time.time()
        try:
            nt, ni = process_pdf(stem, auto, src_pdf, client, do_describe=do_describe)
            done.add(stem)
            if cm:
                md5map[cm] = stem  # 记录首个入库此内容的 stem
                save_md5map(md5map)
            save_ckpt(done)
            print(f"[{i}/{len(pdfs)}] OK {stem[:50]:50s} text={nt:4d} img={ni:3d}  {time.time()-t0:.0f}s")
        except Exception as e:
            print(f"[{i}/{len(pdfs)}] FAIL {stem[:50]:50s} {str(e)[:120]}")
    print(f"完成。累计入库 {len(done)} 个 PDF,本次跳过重复副本 {n_dup} 个")


if __name__ == "__main__":
    # 默认全量入库整个清洗产物根;也可命令行传入子目录:python ingest.py "<子目录>"
    CLEAN_SUB = sys.argv[1] if (len(sys.argv) > 1 and not sys.argv[1].startswith("--")) else C.CLEAN_ROOT
    SRC_ROOT  = C.SRC_ROOT   # 资料库根,可用环境变量 SRC_ROOT 覆盖(见 config.py)
    do_describe = "--no-desc" not in sys.argv
    force = "--force" in sys.argv
    print(f"入库目录: {CLEAN_SUB}")
    print(f"LLM 图描述: {'开' if do_describe else '关(仅 CLIP)'}")
    ingest_dir(CLEAN_SUB, SRC_ROOT, do_describe=do_describe, force=force)
