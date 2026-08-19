# -*- coding: utf-8 -*-
r"""把切分器产出导出成文件:每个 PDF 生成 <stem>_chunks.json + <stem>_chunks.jsonl,
镜像 清洗产物 的目录结构(<rel>\<stem>)放到 分块产物\ 下。

切分参数与 ingest --no-desc 一致(llm_split=None,超长块保留),产物与已入库的块一致。
每个块带 chunk_type(text/image);文本块含 content/heading_path/embed_text 等,
图像块含 image_path/description/caption/parent_text_chunk_id 等。
"""
import os, sys, json
_HERE = os.path.dirname(os.path.abspath(__file__))
_RAG = os.path.dirname(_HERE)
_PROJECT = os.path.dirname(_RAG)
_CONFIG = os.path.join(_PROJECT, "config")
for _p in (_CONFIG, _RAG):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import config as C
import chunker

CLEAN_SUB = r"D:\清洗文件\0001 半导体设备资料合集\ALD原子沉积资料\ALD原子沉积资料"
OUT_ROOT  = r"C:\project3\分块产物"


def _long(p):
    r"""Windows 长路径(>200)加 \\?\ 前缀,避免写不进去。"""
    p = os.path.abspath(p)
    if len(p) > 200 and not p.startswith("\\\\?\\"):
        return "\\\\?\\" + p
    return p


def export_one(stem, auto_dir, src_pdf):
    # do_describe=False + llm_split=None:与 ingest --no-desc 的切分参数保持一致
    # (文件头声明产物与已入库的块一致;否则会对无缓存图片真实调用 LLM 图描述)
    tc, ic = chunker.chunk_one_pdf(stem, auto_dir, src_pdf,
                                   target=C.CHUNK_TARGET, max_size=C.CHUNK_MAX,
                                   min_size=C.CHUNK_MIN, do_describe=False,
                                   llm_split=None)
    chunks = [{"chunk_type": "text", **t} for t in tc] + \
             [{"chunk_type": "image", **i} for i in ic]
    # 镜像: CLEAN_ROOT\<rel>\<stem>\auto  ->  OUT_ROOT\<rel>\<stem>
    rel = os.path.relpath(os.path.dirname(auto_dir), C.CLEAN_ROOT)
    out_dir = os.path.join(OUT_ROOT, rel)
    os.makedirs(_long(out_dir), exist_ok=True)
    jpath = _long(os.path.join(out_dir, f"{stem}_chunks.json"))
    lpath = _long(os.path.join(out_dir, f"{stem}_chunks.jsonl"))
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    with open(lpath, "w", encoding="utf-8") as f:
        for ch in chunks:
            f.write(json.dumps(ch, ensure_ascii=False) + "\n")
    return len(tc), len(ic)


def main():
    clean_sub = sys.argv[1] if len(sys.argv) > 1 else CLEAN_SUB
    pdfs = []
    for dp, dn, fn in os.walk(clean_sub):
        if os.path.basename(dp) != "auto":
            continue
        for f in fn:
            if f.endswith("_content_list.json") and "_v2" not in f:
                stem = f[:-len("_content_list.json")]
                pdfs.append((stem, dp, os.path.dirname(dp)))
    pdfs.sort()
    print(f"待切块 PDF: {len(pdfs)}  ->  输出根: {OUT_ROOT}")
    gt = gi = 0
    for i, (stem, auto, _) in enumerate(pdfs, 1):
        rel = os.path.relpath(os.path.dirname(auto), C.CLEAN_ROOT)
        src_pdf = os.path.join(C.SRC_ROOT, rel, stem + ".pdf")
        try:
            nt, ni = export_one(stem, auto, src_pdf)
            gt += nt; gi += ni
            print(f"[{i}/{len(pdfs)}] {stem[:50]:50s} text={nt:4d} img={ni:3d}")
        except Exception as e:
            print(f"[{i}/{len(pdfs)}] FAIL {stem[:50]:50s} {str(e)[:100]}")
    print(f"\n完成: {len(pdfs)} PDF  |  文本块 {gt}  图像块 {gi}")


if __name__ == "__main__":
    main()
