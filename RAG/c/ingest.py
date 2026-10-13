# -*- coding: utf-8 -*-
"""PPTX 入库:复用 DOCX 的切块引擎(content_list 契约与格式无关)。

PPT 清洗产物(``RAG/pptx/clean.py``)与 DOCX 同构:
``D:\\清洗文件\\pptx\\<镜像相对路径>\\<stem>\\auto\\<stem>_content_list.json``。
``_converted`` 仅存放 .ppt->.pptx 的转换源,不含 content_list,glob 自然不会命中。
因此直接复用 ``RAG/docx/chunker.py`` 的 ``discover(root=...)`` 与 ``chunk_one_docx``,
差异仅:

  - 清洗根为 ``D:\\清洗文件\\pptx``;
  - 源扩展名反推为 ``.pptx``(原始 .ppt 已转 .pptx,源 .ppt 仍在资料库,指纹回退 content_list);
  - 独立 checklist.json / _ingest_md5.json。

用法:
    python -m RAG.pptx.ingest                 # 入库全部已清洗 pptx
    python -m RAG.pptx.ingest --no-desc       # 跳过 LLM 图片描述(仅 CLIP)
    python -m RAG.pptx.ingest --force         # 强制重做(忽略断点)
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))   # RAG/pptx
_RAG = os.path.dirname(_HERE)                          # RAG
_PROJECT = os.path.dirname(_RAG)
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from RAG import ingest_core as core          # noqa: E402
from RAG.docx import chunker as docx_chunker  # noqa: E402

PPTX_OUT_ROOT = r"D:\清洗文件\pptx"
CKPT = os.path.join(_HERE, "checklist.json")
MD5MAP = os.path.join(_HERE, "_ingest_md5.json")


def _fix_src(stem, auto_dir):
    """discover 默认反推 .docx,这里改为 .pptx/.ppt 真实源路径(不存在则 None)。"""
    stem_dir = os.path.dirname(auto_dir)
    rel = os.path.relpath(os.path.dirname(stem_dir), PPTX_OUT_ROOT)
    for ext in (".pptx", ".ppt"):
        p = os.path.join(docx_chunker.common.SRC_ROOT, rel, stem + ext)
        if os.path.exists(p):
            return p
    return None


def discover_jobs(keyword=None):
    """返回 [(stem, auto_dir, source_path)];keyword 过滤 stem(调试用)。"""
    jobs = []
    for stem, auto_dir, _src_docx in docx_chunker.discover(PPTX_OUT_ROOT):
        jobs.append((stem, auto_dir, _fix_src(stem, auto_dir)))
    if keyword:
        k = keyword.lower()
        jobs = [j for j in jobs if k in j[0].lower()]
    return jobs


def content_md5(job):
    """内容指纹:源 .pptx 的 MD5(最稳);缺失时回退 content_list.json。"""
    stem, auto_dir, src_pptx = job
    if src_pptx and os.path.exists(src_pptx):
        m = core.md5_file(src_pptx)
        if m:
            return m, "src"
    cl = os.path.join(auto_dir, stem + "_content_list.json")
    m = core.md5_file(cl)
    if m:
        return m, "cl"
    return None, None


def chunk_one(job, do_describe=True):
    stem, auto_dir, _src = job
    return docx_chunker.chunk_one_docx(
        stem, auto_dir, _src, do_describe=do_describe)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = sys.argv[1:]
    keyword = args[0] if args else None
    do_describe = "--no-desc" not in flags
    force = "--force" in flags

    jobs = discover_jobs(keyword)
    print(f"入库根: {PPTX_OUT_ROOT}")
    print(f"LLM 图描述: {'开' if do_describe else '关(仅 CLIP)'}")
    if keyword:
        print(f"关键字过滤: {keyword} -> {len(jobs)} 篇")
    core.run_ingest(
        jobs, chunk_one=chunk_one, ckpt_path=CKPT, md5map_path=MD5MAP,
        content_md5=content_md5, do_describe=do_describe, force=force,
        label="PPTX", with_images=True, describe_arg=True)


if __name__ == "__main__":
    main()
