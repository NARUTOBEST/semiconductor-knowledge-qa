# -*- coding: utf-8 -*-
"""一次性迁移脚本:把 Qdrant 引用的全部媒体文件(图片+视频)上传到阿里云 OSS。

背景:原火山 TOS 欠费停用(读写全挂),桶内旧对象不可读;改用阿里云 OSS 后
需要把知识库引用的媒体重新补齐。本脚本:

1. 读 _media_paths.json(Qdrant 两库 scroll 出的全部引用媒体路径,270,066 条);
2. 三级还原本地文件:
   a. 原路径直接命中;
   b. 根后补模态段(pdf/docx/pptx/xlsx/image/cad/mp4)命中(清洗目录重整遗留);
   c. 文件名为内容指纹(≥32 位 hex,同 hash=同内容)时,全盘 basename 反查;
3. 每个文件经 image_s3.abs_to_key 映射对象 key(含旧 PDF 补 pdf/ 段逻辑),
   与检索签名链路完全一致;
4. 并发上传到 OSS,checkpoint(_oss_uploaded.json)支持断点续跑。

用法:
  python _oss_media_upload.py plan      # 只统计还原结果,不上传
  python _oss_media_upload.py upload    # 还原 + 上传
"""
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "config"))
sys.path.insert(1, os.path.join(ROOT, "mcp_servers", "retrieval"))

import config as C  # noqa: E402
from image_s3 import get_s3, abs_to_key  # noqa: E402

MODALS = ["pdf", "docx", "pptx", "xlsx", "image", "cad", "mp4"]
HASH_RE = re.compile(r"^[0-9a-f]{32,64}$", re.I)  # 内容指纹文件名(去扩展名后)
CKPT = os.path.join(ROOT, "_oss_uploaded.json")
FAILED = os.path.join(ROOT, "_oss_failed.json")
PATHS = os.path.join(ROOT, "_media_paths.json")

EXT_CTYPE = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
    ".gif": "image/gif", ".bmp": "image/bmp", ".webp": "image/webp",
    ".svg": "image/svg+xml", ".mp4": "video/mp4",
}


def split_root(path):
    """D:\\清洗文件\\<rest> -> <rest>;否则 None(含相对路径 images/<hash>.jpg)。"""
    s = str(path).replace("\\", "/")
    m = re.match(r"^[A-Za-z]:/(.+)$", s)
    if not m:
        return None
    body = m.group(1)
    root_body = str(C.CLEAN_FILES_ROOT).replace("\\", "/").strip("/")
    parts, root_parts = body.split("/"), root_body.split("/")
    if [t.lower() for t in parts[:len(root_parts)]] != [t.lower() for t in root_parts]:
        return None
    return "/".join(parts[len(root_parts):])


def build_basename_index():
    """全盘 walk 清洗根,指纹文件名(hash.jpg) -> 本地绝对路径(仅唯一者)。"""
    idx = {}
    for dirpath, _dirs, files in os.walk(C.CLEAN_FILES_ROOT):
        for f in files:
            stem, ext = os.path.splitext(f)
            if HASH_RE.match(stem) and f.lower() not in idx:
                idx[f.lower()] = os.path.join(dirpath, f)
    return idx


def resolve(path, bn_idx):
    """引用路径 -> 本地文件绝对路径;还原失败返回 None。"""
    if os.path.isfile(path):
        return path
    rel = split_root(path)
    if rel is not None:
        # b. 补模态段
        toks = rel.split("/")
        for mod in MODALS:
            cand = os.path.join(C.CLEAN_FILES_ROOT, mod, *toks)
            if os.path.isfile(cand):
                return cand
        # c. 指纹 basename 反查(同 hash 同内容,目录不同无所谓)
        bn = toks[-1].lower()
        hit = bn_idx.get(bn)
        if hit and HASH_RE.match(os.path.splitext(bn)[0]):
            return hit
        return None
    # 相对路径(images/<hash>.jpg):只能 basename 反查
    bn = os.path.basename(str(path).replace("\\", "/")).lower()
    hit = bn_idx.get(bn)
    return hit if hit and HASH_RE.match(os.path.splitext(bn)[0]) else None


def ctype_of(path):
    return EXT_CTYPE.get(os.path.splitext(path)[1].lower(), "application/octet-stream")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "plan"
    paths = json.load(open(PATHS, encoding="utf-8"))
    print(f"referenced paths: {len(paths)}", flush=True)

    t0 = __import__("time").time()
    bn_idx = build_basename_index()
    print(f"basename index: {len(bn_idx)} hash-named files "
          f"({__import__('time').time() - t0:.1f}s)", flush=True)

    jobs = {}          # key -> local path(按对象 key 去重)
    miss = []
    for p in paths:
        local = resolve(p, bn_idx)
        if not local:
            miss.append(p)
            continue
        key = abs_to_key(local)
        if not key:
            continue
        jobs.setdefault(key, local)

    total = sum(os.path.getsize(lp) for lp in jobs.values() if os.path.isfile(lp))
    print(f"resolved: {len(jobs)} unique objects, {total / 1e9:.2f} GB; "
          f"unresolved: {len(miss)}", flush=True)

    if mode != "upload":
        return

    s3 = get_s3()
    bucket = C.TOS_BUCKET
    done = set()
    if os.path.isfile(CKPT):
        done = set(json.load(open(CKPT, encoding="utf-8")))
    todo = [(k, lp) for k, lp in jobs.items() if k not in done]
    print(f"todo: {len(todo)} (done in checkpoint: {len(done)})", flush=True)

    from boto3.s3.transfer import TransferConfig
    cfg = TransferConfig(multipart_threshold=32 * 1024 ** 2,
                         multipart_chunksize=32 * 1024 ** 2, max_concurrency=4)
    lock = threading.Lock()
    ok, fail = [], []
    stat = {"n": 0}

    def work(item):
        key, lp = item
        try:
            s3.upload_file(lp, bucket, key, Config=cfg,
                           ExtraArgs={"ContentType": ctype_of(lp)})
            return key, None
        except Exception as e:
            return key, f"{type(e).__name__}: {e}"

    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(work, it) for it in todo]
        for fu in as_completed(futs):
            key, err = fu.result()
            with lock:
                if err:
                    fail.append({"key": key, "err": err})
                else:
                    ok.append(key)
                stat["n"] += 1
                if stat["n"] % 500 == 0:
                    json.dump(sorted(done | set(ok)), open(CKPT, "w"))
                    gb = stat["n"] * 0  # 占位;体积汇总在结束时打
                    print(f"progress {stat['n']}/{len(todo)} fail={len(fail)}",
                          flush=True)

    json.dump(sorted(done | set(ok)), open(CKPT, "w"))
    json.dump(fail, open(FAILED, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"FINISHED ok={len(ok)} fail={len(fail)}", flush=True)


if __name__ == "__main__":
    main()
