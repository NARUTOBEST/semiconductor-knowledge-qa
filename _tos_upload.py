# -*- coding: utf-8 -*-
"""一次性:把 D:\\清洗文件 下的图片上传到火山引擎 TOS 私有桶。

用法(先配好 TOS_* 环境变量,见 env/env.example):
    .venv_mineru\\Scripts\\python.exe _tos_upload.py            # 上传(断点续传)
    .venv_mineru\\Scripts\\python.exe _tos_upload.py --dry-run  # 只统计不上传
    .venv_mineru\\Scripts\\python.exe _tos_upload.py --workers 16

对象 key 规则与检索端一致(mcp_servers/retrieval/image_s3.py:abs_to_key):
    <TOS_KEY_PREFIX>/<相对 CLEAN_FILES_ROOT 的路径,正斜杠>
已存在的 key 自动跳过(head_object 校验),中断后重跑即可续传。
"""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join("mcp_servers", "retrieval"))
sys.path.insert(0, "config")
import config as C          # noqa: E402
import image_s3             # noqa: E402

IMG_EXT = (".jpg", ".jpeg", ".png", ".webp", ".gif")


def iter_images(root):
    """产出 (abs_path, key),仅图片文件且能映射到 key。"""
    for dp, _dn, fns in os.walk(root):
        for fn in fns:
            if fn.lower().endswith(IMG_EXT):
                ap = os.path.join(dp, fn)
                key = image_s3.abs_to_key(ap)
                if key:
                    yield ap, key


def main():
    dry = "--dry-run" in sys.argv
    workers = 16
    if "--workers" in sys.argv:
        workers = int(sys.argv[sys.argv.index("--workers") + 1])

    if not C.TOS_ENABLED:
        print("[X] 未配置 TOS_* 环境变量(TOS_ENDPOINT/TOS_REGION/TOS_BUCKET/"
              "TOS_ACCESS_KEY/TOS_SECRET_KEY),无法上传。")
        return 1
    s3 = image_s3.get_s3()
    root = os.path.normpath(C.CLEAN_FILES_ROOT)
    print(f"[i] 根目录 : {root}")
    print(f"[i] 桶/端点: {C.TOS_BUCKET} @ {C.TOS_ENDPOINT} (region={C.TOS_REGION})")
    print(f"[i] key前缀: {C.TOS_KEY_PREFIX or '(无)'}")

    jobs = list(iter_images(root))
    print(f"[i] 发现图片 {len(jobs)} 张,线程 {workers}{' [DRY-RUN]' if dry else ''}")
    if dry:
        return 0

    # 先统计已存在(断点续传):逐个 head 太慢,用已上传清单缓存
    state_file = "_tos_uploaded.txt"
    done = set()
    if os.path.exists(state_file):
        with open(state_file, "r", encoding="utf-8") as f:
            done = {ln.strip() for ln in f if ln.strip()}

    todo = [(ap, k) for ap, k in jobs if k not in done]
    print(f"[i] 已上传 {len(done)},待传 {len(todo)}")

    n_ok = n_skip = n_err = 0
    t0 = time.time()
    lock = __import__("threading").Lock()
    sf = open(state_file, "a", encoding="utf-8")

    def upload(item):
        ap, key = item
        try:
            s3.upload_file(ap, C.TOS_BUCKET, key)
            return key, None
        except Exception as e:  # noqa: BLE001
            return key, str(e)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(upload, it): it for it in todo}
        for i, fu in enumerate(as_completed(futs), 1):
            key, err = fu.result()
            with lock:
                if err:
                    n_err += 1
                    if n_err <= 20:
                        print(f"  [ERR] {key}: {err}")
                else:
                    n_ok += 1
                    sf.write(key + "\n")
                    sf.flush()
            if i % 500 == 0:
                rate = i / max(time.time() - t0, 1e-6)
                print(f"  进度 {i}/{len(todo)}  成功{n_ok} 失败{n_err}  {rate:.1f}张/秒")
    sf.close()
    dt = time.time() - t0
    print(f"[完成] 新传 {n_ok}  失败 {n_err}  用时 {dt/60:.1f} 分钟")
    if n_err:
        print("[i] 有失败,重跑本脚本即可续传(仅重试失败与未传)。")
    return 0 if n_err == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
