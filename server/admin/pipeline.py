# -*- coding: utf-8 -*-
"""文档处理流水线:MinerU 清洗 -> 切块 -> 嵌入 -> 入库。后台线程执行。"""
import os
import sys
import time
import uuid
import subprocess
import threading
import logging

logger = logging.getLogger("admin")

_HERE = os.path.dirname(os.path.abspath(__file__))            # server/admin/
_PROJECT = os.path.dirname(os.path.dirname(_HERE))             # 项目根(server/ 再上一层)
_RAG = os.path.join(_PROJECT, "RAG")
_CONFIG = os.path.join(_PROJECT, "config")
for _p in (_PROJECT, _RAG, _CONFIG):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config as C

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

MINERU = os.path.join(_PROJECT, ".venv_mineru", "Scripts", "mineru.exe")

# 任务状态存储(内存,单进程)
_tasks = {}
_lock = threading.Lock()


def start_pipeline(pdf_path, filename):
    """启动后台流水线,返回 task_id。"""
    task_id = str(uuid.uuid4())
    with _lock:
        _tasks[task_id] = {
            "task_id": task_id,
            "filename": filename,
            "status": "pending",
            "message": "排队中...",
            "started_at": time.time(),
        }
    t = threading.Thread(
        target=_run_pipeline, args=(task_id, pdf_path, filename),
        daemon=True, name=f"pipeline-{task_id}")
    t.start()
    return task_id


def get_task(task_id):
    with _lock:
        return _tasks.get(task_id)


def list_tasks():
    with _lock:
        return list(_tasks.values())


def _update(task_id, status, message):
    with _lock:
        if task_id in _tasks:
            _tasks[task_id]["status"] = status
            _tasks[task_id]["message"] = message


def _run_pipeline(task_id, pdf_path, filename):
    """后台执行:清洗 -> 切块 -> 入库。"""
    try:
        import config as C
        stem = os.path.splitext(filename)[0]

        # ---- Step 1: MinerU 清洗 ----
        _update(task_id, "cleaning", "MinerU 清洗中(可能需要几分钟)...")
        out_parent = C.CLEAN_ROOT
        os.makedirs(out_parent, exist_ok=True)

        cmd = [MINERU, "-p", pdf_path, "-o", out_parent, "-b", "pipeline"]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=1800,
            env={**os.environ, "NO_PROXY": "127.0.0.1,localhost"})
        if result.returncode != 0:
            raise RuntimeError(f"MinerU 失败: {(result.stderr or '')[-200:]}")

        auto_dir = os.path.join(out_parent, stem, "auto")
        if not os.path.exists(os.path.join(auto_dir, stem + "_content_list.json")):
            raise RuntimeError("MinerU 未生成 content_list.json,请检查 PDF 是否有效")

        # ---- Step 2: 切块 + 嵌入 + 入库 ----
        _update(task_id, "ingesting", "切块 + 嵌入 + 入库中...")

        import ingest
        import embed

        client = ingest.get_client()
        ingest.ensure_collections(client)
        if ingest.TE is None:
            ingest.TE = embed.get_text_encoder()
        if ingest.IE is None:
            ingest.IE = embed.get_image_encoder()

        src_pdf = os.path.join(C.SRC_ROOT, filename)
        nt, ni = ingest.process_pdf(
            stem, auto_dir, src_pdf, client, do_describe=True)

        _update(task_id, "done", f"完成: 文本块 {nt}, 图像块 {ni}")

    except subprocess.TimeoutExpired:
        _update(task_id, "error", "MinerU 超时(>30分钟)")
    except Exception as e:
        logger.exception(f"Pipeline error: {e}")
        _update(task_id, "error", str(e)[:200])
