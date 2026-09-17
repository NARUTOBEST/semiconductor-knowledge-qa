# -*- coding: utf-8 -*-
"""文档处理流水线:MinerU 清洗 -> 切块 -> 嵌入 -> 入库。后台线程执行。"""
import json
import os
import sys
import time
import uuid
import shutil
import subprocess
import threading
import logging

logger = logging.getLogger("admin")

_HERE = os.path.dirname(os.path.abspath(__file__))            # server/admin/
_PROJECT = os.path.dirname(os.path.dirname(_HERE))             # 项目根(server/ 再上一层)
_RAG = os.path.join(_PROJECT, "RAG")
_RAG_PDF = os.path.join(_RAG, "pdf")                            # PDF 入库脚本(chunker/ingest)
_CONFIG = os.path.join(_PROJECT, "config")
for _p in (_PROJECT, _RAG, _RAG_PDF, _CONFIG):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config as C

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# MinerU 可执行文件解析(跨平台):
#   1) 环境变量 MINERU_BIN 显式指定;
#   2) Windows 本地开发:项目内 .venv_mineru/Scripts/mineru.exe;
#   3) PATH 上的 mineru(如 Linux 全局安装)。
# Docker 部署不内置 MinerU(清洗在本地进行),解析不到时由 _run_pipeline 给出明确提示。
def _resolve_mineru():
    env_bin = os.environ.get("MINERU_BIN", "").strip()
    if env_bin and os.path.exists(env_bin):
        return env_bin
    win_venv = os.path.join(_PROJECT, ".venv_mineru", "Scripts", "mineru.exe")
    if os.path.exists(win_venv):
        return win_venv
    return shutil.which("mineru")


MINERU = _resolve_mineru()

# 任务状态存储:默认外置 Redis(多 worker 一致、应用重启不丢;TTL 7 天自动清理),
# Redis 不可用时回退进程内存(=外置前行为:单进程、重启清零)。见 support.state_store。
_TASK_TTL_S = int(os.getenv("ADMIN_TASK_TTL_DAYS", "7")) * 86400
_R_KEY = "admin:task:{}"          # JSON 串
_R_INDEX = "admin:task:index"     # ZSET:member=task_id, score=started_at(列表排序用)

# 入库轮询(wait=False 异步模式):间隔与总超时
_INGEST_POLL_INTERVAL_S = float(os.getenv("ADMIN_INGEST_POLL_INTERVAL_S", "3"))
_INGEST_POLL_TIMEOUT_S = float(os.getenv("ADMIN_INGEST_POLL_TIMEOUT_S", "7200"))

# 内存回退存储(monkeypatch 友好)
_tasks = {}
_lock = threading.Lock()


def _state_redis():
    from support import state_store
    return state_store.get_state_redis()


def _save_redis(r, task):
    p = r.pipeline(transaction=False)
    p.set(_R_KEY.format(task["task_id"]),
          json.dumps(task, ensure_ascii=False), ex=_TASK_TTL_S)
    p.zadd(_R_INDEX, {task["task_id"]: task["started_at"]})
    p.expire(_R_INDEX, _TASK_TTL_S)
    p.execute()


def start_pipeline(pdf_path, filename):
    """启动后台流水线,返回 task_id。"""
    task_id = str(uuid.uuid4())
    task = {
        "task_id": task_id,
        "filename": filename,
        "status": "pending",
        "message": "排队中...",
        "started_at": time.time(),
    }
    r = _state_redis()
    if r is not None:
        try:
            _save_redis(r, task)
        except Exception:
            from support import state_store
            state_store.note_fail()
            r = None
    if r is None:
        with _lock:
            _tasks[task_id] = task
    t = threading.Thread(
        target=_run_pipeline, args=(task_id, pdf_path, filename),
        daemon=True, name=f"pipeline-{task_id}")
    t.start()
    return task_id


def get_task(task_id):
    r = _state_redis()
    if r is not None:
        try:
            raw = r.get(_R_KEY.format(task_id))
            return json.loads(raw) if raw else None
        except Exception:
            from support import state_store
            state_store.note_fail()
    with _lock:
        return _tasks.get(task_id)


def list_tasks():
    r = _state_redis()
    if r is not None:
        try:
            # 按 started_at 升序(与内存 dict 插入序一致);TTL 过期的任务键跳过,
            # 顺带把索引里的悬挂条目清掉。
            now = time.time()
            r.zremrangebyscore(_R_INDEX, "-inf", now - _TASK_TTL_S)
            ids = r.zrange(_R_INDEX, 0, -1)
            out = []
            for tid in ids:
                raw = r.get(_R_KEY.format(tid))
                if raw:
                    out.append(json.loads(raw))
                else:
                    r.zrem(_R_INDEX, tid)
            return out
        except Exception:
            from support import state_store
            state_store.note_fail()
    with _lock:
        return list(_tasks.values())


def _update(task_id, status, message):
    r = _state_redis()
    if r is not None:
        try:
            raw = r.get(_R_KEY.format(task_id))
            if not raw:
                return
            task = json.loads(raw)
            task["status"] = status
            task["message"] = message
            _save_redis(r, task)
            return
        except Exception:
            from support import state_store
            state_store.note_fail()
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
        # Docker/服务器未内置 MinerU(清洗是本地工具);直接给出可操作提示而非崩溃。
        mineru_bin = _resolve_mineru()
        if not mineru_bin:
            _update(task_id, "error",
                    "本服务未配置 MinerU(PDF 清洗在本地进行)。请在本地用 MinerU 清洗后, "
                    "运行入库脚本并设 QDRANT_URL 指向服务器导入, 详见 deploy/README。")
            return
        _update(task_id, "cleaning", "MinerU 清洗中(可能需要几分钟)...")
        out_parent = C.CLEAN_ROOT
        os.makedirs(out_parent, exist_ok=True)

        cmd = [mineru_bin, "-p", pdf_path, "-o", out_parent, "-b", "pipeline"]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=1800,
            env={**os.environ, "NO_PROXY": "127.0.0.1,localhost"})
        if result.returncode != 0:
            raise RuntimeError(f"MinerU 失败: {(result.stderr or '')[-200:]}")

        auto_dir = os.path.join(out_parent, stem, "auto")
        if not os.path.exists(os.path.join(auto_dir, stem + "_content_list.json")):
            raise RuntimeError("MinerU 未生成 content_list.json,请检查 PDF 是否有效")

        # ---- Step 2: 切块 + 嵌入 + 入库(全部在检索微服务进程内执行)----
        # 主服务不加载 torch/模型、不打开 Qdrant 本地库(文件锁归微服务),
        # 切块/嵌入/入库经 /ingest_document 交给持有模型与向量库的微服务。
        # wait=False 异步模式:提交后轮询 /ingest_status,HTTP 不再长挂 1 小时
        # (嵌入让路在线检索后总耗时变长,长连接易超时);旧版微服务不认识 wait
        # 字段时会同步返回计数,自动兼容。
        _update(task_id, "ingesting", "切块 + 嵌入 + 入库中...")

        import httpx
        base = getattr(C, "RETRIEVAL_SERVICE_URL",
                       "http://127.0.0.1:8002").rstrip("/")
        tok = getattr(C, "RETRIEVAL_INTERNAL_TOKEN", "")   # 服务间鉴权
        headers = {"X-Internal-Token": tok} if tok else None
        src_pdf = os.path.join(C.SRC_ROOT, filename)
        resp = httpx.post(
            base + "/ingest_document",
            json={"stem": stem, "auto_dir": auto_dir,
                  "source_path": src_pdf, "do_describe": True, "wait": False},
            headers=headers,
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("error"):
            raise RuntimeError(str(data["error"])[:200])

        if data.get("task_id"):
            # 异步模式:轮询微服务任务态直到 done/error
            ingest_tid = data["task_id"]
            deadline = time.time() + _INGEST_POLL_TIMEOUT_S
            while True:
                if time.time() >= deadline:
                    raise RuntimeError("入库超时(>2小时),请检查检索服务日志")
                time.sleep(_INGEST_POLL_INTERVAL_S)
                try:
                    st = httpx.get(f"{base}/ingest_status/{ingest_tid}",
                                   headers=headers, timeout=30).json()
                except Exception:
                    continue    # 微服务繁忙/瞬时不可达:继续轮询,由 deadline 兜底
                status = st.get("status")
                if status == "done":
                    nt, ni = st.get("text_count", 0), st.get("image_count", 0)
                    break
                if status == "error":
                    raise RuntimeError(str(st.get("error", "入库失败"))[:200])
        else:
            # 同步兼容(旧版微服务直接返回计数)
            nt, ni = data.get("text_count", 0), data.get("image_count", 0)

        _update(task_id, "done", f"完成: 文本块 {nt}, 图像块 {ni}")

    except subprocess.TimeoutExpired:
        _update(task_id, "error", "MinerU 超时(>30分钟)")
    except Exception as e:
        logger.exception(f"Pipeline error: {e}")
        _update(task_id, "error", str(e)[:200])
