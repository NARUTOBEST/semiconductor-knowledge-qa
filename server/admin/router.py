# -*- coding: utf-8 -*-
"""管理员路由:文档上传 + 状态查询。仅 admin 角色可访问。"""
import os
import logging
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException

from auth.deps import get_current_user
from admin.pipeline import start_pipeline, get_task, list_tasks

logger = logging.getLogger("admin")
router = APIRouter()

# 上传保存目录必须与 config.SRC_ROOT(流水线入库的源目录)一致:
# server/admin/router.py -> 三层 dirname 才是项目根
_SRC_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "180-半导体设备相关资料！")
_MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # 200MB


def _require_admin(user):
    if user["role"] != "admin":
        raise HTTPException(403, "仅管理员可执行此操作")


@router.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    user=Depends(get_current_user),
):
    """上传 PDF,自动清洗+切块+入库。仅 admin。"""
    _require_admin(user)

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "仅支持 PDF 文件")

    # 安全:防止路径穿越
    safe_name = os.path.basename(file.filename)
    if safe_name != file.filename:
        raise HTTPException(400, "文件名非法")

    # 分块流式写盘,边写边计数,避免整个文件读入内存后才校验大小
    os.makedirs(_SRC_ROOT, exist_ok=True)
    save_path = os.path.join(_SRC_ROOT, safe_name)
    size = 0
    try:
        with open(save_path, "wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)  # 1MB
                if not chunk:
                    break
                size += len(chunk)
                if size > _MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "文件过大(上限 200MB)")
                f.write(chunk)
    except Exception:
        if os.path.exists(save_path):
            os.remove(save_path)
        raise
    if size == 0:
        if os.path.exists(save_path):
            os.remove(save_path)
        raise HTTPException(400, "文件为空")

    # 启动后台流水线
    task_id = start_pipeline(save_path, safe_name)
    logger.info(f"上传: {safe_name} -> task {task_id} (by {user['username']})")

    return {"task_id": task_id, "filename": safe_name}


@router.get("/upload/status/{task_id}")
def upload_status(task_id: str, user=Depends(get_current_user)):
    """查询单个任务状态。仅 admin。"""
    _require_admin(user)
    task = get_task(task_id)
    if not task:
        raise HTTPException(404, "任务不存在")
    return task


@router.get("/upload/tasks")
def upload_tasks(user=Depends(get_current_user)):
    """列出所有上传任务。仅 admin。"""
    _require_admin(user)
    return list_tasks()
