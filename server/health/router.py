# -*- coding: utf-8 -*-
"""监控端点路由:GET / (存活探针)+ GET /health (健康检查)。"""
import os

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from health.service import check_qdrant, check_llm

router = APIRouter()


@router.get("/")
def root():
    """存活探针(轻量,不检查依赖)。"""
    return {"msg": "ok", "service": "semi-agent-backend", "env": os.getenv("ENV", "dev")}


@router.get("/health")
def health():
    """健康检查:探测关键依赖(Qdrant / LLM)是否可用。"""
    checks = {"api": "ok", "qdrant": check_qdrant(), "llm": check_llm()}
    all_ok = all(v == "ok" for v in checks.values())
    return JSONResponse(checks, status_code=200 if all_ok else 503)
