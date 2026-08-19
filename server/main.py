# -*- coding: utf-8 -*-
"""半导体知识助手后端 -- FastAPI HTTP 壳。

职责仅限 HTTP 层:
  GET  /          存活探针(路由在 health/router.py)
  GET  /health    健康检查(路由在 health/router.py,逻辑在 health/service.py)
  POST /api/chat  聊天接口(路由在 chat/router.py,逻辑在 chat/service.py)

安全:
  - 生产环境(ENV!=dev)自动关闭 /docs /redoc /openapi.json,避免暴露 API 结构
  - host 默认 127.0.0.1(仅本机);部署到服务器时设 HOST=0.0.0.0
  - CORS 默认允许 localhost 前端;生产环境用 CORS_ORIGINS 环境变量配置
  - /api/chat 异常不向前端泄露堆栈,仅返回通用错误提示

环境变量(可选):
  ENV            dev(默认) / prod;prod 时关闭文档端点
  HOST           监听地址,默认 127.0.0.1(生产部署设 0.0.0.0)
  PORT           监听端口,默认 8001
  CORS_ORIGINS   逗号分隔的允许源,默认 "http://localhost:3000,http://127.0.0.1:3000"
"""
import os
import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import uvicorn

from chat.router import router as chat_router
from health.router import router as health_router
from auth.router import router as auth_router
from support.metrics import metrics
from chat.conversation.router import router as conversation_router
from admin.router import router as admin_router
from chat.conversation.db import init_table as init_conv_table
from support import retriever_warmup
import config as C
# ==================== 环境配置 ====================
# ENV=prod 时关闭文档端点、收紧 CORS;开发时 ENV=dev(默认)保留 /docs 方便调试
ENV    = os.getenv("ENV", "dev").lower()
IS_DEV = ENV == "dev"

# ==================== 结构化日志 ====================
# 必须先于任何 logger 使用(如下方 JWT_SECRET 检查)
logging.basicConfig(
    level=logging.DEBUG if IS_DEV else logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("agent.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("main")

# ---- JWT_SECRET 安全检查 ----
if C.JWT_SECRET == "change-me-in-production":
    if IS_DEV:
        logger.warning("⚠️ JWT_SECRET 未设置,正在使用默认值! 请在 env.env 中设置 JWT_SECRET")
    else:
        raise RuntimeError("生产环境(ENV=prod)必须设置 JWT_SECRET!")
HOST   = os.getenv("HOST", "127.0.0.1")     # 默认仅本机;部署服务器设 0.0.0.0
PORT   = int(os.getenv("PORT", "8001"))

# ==================== FastAPI 应用 ====================
# 生产环境关闭文档端点(/docs /redoc /openapi.json),避免暴露 API 结构
app = FastAPI(
    title="半导体知识助手",
    docs_url="/docs" if IS_DEV else None,
    redoc_url="/redoc" if IS_DEV else None,
    openapi_url="/openapi.json" if IS_DEV else None,
)

# ==================== CORS ====================
# 开发:允许 localhost 前端;生产:通过 CORS_ORIGINS 环境变量配置允许的源
# 注:前端经 Next.js rewrites 代理走同源,不触发 CORS;此中间件为直接访问后端的场景兜底
_default_origins = "http://localhost:3000,http://127.0.0.1:3000"
_cors_raw = os.getenv("CORS_ORIGINS", _default_origins)
_cors_origins = [o.strip() for o in _cors_raw.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ==================== 请求体大小限制 ====================
# 拒绝超大请求(>100KB),防止内存耗尽攻击
MAX_REQUEST_BYTES = 500_000  # 500KB(会话同步需要)

@app.middleware("http")
async def limit_request_size(request: Request, call_next):
    if request.method == "POST" and not request.url.path.startswith("/api/admin/upload"):
        cl = request.headers.get("content-length")
        try:
            cl_val = int(cl) if cl else 0
        except ValueError:
            cl_val = 0  # 畸形 Content-Length 交给下层框架处理,此处不拦截
        if cl_val > MAX_REQUEST_BYTES:
            return JSONResponse(
                {"error": f"请求体过大(上限 {MAX_REQUEST_BYTES // 1024}KB)"},
                status_code=413,
            )
    return await call_next(request)


# ==================== 校验错误处理 ====================
# Pydantic 422 -> 400,提取第一条错误信息返回给前端
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    if errors:
        msg = errors[0].get("msg", "输入格式错误")
        # 去掉 Pydantic 前缀 "Value error, "
        if msg.startswith("Value error, "):
            msg = msg[len("Value error, "):]
    else:
        msg = "输入格式错误"
    return JSONResponse({"error": msg}, status_code=400)

# 注册功能路由
app.include_router(chat_router, prefix="/api")
app.include_router(health_router)
app.include_router(auth_router, prefix="/api/auth")
app.include_router(conversation_router, prefix="/api/conversations")
app.include_router(admin_router, prefix="/api/admin")

# 初始化会话表
init_conv_table()

# ==================== Metrics 端点 ====================
@app.get("/metrics")
def get_metrics():
    """返回运行指标(JSON)。无需认证,内部监控用。"""
    return metrics.get_stats()

# ==================== 启动 ====================
if __name__ == "__main__":
    print("半导体知识助手后端  http://{}:{}  (ReAct 流式, env={})".format(HOST, PORT, ENV), flush=True)
    # 启动时后台预热文本检索模型(BGE-m3),避免用户首次检索时多等约 30s。
    # daemon 线程:不阻塞 uvicorn 启动,服务立即可用;模型在后台加载,
    # 若用户在加载完成前提问,ensure_retriever 的锁会等其加载完(不会重复加载)。
    import threading
    def _warmup_retriever():
        try:
            print("[startup] 后台预热 BGE-m3 文本检索模型…", flush=True)
            retriever_warmup.ensure_retriever()
        except Exception as e:
            print("[startup] BGE-m3 预热失败: {}".format(e), flush=True)
    threading.Thread(target=_warmup_retriever, daemon=True, name="bge-warmup").start()

    # 工作记忆滚动 TTL:daemon 线程每日清理 30 天未活动会话的 checkpoint + 短期流水
    try:
        from memories.orchestration import start_prune_daemon
        start_prune_daemon()
        print("[startup] 工作记忆滚动 TTL 守护线程已启动(30 天)", flush=True)
    except Exception as e:
        print("[startup] TTL 守护启动失败: {}".format(e), flush=True)

    # 升迁补偿:进程退出时未落库的 daemon 升迁靠水位下次重试,但若该会话之后
    # 再无对话就永不升迁。启动时后台扫一遍待升迁 thread 补齐(不阻塞启动)。
    try:
        from memories.orchestration import start_recovery_daemon
        start_recovery_daemon()
        print("[startup] 升迁补偿扫描已在后台启动", flush=True)
    except Exception as e:
        print("[startup] 升迁补偿启动失败: {}".format(e), flush=True)

    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
