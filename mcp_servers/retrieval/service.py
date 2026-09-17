# -*- coding: utf-8 -*-
"""检索服务:独立进程,BGE-m3 + CLIP + Reranker + Qdrant。

双通道对外:
  - MCP 工具接口(挂载在 /mcp):agent 经 tools/mcp_bridge.py 以 MCP 客户端取回
    search_text / search_image / get_chunk 三件套(工具声明见 tools.py);
  - 传统 HTTP 端点:/embed_text /rerank /ingest_document 是能力型端点
    (长期记忆嵌入、重排、管理后台上传流水线复用),不是 agent 工具,继续走 HTTP。

Agent 进程不加载任何检索模型;启动时后台预热,首次请求可能稍慢(等模型加载完)。
启动方式:
    python -m mcp_servers.retrieval.service
或(启动脚本)直接以脚本路径运行。
"""
import os
import secrets
import sys
import threading
import logging
import uuid
from contextlib import asynccontextmanager

# ---- 路径设置:本文件位于 mcp_servers/retrieval/,项目根在上 2 级 ----
# 不得把 _HERE 插入 sys.path——目录里的 tools.py 会劫持 agent 侧的 ``import tools``;
# 包内模块(query/engine_api/tools)一律相对导入。
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(os.path.dirname(_HERE))
_RAG = os.path.join(_PROJECT, "RAG")
_CONFIG = os.path.join(_PROJECT, "config")
for _p in (_PROJECT, _RAG, _CONFIG):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# 以脚本路径直跑(启动脚本/Dockerfile CMD)时补包上下文,否则相对导入报错
if __package__ in (None, ""):
    import mcp_servers.retrieval  # noqa: F401  触发父包初始化
    __package__ = "mcp_servers.retrieval"

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("retrieval")

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel
from starlette.responses import JSONResponse
import uvicorn

import config as C
import embed
from . import engine_api
from . import query as Q
from .tools import mcp as retrieval_mcp


# ---- 服务间鉴权:RETRIEVAL_INTERNAL_TOKEN 非空时,所有端点(除 /health)校验
# X-Internal-Token 请求头;不一致或不携带 -> 403。留空 = 关闭(本地开发)。----
_TOKEN_HEADER = "x-internal-token"


def _token_ok(received: str) -> bool:
    """常量时间比较,避免时序侧信道。"""
    expected = C.RETRIEVAL_INTERNAL_TOKEN
    return (not expected) or secrets.compare_digest(received or "", expected)


def _require_internal_token(x_internal_token: str = Header(default="")):
    if not _token_ok(x_internal_token):
        raise HTTPException(status_code=403, detail="forbidden")


class _TokenGuardASGI:
    """包在 /mcp 挂载子应用外的 ASGI 中间件:同样校验内部 token。

    MCP 挂载是 Starlette 子应用,不走 FastAPI 依赖体系,只能包一层 ASGI。
    """

    def __init__(self, asgi_app):
        self._app = asgi_app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and not _token_ok(_header_value(scope, _TOKEN_HEADER)):
            resp = JSONResponse({"detail": "forbidden"}, status_code=403)
            await resp(scope, receive, send)
            return
        await self._app(scope, receive, send)


def _header_value(scope, name: str) -> str:
    for k, v in scope.get("headers") or []:
        if k.decode("latin-1").lower() == name:
            return v.decode("latin-1")
    return ""


# ---- MCP 工具接口:挂载到 FastAPI(/mcp)----
# 注意:必须先 streamable_http_app() 创建 session manager,再在父应用 lifespan
# 里 run() 它——挂载的子应用 lifespan 不会被父应用自动执行。
# transport_security:SDK 默认的 DNS rebinding 防护只放行 localhost/127.0.0.1 的 Host,
# 容器互联时 Host 为服务名(如 retrieval:8002)会被 421 拒掉;本服务只绑 127.0.0.1
# 且有 X-Internal-Token 门禁,这里关闭该防护。
from mcp.server.transport_security import TransportSecuritySettings

mcp_app = retrieval_mcp.streamable_http_app(
    streamable_http_path="/",
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)


@asynccontextmanager
async def _lifespan(app):
    async with retrieval_mcp.session_manager.run():
        yield


app = FastAPI(title="检索服务(MCP + HTTP)", lifespan=_lifespan)
app.mount("/mcp", _TokenGuardASGI(mcp_app))


# ---- 请求模型 ----
class SearchTextRequest(BaseModel):
    query: str
    k: int = 3
    score_ratio: float = 0.6

class SearchImageRequest(BaseModel):
    query: str
    k: int = 3
    include_portraits: bool = False
    score_ratio: float = 0.6

class GetChunkRequest(BaseModel):
    chunk_id: str


# ---- API 端点(业务实现在 engine_api,MCP 工具共用同一实现)----
@app.post("/search_text", dependencies=[Depends(_require_internal_token)])
def search_text(req: SearchTextRequest):
    """文本库混合检索(dense+sparse RRF)+ 精确代码兜底召回 + Cross-Encoder 重排。"""
    return engine_api.search_text(req.query, k=req.k, score_ratio=req.score_ratio)


@app.post("/search_image", dependencies=[Depends(_require_internal_token)])
def search_image(req: SearchImageRequest):
    """图像库检索(CLIP+BGE-m3 双路 RRF)+ Cross-Encoder 重排。"""
    return engine_api.search_image(
        req.query, k=req.k, include_portraits=req.include_portraits,
        score_ratio=req.score_ratio)


@app.post("/get_chunk", dependencies=[Depends(_require_internal_token)])
def get_chunk(req: GetChunkRequest):
    """按 chunk_id 取完整块/图。"""
    return engine_api.get_chunk(req.chunk_id)


# ---- 模型服务端点(供主服务 embed_http 调用,模型只在本进程加载)----
class EmbedTextRequest(BaseModel):
    texts: list[str]


class RerankRequest(BaseModel):
    query: str
    documents: list[str]


class IngestRequest(BaseModel):
    stem: str
    auto_dir: str
    source_path: str
    do_describe: bool = True
    wait: bool = True   # False=后台执行,立即返回 task_id(经 /ingest_status 查进度)


_te = None   # BGE-m3(懒加载)
_ie = None   # CLIP(懒加载)
_re = None   # Reranker(懒加载)


def _ensure_te(offline=False):
    """懒加载 BGE-m3 文本编码器(加锁防并发首请求重复加载)。

    在线(默认):查询/记忆嵌入;offline=True:入库专用独立副本(与在线并行,
    入库不再阻塞查询)。"""
    if offline:
        return embed.get_text_encoder(offline=True)
    global _te
    if _te is None:
        with engine_api._model_lock:
            if _te is None:
                _te = embed.get_text_encoder()
    return _te


def _ensure_ie(offline=False):
    """懒加载 CLIP 图像编码器。offline=True 为入库专用独立副本。"""
    if offline:
        return embed.get_image_encoder(offline=True)
    global _ie
    if _ie is None:
        with engine_api._model_lock:
            if _ie is None:
                _ie = embed.get_image_encoder()
    return _ie


@app.post("/embed_text", dependencies=[Depends(_require_internal_token)])
def embed_text(req: EmbedTextRequest):
    """BGE-m3 文本编码:返回 dense(n×1024)与 sparse 词权(长期记忆嵌入复用)。"""
    te = _ensure_te()
    dense, sparse = te.encode(req.texts)
    return {
        "dense": dense.tolist(),
        "sparse": [{str(k): float(v) for k, v in (s or {}).items()} for s in sparse],
    }


@app.post("/rerank", dependencies=[Depends(_require_internal_token)])
def rerank(req: RerankRequest):
    """BGE-reranker 对 (query, doc) 打分,返回归一化分数列表(长期记忆召回重排复用)。"""
    global _re
    if _re is None:
        with engine_api._model_lock:
            if _re is None:
                _re = embed.get_reranker()
    # rerank 不走攒批包装器,持自己的推理锁(与 encode/CLIP 并行:不同模型
    # 各持一把锁,互不排队)
    with embed.RERANK_LOCK:
        scores = _re.rerank(req.query, req.documents)
    return {"scores": [float(s) for s in scores]}


# 后台入库任务状态(wait=False 模式):单进程内存态即可(服务本身单进程部署),
# 只保留最近 _INGEST_TASK_KEEP 条,防长驻进程无限积累。
_ingest_tasks = {}
_ingest_lock = threading.Lock()
_INGEST_TASK_KEEP = 50


def _set_ingest_task(task_id, **kw):
    with _ingest_lock:
        t = _ingest_tasks.setdefault(task_id, {})
        t.update(kw)
        # 简单容量兜底:超限丢弃最早的已完成任务
        if len(_ingest_tasks) > _INGEST_TASK_KEEP:
            for k in [k for k, v in _ingest_tasks.items()
                      if v.get("status") in ("done", "error")][:_INGEST_TASK_KEEP // 2]:
                _ingest_tasks.pop(k, None)


def _run_ingest(req, task_id=None):
    """执行入库主体;task_id 非空时把进度写进 _ingest_tasks(后台模式)。"""
    try:
        pdf_dir = os.path.join(_RAG, "pdf")
        if pdf_dir not in sys.path:
            sys.path.insert(0, pdf_dir)
        import ingest as pdf_ingest  # RAG/pdf/ingest.py
        # 入库用离线(独立)模型副本:与在线查询编码完全并行,不阻塞用户提问
        pdf_ingest.TE = _ensure_te(offline=True)
        pdf_ingest.IE = _ensure_ie(offline=True)
        client = Q.get_client()
        pdf_ingest.ensure_collections(client)
        nt, ni = pdf_ingest.process_pdf(
            req.stem, req.auto_dir, req.source_path, client,
            do_describe=req.do_describe)
        if task_id:
            _set_ingest_task(task_id, status="done", text_count=nt,
                             image_count=ni)
        return {"ok": True, "text_count": nt, "image_count": ni}
    except Exception as e:
        logger.exception("ingest_document failed: %s", req.stem)
        if task_id:
            _set_ingest_task(task_id, status="error",
                             error=f"{type(e).__name__}: {str(e)[:200]}")
        return {"ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}"}


@app.post("/ingest_document", dependencies=[Depends(_require_internal_token)])
def ingest_document(req: IngestRequest):
    """切块 + 嵌入 + 入库一个已清洗 PDF(管理后台上传流水线调用)。

    全部在本进程执行:复用已加载的 BGE-m3/CLIP 与持有的 Qdrant 连接,
    主服务不接触模型/向量库。VL 图描述由 chunker 阶段经 ark_client 完成。

    wait=True(默认,兼容旧行为):同步执行,返回最终计数。
    wait=False:后台线程执行,立即返回 task_id;进度经 GET /ingest_status/{id}
    轮询(嵌入在大批量下会让路在线检索,总耗时变长,异步避免 HTTP 长挂/超时)。
    """
    if req.wait:
        return _run_ingest(req)
    task_id = uuid.uuid4().hex
    _set_ingest_task(task_id, status="running", stem=req.stem)
    threading.Thread(target=_run_ingest, args=(req, task_id),
                     daemon=True, name=f"ingest-{task_id[:8]}").start()
    return {"ok": True, "task_id": task_id, "status": "running"}


@app.get("/ingest_status/{task_id}",
         dependencies=[Depends(_require_internal_token)])
def ingest_status(task_id: str):
    """查询后台入库任务进度(wait=False 模式)。"""
    with _ingest_lock:
        t = dict(_ingest_tasks.get(task_id) or {})
    if not t:
        return {"ok": False, "error": "task not found"}
    t.setdefault("status", "running")
    t["task_id"] = task_id
    return {"ok": True, **t}


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    # 后台预热模型
    def _warmup():
        try:
            logger.info("后台预热 BGE-m3 + Reranker...")
            embed.get_text_encoder()
            embed.get_reranker()
            logger.info("预热完成")
        except Exception as e:
            # transformers 报错带超长模型类型列表,只取首行;reranker 缺失属可选增强,非致命
            reason = str(e).strip().splitlines()[0][:160] if str(e).strip() else type(e).__name__
            logger.warning(f"预热未完全完成(BGE-m3 已就绪,reranker 等可选模型缺失将自动降级): {reason}")

    threading.Thread(target=_warmup, daemon=True).start()

    _host = os.getenv("RETRIEVAL_HOST", "127.0.0.1")   # 容器部署设 0.0.0.0
    _port = int(os.getenv("RETRIEVAL_PORT", "8002"))
    print(f"检索服务(MCP /mcp + HTTP)  http://{_host}:{_port}", flush=True)
    uvicorn.run(app, host=_host, port=_port, log_level="info")
