# -*- coding: utf-8 -*-
"""检索业务实现:三个工具函数(向量召回 + 重排 + payload 归一化)。

由两条通道共用:
  - MCP 工具(tools.py,agent 经 tools/mcp_bridge.py 调用);
  - 传统 HTTP 端点(service.py,/search_text /search_image /get_chunk,兼容保留)。

模型(BGE-m3/CLIP/Reranker)只在本服务进程加载;失败直接抛异常,由调用方
(agent 侧韧性中间件 / HTTP raise_for_status)处理,不在此吞成 error-dict。
"""
import os
import sys
import logging

# ---- 路径设置:本文件位于 mcp_servers/retrieval/,项目根在上 2 级 ----
# 兜底 sys.path(仅 config/ 与 RAG/,供裸 import config/embed):
# 注意:不得把本目录(_HERE)插入 sys.path——agent 进程会经 MCP 桥加载本包,
# 目录里有 tools.py,插入会让 ``import tools`` 解析到检索服务的工具模块而非
# agent 工具包。包内模块一律相对导入。
_HERE = os.path.dirname(os.path.abspath(__file__))          # mcp_servers/retrieval
_PROJECT = os.path.dirname(os.path.dirname(_HERE))          # project root
for _p in (os.path.join(_PROJECT, "config"),
           os.path.join(_PROJECT, "RAG")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

logger = logging.getLogger("retrieval.engine")

import threading                       # noqa: E402
import config as C                     # noqa: E402
import embed                           # noqa: E402  (RAG/embed.py,模型懒加载)
from . import query as Q               # noqa: E402  (Qdrant 检索引擎)
from .chunk_format import (            # noqa: E402
    _pid, _text_dict, _image_dict, _collection_hint,
)

_re = None               # Reranker(懒加载)
_reranker_failed = False  # reranker 一旦加载失败即粘性禁用(本进程不再重试),避免每请求重复尝试+刷巨屏警告
_model_lock = threading.Lock()  # 懒加载互斥:并发首请求不重复加载模型


def _rerank(query, points, k=3, text_fn=None):
    """Cross-Encoder 重排;reranker 为可选增强,加载失败即粘性禁用(本进程不再重试),
    直接按召回分数排序返回。"""
    global _re, _reranker_failed
    if len(points) <= 1:
        return [(p, p.score) for p in points]
    if _reranker_failed:
        return [(p, p.score) for p in points[:k]]
    try:
        if _re is None:
            # 加锁:启动初期并发首请求会各自触发加载(数 GB 内存/数分钟)
            with _model_lock:
                if _re is None:
                    _re = embed.get_reranker()
        documents = (
            [text_fn(p) for p in points] if text_fn
            else [(p.payload or {}).get("content", "") for p in points]
        )
        # rerank 不走攒批包装器,持自己的推理锁(与 encode/CLIP 并行)
        with embed.RERANK_LOCK:
            scores = _re.rerank(query, documents)
        ranked = sorted(zip(points, scores), key=lambda x: x[1], reverse=True)
        return [(p, s) for p, s in ranked[:k]]
    except Exception as e:
        # 只记录一次精简原因(transformers 报错会带超长模型类型列表,截断到首行)
        reason = str(e).strip().splitlines()[0][:160] if str(e).strip() else type(e).__name__
        logger.warning(f"reranker 不可用,本次运行回退为按召回分数排序(不再重试): {reason}")
        _reranker_failed = True
        return [(p, p.score) for p in points[:k]]


def _image_text(p):
    pl = p.payload or {}
    parts = [pl.get("caption", ""), pl.get("description", "")]
    text = "\n".join(s for s in parts if s).strip()
    return text or "[无描述]"


def _text_rerank_text(p):
    """重排文本 = 标题路径 + 正文。与索引侧对齐(chunker 落库时 embedding 文本
    即 f"{heading_path}\n{content}"),让重排器看到与向量检索一致的主题上下文,
    提升区分度(纯 content 时无关块 p75 分高达 0.89,τ 截不准)。"""
    pl = p.payload or {}
    hp = (pl.get("heading_path") or "").strip()
    content = pl.get("content", "") or ""
    return f"{hp}\n{content}" if hp else content


def _dynamic_cut(ranked, k):
    """重排后按绝对分动态截断:score≥RERANK_TAU 的进结果,cap MAX_K;
    全部低于 τ 时保底返回 top RERANK_MIN_K(通常 1,避免空结果)。
    reranker 不可用的回退路径直接按原 k 截断——回退排序用的是 RRF 分,
    与 0~1 的 rerank 分不同标,不适用 τ。"""
    if _reranker_failed:
        return ranked[:k]
    tau = float(getattr(C, "RERANK_TAU", 0.5))
    max_k = int(getattr(C, "RERANK_MAX_K", 6)) or k
    min_k = max(1, int(getattr(C, "RERANK_MIN_K", 1)))
    kept = [(p, s) for p, s in ranked if s >= tau][:max_k]
    if len(kept) < min_k:
        kept = ranked[:min_k]
    return kept


def _ratio_pool(fused, score_ratio):
    """ratio 过滤 + 保底池。RRF 归一化分 Top-Heavy(第2名常仅 ~0.58),纯 ratio
    过滤可能只剩 1~2 个候选,压死重排池(RERANK_RECALL_K 白加)。过滤后不足
    RERANK_POOL_MIN 时按 RRF 序取前缀补齐——最终精度由 reranker+τ 截断负责。"""
    if not fused:
        return []
    top_score = fused[0].score
    kept = sum(1 for p in fused if p.score >= top_score * score_ratio)
    pool_min = max(1, int(getattr(C, "RERANK_POOL_MIN", 8)))
    return fused[:max(kept, pool_min)]


def search_text(query, k=3, score_ratio=0.4):
    """文本库混合检索(dense+sparse RRF)+ 精确代码兜底召回 + Cross-Encoder 重排。
    截断不再固定 top-k:重排分 ≥RERANK_TAU 才保留(动态 0~MAX_K 块,保底 top-1)。"""
    recall_k = getattr(C, "RERANK_RECALL_K", 20)
    fused = Q.query_text(query, k=recall_k)
    # 精确报警/故障代码兜底:含代码词面的块常排在 RRF 池外,显式并入候选,
    # 再统一交给 reranker 按相关性重排。
    exact = Q.exact_code_recall(query, limit=max(k, 6))
    exact_ids = {p.id for p in exact}
    # score_ratio 阈值只约束向量融合召回部分(兜底块原始 sparse 分偏低,不参与)。
    filtered = _ratio_pool(fused, score_ratio)
    have = {p.id for p in filtered}
    filtered = [p for p in exact if p.id not in have] + filtered
    if not filtered:
        return []
    ranked = _rerank(query, filtered,
                     k=max(int(getattr(C, "RERANK_MAX_K", 6)), k),
                     text_fn=_text_rerank_text)
    out = [_text_dict(p.payload or {}, score=s)
           for p, s in _dynamic_cut(ranked, k)]
    # sparse 保底:重排器对表格/码表块的词面强命中系统性打低分(run20 id10 探针),
    # τ 截断把它们整块丢掉。这里把 sparse top-N 追加到结果尾部(chunk_id 去重,
    # 评分压在 τ 之下),后端宽喂 prompt 时兜住"表在库里但排不进"类问题。
    if getattr(C, "RETRIEVAL_SPARSE_RESCUE", True) and not _reranker_failed:
        have = {d["chunk_id"] for d in out}
        rescue_score = float(getattr(C, "RERANK_TAU", 0.4)) * 0.9
        for p in Q.sparse_rescue(query, limit=int(getattr(C, "RETRIEVAL_SPARSE_RESCUE_K", 2))):
            cid = (p.payload or {}).get("chunk_id")
            if cid and cid not in have:
                out.append(_text_dict(p.payload or {}, score=rescue_score))
    return out


def search_image(query, k=3, include_portraits=False, score_ratio=0.4):
    """图像库检索(CLIP+BGE-m3 双路 RRF)+ Cross-Encoder 重排(动态截断同文本)。"""
    recall_k = getattr(C, "RERANK_RECALL_K", 20)
    points = Q.query_image(query, k=recall_k, include_portraits=include_portraits)
    if not points:
        return []
    filtered = _ratio_pool(points, score_ratio)
    ranked = _rerank(query, filtered,
                     k=max(int(getattr(C, "RERANK_MAX_K", 6)), k),
                     text_fn=_image_text)
    return [_image_dict(p.payload or {}, score=s)
            for p, s in _dynamic_cut(ranked, k)]


def get_chunk(chunk_id):
    """按 chunk_id 取完整块/图(文本库/图像库各查一次)。找不到返回 None。"""
    client = Q.get_client()
    pid = _pid(chunk_id)
    hint = _collection_hint(chunk_id)
    if hint is not None:
        other = C.IMAGE_COLLECTION if hint == C.TEXT_COLLECTION else C.TEXT_COLLECTION
        order = [hint, other]
    else:
        order = [C.TEXT_COLLECTION, C.IMAGE_COLLECTION]
    for coll in order:
        try:
            hits = client.retrieve(coll, ids=[pid], with_payload=True, with_vectors=False)
        except Exception:
            hits = []
        if hits:
            pl = hits[0].payload or {}
            return _text_dict(pl) if coll == C.TEXT_COLLECTION else _image_dict(pl)
    return None
