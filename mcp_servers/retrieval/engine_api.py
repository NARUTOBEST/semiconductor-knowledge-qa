# -*- coding: utf-8 -*-
"""检索业务实现:三个工具函数(向量召回 + 重排 + payload 归一化)。

由两条通道共用:
  - MCP 工具(tools.py,agent 经 tools/mcp_bridge.py 调用);
  - 传统 HTTP 端点(service.py,/search_text /search_image /get_chunk,兼容保留)。

模型(BGE-m3/CLIP/Reranker)只在本服务进程加载;失败直接抛异常,由调用方
(agent 侧韧性中间件 / HTTP raise_for_status)处理,不在此吞成 error-dict。
"""
import os
import json
import re
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

import time                            # noqa: E402
import threading                       # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402  (混合重排双路并行)
import config as C                     # noqa: E402
import embed                           # noqa: E402  (RAG/embed.py,模型懒加载)
from tool_circuit import (  # noqa: E402  项目根级模块(GPU 部署面无 agent_reasoning 包)
    CircuitBreaker, CircuitOpenError, CircuitState,
)
from . import query as Q               # noqa: E402  (Qdrant 检索引擎)
from .chunk_format import (            # noqa: E402
    _pid, _text_dict, _image_dict, _collection_hint,
)

_re = None               # Reranker(懒加载)
_re2 = None              # 混合重排副路 Reranker(懒加载;未配 RERANK2_MODEL 时为 None)
_reranker_failed = False  # reranker 一旦加载失败即粘性禁用(本进程不再重试),避免每请求重复尝试+刷巨屏警告
_model_lock = threading.Lock()  # 懒加载互斥:并发首请求不重复加载模型


# ---- 外呼三状态熔断(translate/rerank;复用 agent 侧状态机,本服务独立进程)----
def _breaker(name):
    return CircuitBreaker(
        "retrieval." + name,
        failure_threshold=int(getattr(C, "RETRIEVAL_BREAKER_THRESHOLD", 2)),
        cooldown_seconds=float(getattr(C, "RETRIEVAL_BREAKER_COOLDOWN_S", 30)),
        on_state_change=lambda n, o, new: logger.warning(
            "circuit breaker %s: %s -> %s", n, o, new))


_translate_brk = _breaker("translate")
_rerank_brk = _breaker("rerank")
_rerank2_brk = _breaker("rerank2")       # 混合重排副路(词面系)独立熔断


def _err_severity(e):
    """4xx(除 429)按 fatal:不重试、CLOSED 下不计数(配置错不该烧熔断)。"""
    code = getattr(e, "code", None)
    if isinstance(code, int) and 400 <= code < 500 and code != 429:
        return "fatal"
    return "retryable"


def _call_with_backoff(brk, fn, *args, **kwargs):
    """三状态熔断驱动外呼:
    CLOSED 内超时/失败按指数退避重试,重试耗尽记一次失败(连续 THRESHOLD 次
    转 OPEN);OPEN 快速失败抛 CircuitOpenError(调用方静默降级 fail-open);
    冷却到期由状态机转 HALF_OPEN,放 1 个单次探测,成功回 CLOSED、失败回 OPEN。
    """
    attempts = int(getattr(C, "RETRIEVAL_BREAKER_RETRY_ATTEMPTS", 2))
    if brk.state == CircuitState.HALF_OPEN:
        attempts = 1                       # 半开探测只打一发,失败即回 OPEN
    brk.before_call()                      # OPEN -> CircuitOpenError
    base = float(getattr(C, "RETRIEVAL_BREAKER_BACKOFF_BASE_S", 0.5))
    cap = float(getattr(C, "RETRIEVAL_BREAKER_BACKOFF_MAX_S", 4.0))
    for i in range(max(1, attempts)):
        try:
            out = fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001
            sev = _err_severity(e)
            if i + 1 < max(1, attempts) and sev == "retryable":
                time.sleep(min(base * (2 ** i), cap))
                continue
            brk.on_failure(sev)
            raise
        brk.on_success()
        return out
    raise RuntimeError("unreachable")      # pragma: no cover


def breaker_states():
    """/health 观测:各外呼熔断器当前状态。"""
    return {"translate": _translate_brk.state, "rerank": _rerank_brk.state,
            "rerank2": _rerank2_brk.state}



def _score_once(base, model, query, documents):
    """调一个 vLLM 重排服务 POST {base}/rerank(Cohere/Jina 兼容 rerank API):
    {"model","query","documents"} -> results[{index,relevance_score}]。
    兼容旧 /score 响应(data[{index,score}])与新响应两种形状。"""
    import json as _json
    import urllib.request
    body = _json.dumps({
        "model": model,
        "query": query,
        "documents": documents,
    }).encode()
    req = urllib.request.Request(
        base.rstrip("/") + "/rerank", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=25) as r:
        d = _json.load(r)
    items = d.get("results") if isinstance(d.get("results"), list) \
        else (d.get("data") or [])
    scores = [0.0] * len(documents)
    for it in items:
        i = int(it.get("index", -1))
        if 0 <= i < len(documents):
            scores[i] = float(it.get("relevance_score",
                                     it.get("score")) or 0.0)
    return scores


def _rerank_http(query, documents):
    """主路 vLLM /score(兼容旧入口;参数化版本见 _score_once)。"""
    return _score_once(C.RERANK_API_BASE,
                       getattr(C, "RERANK_MODEL", "BAAI/bge-reranker-v2-m3"),
                       query, documents)


def _blend_score_sets(score_sets, weights):
    """多路重排分混合:各路按本请求的 min-max 归一到 [0,1] 后加权平均。

    保持 0~1 标尺是硬约束:下游 _dynamic_cut/弱末位补录用 RERANK_TAU 比较的
    是这个分数。单路(另一路失败)原样返回该路原始分,标尺不变。全部缺失
    返回 None(调用方回退召回序)。
    """
    sets = [(s, w) for s, w in zip(score_sets, weights) if s is not None]
    if not sets:
        return None
    if len(sets) == 1:
        return list(sets[0][0])
    n = len(sets[0][0])
    out = [0.0] * n
    wsum = 0.0
    for scores, w in sets:
        lo, hi = min(scores), max(scores)
        if hi == lo:
            norm = [0.5] * len(scores)          # 恒分路无区分度,贡献中性 0.5
        else:
            rng = hi - lo
            norm = [(s - lo) / rng for s in scores]
        wsum += w
        for i, s in enumerate(norm):
            out[i] += w * s
    return [v / wsum for v in out]

_sib_cache = {}          # chunk_id -> 邻居行前缀(方案二A;None=查过但无邻居/不可用)

# ---- 查询英文翻译路(轮次9:跨语言盲区治理) ----
_tr_cache = {}           # query -> 英文翻译(fail-open 为空串;上限 512 条后清空)
_TRA_PROMPT = ("把用户的中文问题翻译成一句用于检索英文半导体设备手册的英文查询。"
               "保留型号/数字/单位原样,输出英文查询本身,不要解释。\n\n用户问题:")


def _translate_query(query):
    """中文 query -> 英文检索查询(副模型直调,阻塞;调用方自行权衡延迟)。
    任何失败 fail-open 返回 ""(召回单路照常);基本 ASCII 的 query 直接跳过。"""
    q = (query or "").strip()
    if not q or not _CJK_SEG.search(q):
        return ""
    hit = _tr_cache.get(q)
    if hit is not None:
        return hit
    out = ""
    try:
        import ssl
        import urllib.request
        # AutoDL 镜像缺系统 CA,urllib 直连云方舟会 SSLCertVerificationError:
        # 优先 certifi 证书,缺包则跳过校验(端点固定为方舟、内容仅 query 文本)
        try:
            import certifi
            ctx = ssl.create_default_context(cafile=certifi.where())
        except Exception:  # noqa: BLE001
            ctx = ssl._create_unverified_context()
        base = (getattr(C, "TRANSLATE_BASE", "") or C.OPENAI_BASE_URL).rstrip("/")
        key = getattr(C, "TRANSLATE_KEY", "") or C.OPENAI_API_KEY
        model = getattr(C, "TRANSLATE_MODEL", "") or C.MODEL_LIGHT
        body = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": _TRA_PROMPT + q}],
            "temperature": 0,
            "max_tokens": int(getattr(C, "TRANSLATE_MAX_TOKENS", 120)),
            # Qwen3 默认开思考,120 tokens 会被 reasoning 烧光导致 content 为空;
            # 仅本地 vLLM 路径需要关思考(云端兜底时本字段由服务端 drop_params 吞掉)
            **({"chat_template_kwargs": {"enable_thinking": False}}
               if getattr(C, "TRANSLATE_NO_THINK", True) else {}),
            # doubao-seed 系默认开思考:思考烧延迟(实测单次 8-24s)且 120
            # max_tokens 会被 reasoning 吃掉;Ark 云端用 thinking 参数关之
            # (20261009 实测 8-24s -> 1-3s;不支持的端点由服务端忽略)
            **({"thinking": {"type": "disabled"}}
               if getattr(C, "TRANSLATE_NO_THINK", True) else {}),
        }).encode()
        req = urllib.request.Request(
            base + "/chat/completions", data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer " + key})

        def _once():
            with urllib.request.urlopen(
                    req, timeout=float(getattr(C, "TRANSLATE_TIMEOUT", 8)),
                    context=ctx) as r:
                return json.load(r)

        # 三状态熔断驱动:CLOSED 内超时指数退避重试;OPEN 快速失败静默降级
        # 单路召回(翻译是增强层,fail-open 零伤害)
        d = _call_with_backoff(_translate_brk, _once)
        out = (d["choices"][0]["message"]["content"] or "").strip()
        out = re.sub(r"^(英文查询|英文|English)\s*[:：]\s*", "", out,
                     flags=re.IGNORECASE).strip('`"“” ')
    except CircuitOpenError:
        pass  # 熔断 OPEN/半开探测占满:快速失败,静默单路(不刷日志)
    except Exception:  # noqa: BLE001  翻译是增强层,失败只降级单路召回
        logger.warning("translate query failed (q=%r)", q[:40], exc_info=True)
    if len(_tr_cache) > 512:
        _tr_cache.clear()
    _tr_cache[q] = out
    return out


def _rerank(query, points, k=3, text_fn=None):
    """Cross-Encoder 重排;reranker 为可选增强,加载失败即粘性禁用(本进程不再重试),
    直接按召回分数排序返回。

    2026-10-08 架构:RERANK_API_BASE 非空时改走独立 vLLM 服务的 /score
    (continuous batching,并发吞吐 5-10 倍于进程内 transformers);HTTP 失败
    经三状态熔断(CLOSED 指数退避重试 -> OPEN 静默降级召回序 -> HALF_OPEN
    单探测恢复),vLLM 独立进程重启后自动回 CLOSED。
    """
    global _re, _re2, _reranker_failed
    if len(points) <= 1:
        return [(p, p.score) for p in points]
    if _reranker_failed:
        return [(p, p.score) for p in points[:k]]
    documents = (
        [text_fn(p) for p in points] if text_fn
        else [(p.payload or {}).get("content", "") for p in points]
    )
    # ---- 内容去重(2026-10-13):入库分块产生的完全重复块在 top-N 里白吃席位
    # (dev_184 实锤 top-6 两组一模一样,K=4 实际只有 ~3 席有效),按规范化文本
    # 去重保留首现。去重后为空不可能(>1 块必有首现);sibling 去重按 chunk_id
    # 在 _doc_cap_select,此处只治逐字节重复。
    _seen_txt: set = set()
    _ded: list = []
    for _p, _d in zip(points, documents):
        _h = hash(re.sub(r"\s+", "", _d))
        if _h in _seen_txt:
            continue
        _seen_txt.add(_h)
        _ded.append((_p, _d))
    if len(_ded) < len(points):
        logger.info("rerank dedup: %d -> %d", len(points), len(_ded))
    points = [_p for _p, _ in _ded]
    documents = [_d for _, _d in _ded]
    # ---- 级联粗筛(2026-10-13):副路(词面系,便宜)先对全量召回池打分,
    # 保留 top RERANK_CASCADE_KEEP 进混合双路精排——主路 4B 算力 ÷4(96→24),
    # 治 20u 并发重排排队。粗筛失败/副路未配置自动退全量(fail-open,级联是
    # 增强层:词面系漏召的块本就难进 m3 top-N,混合融合语义保持不变)。
    if getattr(C, "RETRIEVAL_RERANK_CASCADE", False) \
            and len(points) > int(getattr(C, "RERANK_CASCADE_KEEP", 24)):
        try:
            if _re2 is None and getattr(C, "RERANK2_MODEL", "").strip() \
                    and getattr(C, "RERANK2_API_BASE", "") == "":
                with _model_lock:
                    if _re2 is None:
                        _re2 = embed.get_reranker2()
            if _re2 is not None:
                coarse = _re2.rerank(query, documents)
                keep = int(getattr(C, "RERANK_CASCADE_KEEP", 24))
                order = sorted(range(len(points)),
                               key=lambda i: coarse[i], reverse=True)[:keep]
                points = [points[i] for i in order]
                documents = [documents[i] for i in order]
                logger.info("cascade coarse: pool=%d -> keep=%d", len(coarse), len(order))
            else:
                logger.info("cascade skip: reranker2 unavailable")
        except Exception as e:
            logger.warning("级联粗筛失败,本请求退全量精排: %s",
                           str(e).strip().splitlines()[0][:160])
    # ---- 混合重排(2026-10-11):主路(语义系)+ 副路(词面系)并行打分,
    # min-max 归一加权融合;任一路失败退单路,双路皆败回退召回序。
    # 两路独立熔断,互不拖累。
    w1 = float(getattr(C, "RERANK_W1", 0.6))
    weights = [w1, 1.0 - w1]

    def _pick(fut, tag):
        try:
            return fut.result()
        except CircuitOpenError:
            return None                         # 熔断 OPEN:该路静默缺席
        except Exception as e:
            logger.warning("混合重排%s失败,本请求退单路: %s", tag,
                           str(e).strip().splitlines()[0][:160])
            return None

    if getattr(C, "RERANK_API_BASE", ""):
        base2 = getattr(C, "RERANK2_API_BASE", "").strip()
        model2 = getattr(C, "RERANK2_MODEL", "").strip()
        try:
            if base2 and model2:
                with ThreadPoolExecutor(max_workers=2) as ex:
                    f1 = ex.submit(_call_with_backoff, _rerank_brk, _score_once,
                                   C.RERANK_API_BASE,
                                   getattr(C, "RERANK_MODEL", ""),
                                   query, documents)
                    f2 = ex.submit(_call_with_backoff, _rerank2_brk, _score_once,
                                   base2, model2, query, documents)
                blended = _blend_score_sets(
                    [_pick(f1, "主路"), _pick(f2, "副路")], weights)
                if blended is not None:
                    ranked = sorted(zip(points, blended),
                                    key=lambda x: x[1], reverse=True)
                    return [(p, s) for p, s in ranked[:k]]
                return [(p, p.score) for p in points[:k]]   # 双路皆败
            scores = _call_with_backoff(_rerank_brk, _rerank_http,
                                        query, documents)
            ranked = sorted(zip(points, scores), key=lambda x: x[1], reverse=True)
            return [(p, s) for p, s in ranked[:k]]
        except CircuitOpenError:
            # 熔断 OPEN(或半开探测占满):快速失败,本请求回退召回序(静默)
            return [(p, p.score) for p in points[:k]]
        except Exception as e:
            logger.warning("vLLM 重排失败(退避重试后),本请求回退召回序: %s",
                           str(e).strip().splitlines()[0][:160])
            return [(p, p.score) for p in points[:k]]
    try:
        if _re is None:
            # 加锁:启动初期并发首请求会各自触发加载(数 GB 内存/数分钟)
            with _model_lock:
                if _re is None:
                    _re = embed.get_reranker()
        # rerank 走攒批管线(get_rerank_pipeline):跨请求合并 + 副本锁内部自管
        # (与 encode/CLIP 天然并行,不再持全局 RERANK_LOCK)
        model2 = getattr(C, "RERANK2_MODEL", "").strip()
        if model2 and getattr(C, "RERANK2_API_BASE", "") == "":
            if _re2 is None:
                with _model_lock:
                    if _re2 is None:
                        _re2 = embed.get_reranker2()
        if _re2 is not None:
            with ThreadPoolExecutor(max_workers=2) as ex:
                f1 = ex.submit(_re.rerank, query, documents)
                f2 = ex.submit(_re2.rerank, query, documents)
            blended = _blend_score_sets(
                [_pick(f1, "主路"), _pick(f2, "副路")], weights)
            if blended is not None:
                ranked = sorted(zip(points, blended),
                                key=lambda x: x[1], reverse=True)
                return [(p, s) for p, s in ranked[:k]]
            return [(p, p.score) for p in points[:k]]       # 双路皆败
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


def _sibling_prefix(pl):
    """表格/码表行块的上下文前缀(方案二A):同文档同标题路径的相邻行(含表头),
    让 cross-encoder 看到"干瘪短行"所属的表结构。RERANK_ROW_CONTEXT 关闭、
    长块(>200字,不缺上下文)、或查询失败时返回空串;按 chunk_id 缓存。"""
    if not getattr(C, "RERANK_ROW_CONTEXT", False):
        return ""
    cid = pl.get("chunk_id")
    hp = (pl.get("heading_path") or "").strip()
    content = pl.get("content", "") or ""
    stem = pl.get("source_stem", "") or ""
    if not cid or cid in _sib_cache or not hp or not stem or len(content) > 200:
        return _sib_cache.get(cid) or ""
    try:
        res = Q.get_client().scroll(
            collection_name=C.TEXT_COLLECTION,
            scroll_filter=Q.models.Filter(must=[
                Q.models.FieldCondition(key="source_stem",
                                        match=Q.models.MatchValue(value=stem)),
                Q.models.FieldCondition(key="heading_path",
                                        match=Q.models.MatchValue(value=hp)),
            ]),
            limit=8, with_payload=True, with_vectors=False,
        )
        points = res[0] if isinstance(res, tuple) else res.points
        sibs = []
        for pt in points or []:
            spl = pt.payload or {}
            if spl.get("chunk_id") == cid:
                continue
            s = (spl.get("content", "") or "").strip()
            if s:
                sibs.append(s[:120])
            if len(sibs) >= 2:
                break
        prefix = "\n".join(f"[同表其他行] {s}" for s in sibs)
    except Exception as e:
        logger.warning("sibling context lookup failed (chunk=%s): %s", cid, str(e)[:80])
        prefix = ""
    _sib_cache[cid] = prefix
    return prefix


def _text_rerank_text(p):
    """重排文本 = 标题路径 + 正文。与索引侧对齐(chunker 落库时 embedding 文本
    即 f"{heading_path}\n{content}"),让重排器看到与向量检索一致的主题上下文,
    提升区分度(纯 content 时无关块 p75 分高达 0.89,τ 截不准)。
    RERANK_ROW_CONTEXT 开启时,短表格行块额外前置相邻行(方案二A)。"""
    pl = p.payload or {}
    hp = (pl.get("heading_path") or "").strip()
    content = pl.get("content", "") or ""
    sib = _sibling_prefix(pl)
    head = f"{hp}\n{sib}\n" if (hp or sib) else ""
    return f"{head}{content}"


def _dynamic_cut(ranked, k):
    """重排后截断,RERANK_FIXED_K>0 时固定 top-K(评测对照基线),否则两级动态:
    ① 绝对阈值:score≥RERANK_TAU 的进结果,cap MAX_K;全部低于 τ 时保底返回
       top RERANK_MIN_K(避免空结果)。
    ② 断崖检测:过线块从头往后扫,相邻分 score[i] < RERANK_CLIFF_RATIO *
       score[i-1] 处截断(保底 top-1)——金标块群分数扎堆,群后断崖,再后面的
       "过线但平庸"块剔除。设 RERANK_CLIFF_RATIO=0 关闭。
    reranker 不可用的回退路径直接按原 k 截断——回退排序用的是 RRF 分,
    与 0~1 的 rerank 分不同标,不适用 τ/断崖。"""
    if _reranker_failed:
        return ranked[:k]
    fixed_k = int(getattr(C, "RERANK_FIXED_K", 0) or 0)
    if fixed_k > 0:
        return ranked[:fixed_k]
    tau = float(getattr(C, "RERANK_TAU", 0.5))
    max_k = int(getattr(C, "RERANK_MAX_K", 6)) or k
    min_k = max(1, int(getattr(C, "RERANK_MIN_K", 1)))
    kept = [(p, s) for p, s in ranked if s >= tau][:max_k]
    if len(kept) < min_k:
        kept = ranked[:min_k]
    ratio = float(getattr(C, "RERANK_CLIFF_RATIO", 0.6) or 0)
    if ratio > 0 and len(kept) > 1:
        for i in range(1, len(kept)):
            if kept[i][1] < ratio * kept[i - 1][1]:
                kept = kept[:i]
                break
    return kept


def _doc_base(stem):
    """基础文档名:分段入库的页段后缀(_p1-500/_p501-794 等)归一,
    配额按物理文档计,防止同手册多分段绕过配额。"""
    return re.sub(r"_p\d+(?:-\d+)?$", "", stem or "")


def _doc_cap_select(kept, ranked, cap):
    """按基础文档名配额重选:每文档最多 cap 席,挤出名额从其余候选按名次回填
    (回填资格:重排分 ≥ RERANK_TAU,precision 不放水);候选不足时超配额原块
    按原序保底回填,返回块数与 kept 相同。治理多跳题"单文档霸榜挤掉跳板块"。"""
    if cap <= 0 or len(kept) <= 1:
        return kept
    kept_ids = {p.id for p, _ in kept}
    floor = float(getattr(C, "RERANK_TAU", 0.3))
    per, sel, rest = {}, [], []
    for p, s in kept:                       # 先按原序收 kept,超配额的挂起
        doc = _doc_base((p.payload or {}).get("source_stem", ""))
        if per.get(doc, 0) < cap:
            per[doc] = per.get(doc, 0) + 1
            sel.append((p, s))
        else:
            rest.append((p, s))
    for p, s in ranked:                     # 按名次从其余候选回填空出的名额
        if len(sel) >= len(kept):
            break
        if p.id in kept_ids:
            continue
        doc = _doc_base((p.payload or {}).get("source_stem", ""))
        if per.get(doc, 0) >= cap or s < floor:
            continue
        per[doc] = per.get(doc, 0) + 1
        sel.append((p, s))
    for p, s in rest:                       # 候选耗尽仍不满:超配额原块保底
        if len(sel) >= len(kept):
            break
        sel.append((p, s))
    sel.sort(key=lambda x: x[1], reverse=True)   # 回填混排后按重排分归位
    return sel


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


_SPLIT_SUFFIX = re.compile(r"_p\d+[_-]\d*$")   # 分段文档 stem 后缀(…_p501-794)
_ALNUM_TOK = re.compile(r"[a-z0-9][a-z0-9-]*[a-z0-9]|[a-z0-9]")
_CJK_SEG = re.compile(r"[一-鿿]{2,}")

_stem_registry = None   # base_stem -> set(实际 stem);惰性加载一次


def _base_stem(stem):
    """分段文档的实际 stem 归并到 base(…_p1-500 / …_p501-794 → 同一文档)。"""
    return _SPLIT_SUFFIX.sub("", stem or "")


def _load_stem_registry():
    """全库 source_stem 注册表(按 base stem 归并)。语料静态,进程内缓存一次;
    优先读盘上 stem_registry.json(预生成,40 万点全量扫描过隧道太慢);
    盘上无缓存才退回全库 scroll(慢),成功后回写缓存。
    失败返回 None——检索域收窄整体静默降级为无收窄,不影响检索主链路。"""
    global _stem_registry
    if _stem_registry is not None:
        return _stem_registry
    cache = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "stem_registry.json")
    try:
        if os.path.exists(cache):
            with open(cache, encoding="utf-8") as f:
                _stem_registry = {k: set(v) for k, v in json.load(f).items()}
            return _stem_registry
    except Exception:  # noqa: BLE001  缓存损坏按无缓存处理,走全量扫描
        pass
    reg = {}
    try:
        offset, client = None, Q.get_client()
        while True:
            res = client.scroll(collection_name=C.TEXT_COLLECTION,
                                limit=256, with_payload=["source_stem"],
                                with_vectors=False, offset=offset)
            if isinstance(res, tuple):
                points, offset = res
            else:
                points, offset = res.points, res.next_page_offset
            for pt in points or []:
                stem = (pt.payload or {}).get("source_stem") or ""
                if stem:
                    reg.setdefault(_base_stem(stem), set()).add(stem)
            if offset is None:
                break
    except Exception as e:  # noqa: BLE001  收窄是增强层,任何故障都不阻塞检索
        logger.warning("stem registry load failed, source scope disabled: %s",
                       str(e)[:100])
        return None
    try:  # 回写盘上缓存,下次冷启动免全量扫描
        with open(cache, "w", encoding="utf-8") as f:
            json.dump({k: sorted(v) for k, v in reg.items()}, f,
                      ensure_ascii=False, separators=(",", ":"))
    except Exception:  # noqa: BLE001  写缓存失败不影响本次结果
        pass
    _stem_registry = reg
    return reg


def infer_target_stem(query):
    """从 query 推断用户所指的目标文档(base stem):标题词元(alnum 词按长度、
    CJK 连续段按长度)在 query 中出现的分值累加。最高分需 ≥SOURCE_SCOPE_MIN_SCORE
    且 ≥ 次高×SOURCE_SCOPE_MARGIN 才算确信;否则返回 None(宁不加权不误伤)。"""
    reg = _load_stem_registry()
    if not reg or not query:
        return None
    q = query.lower()
    strict = bool(getattr(C, "SOURCE_SCOPE_STRICT_TOKENS", True))
    best, second = None, 0.0
    for base in reg:
        score = 0.0
        for tok in _ALNUM_TOK.findall(base.lower()):
            t = tok.lower()
            if len(t) < 3 or t not in q:
                continue
            # 轮次9 词面误锁护栏:纯字母通用词(screw/test/manual)与纯数字
            # (500 等,易与页码/参数撞车,实测 dev_198 "500 nm" 锁错 SR-70)
            # 都不计分;只认字母数字混合的型号/编号 token(6DS-SP、M6x20)
            # 或全大写缩写(VIII、CVD)。SOURCE_SCOPE_STRICT_TOKENS=0 关闭。
            if strict and not tok.isupper() and not (
                    any(ch.isdigit() for ch in t)
                    and any(ch.isalpha() for ch in t)):
                continue
            score += len(t)
        for seg in _CJK_SEG.findall(base):
            if seg in q:
                score += len(seg)
        if score <= 0:
            continue
        cur = best[1] if best else 0.0
        if score > cur:
            second, best = cur, (base, score)
        elif score > second:
            second = score
    lo = float(getattr(C, "SOURCE_SCOPE_MIN_SCORE", 4))
    mg = float(getattr(C, "SOURCE_SCOPE_MARGIN", 1.5))
    if best and best[1] >= lo and best[1] >= second * mg:
        return best[0]
    return None


def _rrf_merge(a, b, k=60, w_b=1.0):
    """两路检索结果按名次 RRF 合并(轮次9:原路 + 英文翻译路)。
    只用名次不比分值(两路分数量纲同但语义弱可比);辅助路带 w_b 降权
    (默认 0.7,原路名次优先),噪声块要在原路也站得住才能浮上来。
    同块双向受益,单路头部也保留。"""
    scores = {}
    for rank, p in enumerate(a):
        scores[p.id] = scores.get(p.id, 0.0) + 1.0 / (k + rank + 1)
    for rank, p in enumerate(b):
        scores[p.id] = scores.get(p.id, 0.0) + w_b / (k + rank + 1)
    out = [p for p in a if p.id in scores]
    seen = {p.id for p in out}
    out.extend(p for p in b if p.id not in seen)
    out.sort(key=lambda p: scores[p.id], reverse=True)
    return out


def _rescue_seat(out, rescued):
    """固定 top-K 的稀疏补录席位(方案一):重排末位过弱(<RERANK_TAU)时,让位给
    不在场的 sparse 强命中块——表格/码表行被重排系统性压分的治理。无补录、补录
    已在场、或末位分数硬朗,均原样返回(返回块数恒等于传入块数)。"""
    if not rescued or not out:
        return out
    tau = float(getattr(C, "RERANK_TAU", 0.5))
    if out[-1].get("score", 0) >= tau:
        return out
    have = {d.get("chunk_id") for d in out}
    for p in rescued:
        pl = p.payload or {}
        cid = pl.get("chunk_id")
        if cid and cid not in have:
            seat_score = min(tau * 0.9, out[-1].get("score", 0))
            return out[:-1] + [_text_dict(pl, score=seat_score)]
    return out


def _apply_source_scope(ranked, scope):
    """重排分软收窄:目标文档块乘 SOURCE_SCOPE_BOOST 后重排序。scope 为空原样返回。"""
    if not scope or not ranked:
        return ranked
    boost = float(getattr(C, "SOURCE_SCOPE_BOOST", 1.35))
    rescored = [(p, s * boost
                 if _base_stem((p.payload or {}).get("source_stem", "")) == scope
                 else s)
                for p, s in ranked]
    rescored.sort(key=lambda x: x[1], reverse=True)
    return rescored


def _scope_conflict(query, ranked, top_n=4):
    """机械冲突信号(轮次9 反问触发依据,零额外检索、不改变检索结果):
    top-N 按文档族(base_stem)分族,族数≥2 且最高两族分差 ≤ CONFLICT_SCORE_MARGIN
    (gemma raw logits 量级),且词面推断未确信指向榜首族(None 或指向别族——
    后者即"确信地锁错"疑点)→ 返回候选族 base_stem 列表(≤4,作反问选项)。"""
    fams = {}
    for p, s in list(ranked)[:top_n]:
        base = _base_stem((p.payload or {}).get("source_stem", ""))
        if base:
            fams[base] = max(fams.get(base, 0.0), float(s))
    if len(fams) < 2:
        return []
    ordered = sorted(fams.items(), key=lambda kv: -kv[1])
    if ordered[0][1] - ordered[1][1] > float(getattr(C, "CONFLICT_SCORE_MARGIN", 1.0)):
        return []
    try:
        inferred = infer_target_stem(query)
    except Exception:  # noqa: BLE001  推断故障按 None 处理(仍有冲突信号)
        inferred = None
    if inferred and inferred == ordered[0][0]:
        return []   # 词面确信指向榜首族:无歧义,不反问
    return [base for base, _ in ordered[:4]]


def _expand_neighbor(d):
    """邻块扩展:同文档 chunk_index+1 的后接块内容截断后拼在本块尾部。
    金标信息常落在命中块相邻页;块数不增不占 K 席位。失败/无后接块原样返回。"""
    try:
        stem, idx = d.get("source_stem") or "", d.get("chunk_index")
        if not stem or idx is None:
            return d
        res = Q.get_client().scroll(
            collection_name=C.TEXT_COLLECTION,
            scroll_filter=Q.models.Filter(must=[
                Q.models.FieldCondition(key="source_stem",
                                        match=Q.models.MatchValue(value=stem)),
                Q.models.FieldCondition(key="chunk_index",
                                        match=Q.models.MatchValue(value=int(idx) + 1)),
            ]),
            limit=1, with_payload=["content"], with_vectors=False)
        pts = res[0] if isinstance(res, tuple) else res.points
        nxt = ((pts[0].payload or {}).get("content", "") or "").strip() if pts else ""
        if nxt:
            cap = int(getattr(C, "RETRIEVAL_NEIGHBOR_CHARS", 500))
            d["content"] = f"{d.get('content', '')}\n[后接块] {nxt[:cap]}"
    except Exception:  # noqa: BLE001  扩展是增强层,任何故障不阻塞检索
        pass
    return d


def _recall_queries(query):
    """召回查询构造(轮次19):原句(术语表替换) + 英文翻译双路(轮次9)。
    替换只打原句、且在翻译之前——译替换后的中文句,库内术语词面直接进翻译;
    翻译输出(纯英文)不再二次替换,en2en 别名归一不碰 LLM 译文
    (实测:归一会把译文里 release/wafer 逆向缩成 RLS/WAF,毁掉翻译路)。"""
    qs = [query]
    if getattr(C, "RETRIEVAL_GLOSSARY_REPLACE", False):
        from . import glossary
        q0 = glossary.apply(query)
        if q0 and q0 != query:
            qs[0] = q0
            logger.info("glossary replace: %r -> %r", query[:60], q0[:60])
    if getattr(C, "RETRIEVAL_QUERY_TRANSLATE", False):
        en = _translate_query(qs[0])
        if en and en != query:
            qs.append(en)
            logger.info("translate dual-path: %r", en[:60])
    return qs or [query]


def search_text(query, k=3, score_ratio=0.4, with_pool=False,
                with_ranked=False, scope_stem="", with_scope_info=False):
    """文本库混合检索(dense+sparse RRF)+ 精确代码兜底召回 + Cross-Encoder 重排。
    截断不再固定 top-k:重排分 ≥RERANK_TAU 才保留(动态 0~MAX_K 块,保底 top-1)。
    with_pool=True:额外返回重排前候选池(评测口径"池内召回"分母用),返回
    {"items": [...], "pool": [{"chunk_id","source_stem","score"}]}——池含送入
    reranker 的全部候选 + sparse_rescue 尾部块(否则经 rescue 返回的金块会
    分子有分母无,召回率虚高)。"""
    recall_k = getattr(C, "RERANK_RECALL_K", 20)
    # 多路召回构造(轮次19):原句 + 翻译路 + 术语表替换;各路 RRF 并入同池。
    # 只进召回不进重排——英文/分区重排已离线证伪无增益
    # (eval/optimization_log/20261007_inpool_rerank_test.md)。
    _t0 = time.perf_counter()
    qs = _recall_queries(query)
    _t1 = time.perf_counter()
    fused = Q.query_text(qs[0], k=recall_k)
    _t2 = time.perf_counter()
    for extra in qs[1:]:
        fused = _rrf_merge(fused, Q.query_text(extra, k=recall_k),
                           w_b=float(getattr(C, "TRANSLATE_RRF_WEIGHT", 0.7)))
    _t3 = time.perf_counter()
    # 精确报警/故障代码兜底:含代码词面的块常排在 RRF 池外,显式并入候选,
    # 再统一交给 reranker 按相关性重排。RETRIEVAL_EXACT_CODE=0 关闭(实验:
    # 主路修复后是否还需要独立兜底通道)。
    exact = (Q.exact_code_recall(query, limit=max(k, 6))
             if getattr(C, "RETRIEVAL_EXACT_CODE", True) else [])
    exact_ids = {p.id for p in exact}
    # score_ratio 阈值只约束向量融合召回部分(兜底块原始 sparse 分偏低,不参与)。
    # 级联模式跳过 ratio 压池:m3 粗筛直接吃全量召回池(96→24 才有 ÷4 收益,
    # 32→24 仅 ÷1.33);压池职责移交粗筛(fail-open:粗筛失败退全量精排)。
    if getattr(C, "RETRIEVAL_RERANK_CASCADE", False):
        filtered = list(fused)
    else:
        filtered = _ratio_pool(fused, score_ratio)
    have = {p.id for p in filtered}
    filtered = [p for p in exact if p.id not in have] + filtered
    if not filtered:
        return []
    # 池=送入重排的候选(ratio 过滤+保底,再并入 exact 兜底);在重排前记录
    pool_pts = list(filtered)
    # 重排候选取 RERANK_MAX_K×2:多出的尾部给 _doc_cap_select 留跨文档回填空间
    rank_k = max(int(getattr(C, "RERANK_MAX_K", 6)) * 2, k)
    if with_ranked:
        rank_k = len(filtered)     # 扫描模式:全量候选排名(τ/K 离线扫描用)
    ranked = _rerank(query, filtered,
                     k=rank_k,
                     text_fn=_text_rerank_text)
    _t4 = time.perf_counter()
    if os.getenv("RETRIEVAL_PHASE_TIMING"):
        logger.info(
            "phase timing: recall_q=%.2fs q1=%.2fs extra=%.2fs rerank=%.2fs total=%.2fs nq=%d",
            _t1 - _t0, _t2 - _t1, _t3 - _t2, _t4 - _t3, _t4 - _t0, len(qs))
    ranked_pre_scope = list(ranked)   # 冲突判定用自然重排序(boost 前)
    # 检索域收窄:仅 scope_stem(用户反问选定文档族后的会话锁定,轮次9/12)
    # 直接加权——查询时的词面推断收窄已删(20261008 轮次12):反问机制上线后
    # 歧义交由用户消解,词面自动推断属重复防线且有"确信地锁错"风险
    # (dev_067 类),infer_target_stem 保留仅供冲突信号判定使用。
    scope = None
    if scope_stem:
        scope = _base_stem(scope_stem)
        if scope:
            ranked = _apply_source_scope(ranked, scope)
            logger.info("source scope: %s (q=%r)", scope[:40], (query or "")[:40])
    conflict_stems: list[str] = []
    if with_scope_info:
        # 冲突信号在 scope boost 之前、按自然重排序判定(轮次9 反问触发依据)
        try:
            conflict_stems = _scope_conflict(query, ranked_pre_scope)
        except Exception:  # noqa: BLE001  信号层故障不影响检索结果
            logger.warning("scope conflict detect failed", exc_info=True)
    kept = _dynamic_cut(ranked, k)
    # τ/K 离线扫描用:返回截断前全部候选的重排分(scope boost 后、doc_cap 前;
    # rescue 未并入)。生产链路不传此参,零影响。
    if with_ranked:
        ranked_out = [{"chunk_id": (p.payload or {}).get("chunk_id", ""),
                       "source_stem": (p.payload or {}).get("source_stem", ""),
                       "score": round(float(s), 4)} for p, s in ranked]
    cap = int(getattr(C, "RETRIEVAL_DOC_CAP", 0) or 0)
    if cap > 0 and not _reranker_failed:
        kept = _doc_cap_select(kept, ranked, cap)
    out = [_text_dict(p.payload or {}, score=s) for p, s in kept]
    # 邻块扩展:top-N 命中块带后接块内容,扩大上下文覆盖面(金标常在相邻页)
    if getattr(C, "RETRIEVAL_NEIGHBOR_EXPAND", False):
        for d in out[:max(0, int(getattr(C, "RETRIEVAL_NEIGHBOR_TOPN", 2)))]:
            _expand_neighbor(d)
    # sparse 保底:重排器对表格/码表块的词面强命中系统性打低分(run20 id10 探针),
    # τ 截断把它们整块丢掉。非固定模式:追加到结果尾部(chunk_id 去重,评分压在
    # τ 之下);固定 top-K 模式(方案一):RETRIEVAL_RESCUE_SEAT 开启时保留一个
    # "弱末位让位"补录席位,返回块数仍恰为 RERANK_FIXED_K。
    if getattr(C, "RETRIEVAL_SPARSE_RESCUE", True) and not _reranker_failed:
        rescued = Q.sparse_rescue(query, limit=int(getattr(C, "RETRIEVAL_SPARSE_RESCUE_K", 2)))
        pool_pts.extend(rescued)
        if int(getattr(C, "RERANK_FIXED_K", 0) or 0):
            if getattr(C, "RETRIEVAL_RESCUE_SEAT", False):
                out = _rescue_seat(out, rescued)
        else:
            have = {d["chunk_id"] for d in out}
            rescue_score = float(getattr(C, "RERANK_TAU", 0.4)) * 0.9
            for p in rescued:
                cid = (p.payload or {}).get("chunk_id")
                if cid and cid not in have:
                    out.append(_text_dict(p.payload or {}, score=rescue_score))
    if with_scope_info:
        # 轮次9:附带锁定/冲突信号(chat 层反问依据;检索结果本身不变)
        # 轮次20(反问无选项化):附带词面推断 stem——应答轮用户自由文本
        # (设备名/型号描述)靠它消解为文档族,注册会话 scope 锁。
        try:
            inferred = infer_target_stem(query) or ""
        except Exception:  # noqa: BLE001  信号层故障不影响检索结果
            inferred = ""
        scope_info = {"locked_stem": scope or "",
                      "conflict_stems": conflict_stems,
                      "inferred_stem": inferred}
        if with_pool:
            seen = set()
            pool = []
            for p in pool_pts:             # 去重保序(exact 兜底块可能与融合池重复)
                cid = (p.payload or {}).get("chunk_id")
                if cid and cid not in seen:
                    seen.add(cid)
                    pl = p.payload or {}
                    pool.append({"chunk_id": cid,
                                 "source_stem": pl.get("source_stem", ""),
                                 "score": round(float(p.score or 0), 4)})
            if with_ranked:
                return {"items": out, "pool": pool, "ranked": ranked_out,
                        "scope": scope_info}
            return {"items": out, "pool": pool, "scope": scope_info}
        if with_ranked:
            return {"items": out, "ranked": ranked_out, "scope": scope_info}
        return {"items": out, "scope": scope_info}
    if with_pool:
        seen = set()
        pool = []
        for p in pool_pts:                 # 去重保序(exact 兜底块可能与融合池重复)
            cid = (p.payload or {}).get("chunk_id")
            if cid and cid not in seen:
                seen.add(cid)
                pl = p.payload or {}
                pool.append({"chunk_id": cid,
                             "source_stem": pl.get("source_stem", ""),
                             "score": round(float(p.score or 0), 4)})
        if with_ranked:
            return {"items": out, "pool": pool, "ranked": ranked_out}
        return {"items": out, "pool": pool}
    if with_ranked:
        return {"items": out, "ranked": ranked_out}
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
