# -*- coding: utf-8 -*-
"""轻量模型上下文压缩 —— 召回 chunk 先抽关键句再回灌 LLM。

与 truncate_tool_result(同目录)的关系:压缩是截断的上游增强——召回类工具
结果先经轻模型按问题相关性抽关键句;任何失败(熔断/超时/解析失败)都退回
原截断路径,绝不阻断问答。与 grounding.py 同一套失败哲学:
  - 熔断 closed/open/half-open:连续失败达阈值即 OPEN 直接拒绝,冷却到期后半开
    放行单次试探,成功恢复 closed。压缩属锦上添花,OPEN 时秒回退;
  - 超时 + retries=1:压缩调用不重试,不放大问答延迟;
  - 抽出的句子做原文子串机械校验,轻模型幻觉句直接丢弃——进上下文的每个字
    都能在原文里找到出处。
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout

import config as C

logger = logging.getLogger(__name__)

_PROMPT = (
    "你是检索结果压缩器。给定用户问题和若干检索片段,从每个片段中挑出与回答"
    "该问题直接相关的关键句(逐字摘录原文,不得改写/补全/推断)。\n"
    "规则:\n"
    '1. 只输出 JSON: {{"items": [{{"i": 片段编号, "s": ["关键句1", "关键句2"]}}]}};\n'
    "2. 每个片段最多摘 3 句,优先保留含数字、参数、单位、报警码、型号、结论的句子;\n"
    "3. 片段与问题无关时给空数组;\n"
    "4. 禁止输出片段中不存在的句子。\n\n"
    "用户问题:{question}\n\n{chunks}"
)

# 摘句校验前先归一化:去空白/中英文标点差异,避免整句因标点被误判为幻觉
_NORM_RE = re.compile(r'''[\s,.;:!?"、,。;:!?'“”‘’【】\[\]()（）]+''')

# 熔断器:2026-10-10 起并入全链路 LLM 三状态熔断(tool_circuit.get_llm_breaker,
# 按环节 "compress" 独立计数),替代原先本文件内的手写状态机——参数统一
# (阈值 3/冷却 10s/连续 2 次半开试探失败进放弃期),放弃期降级为 truncate。
from agent_reasoning.ReAct.support.tool_circuit import (  # noqa: E402
    CircuitOpenError, get_llm_breaker,
)

# 硬期限执行器:httpx 的 read timeout 只约束"字节间隙",ark 对大 prompt 的
# 涓流式响应可以无限拖(实测 40s 不触发 12s 超时),必须在线程外强制 deadline。
# 超时后工作线程自会跑完并丢弃结果(daemon 语义,不阻塞主链路)。
_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="compress")


def _norm(s: str) -> str:
    return _NORM_RE.sub("", str(s or ""))


def _parse(text: str, chunks: list[dict]) -> list[str] | None:
    """解析轻模型输出并做原文子串校验。返回与 chunks 等长的最终内容列表;
    未抽到句的块回退 content[:KEEP_CHARS]。结构解析失败返回 None。"""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(text[start:end + 1])
    except Exception:
        return None
    items = obj.get("items") if isinstance(obj, dict) else None
    if not isinstance(items, list):
        return None

    picked: dict[int, list[str]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            idx = int(item.get("i")) - 1
        except Exception:
            continue
        if 0 <= idx < len(chunks):
            picked[idx] = item.get("s") if isinstance(item.get("s"), list) else []

    out = []
    for idx, c in enumerate(chunks):
        content = str(c.get("content") or "")
        ncontent = _norm(content)
        keep, seen = [], set()
        for s in picked.get(idx, []):
            s = str(s or "").strip()
            ns = _norm(s)
            if not s or ns in seen:
                continue
            seen.add(ns)
            if ns in ncontent:  # 子串校验:幻觉句直接丢弃
                keep.append(s)
        out.append("\n".join(keep) if keep
                   else content[:C.CONTEXT_COMPRESS_KEEP_CHARS])
    return out


def compress_chunks(chunks: list[dict], question: str, *,
                    trace_id: str = "", min_chars: int | None = None) -> list[str] | None:
    """对召回 chunk 列表做关键句压缩。

    :param chunks: dict 列表,需含 ``content``;可用 ``source_stem``/``page`` 作标注。
    :param min_chars: 压缩门槛(回灌总字数低于此不调用轻模型);None 用
        CONTEXT_COMPRESS_MIN_CHARS。raglite 快路径传更高的
        CONTEXT_COMPRESS_RAGLITE_MIN_CHARS(一次轻模型调用对 ~10s 的单轮
        问答是明显延迟,仅体量显著大时才值得)。
    :returns: 与输入等长的最终内容字符串列表(抽到的关键句按原文换行拼接;
              未抽到句的块回退 content[:CONTEXT_COMPRESS_KEEP_CHARS])。
              整体不可用(开关关闭/体量不足/熔断/调用失败/解析失败)返回 None,
              调用方回退 truncate_tool_result 原截断路径。
    """
    if not getattr(C, "CONTEXT_COMPRESS_ENABLED", True):
        return None
    try:
        # 子问题席位绑定(20261014):sub_pin 块不受 MAX_CHUNKS 配额截断,追加在
        # 常规配额之后参与压缩;返回列表与调用方输入保持等长对齐(未入选块回退截断)。
        full = [c for c in (chunks or [])
                if isinstance(c, dict) and str(c.get("content") or "").strip()]
        cap = C.CONTEXT_COMPRESS_MAX_CHUNKS
        keep = [i for i, c in enumerate(full) if not c.get("sub_pin")][:cap]
        keep_set = set(keep)
        keep += [i for i, c in enumerate(full)
                 if c.get("sub_pin") and i not in keep_set]
        picked = _compress([full[i] for i in keep], question,
                           trace_id=trace_id, min_chars=min_chars)
        if picked is None or len(keep) == len(full):
            return picked
        out = [""] * len(full)
        for i, sub in zip(keep, picked):
            out[i] = sub
        for i, c in enumerate(full):
            if not out[i]:
                out[i] = str(c.get("content") or "")[:C.CONTEXT_COMPRESS_KEEP_CHARS]
        return out
    except Exception:  # noqa: BLE001  压缩链路任何意外都不允许打断问答
        logger.exception("compress unexpected error, fallback truncate (trace=%s)",
                         trace_id)
        return None


def _compress(chunks: list[dict], question: str, *,
              trace_id: str = "", min_chars: int | None = None) -> list[str] | None:
    chunks = [c for c in (chunks or [])
              if isinstance(c, dict) and str(c.get("content") or "").strip()]
    if not chunks:
        return None
    chunks = chunks[:C.CONTEXT_COMPRESS_MAX_CHUNKS]
    total = sum(len(str(c.get("content") or "")) for c in chunks)
    if total < (C.CONTEXT_COMPRESS_MIN_CHARS if min_chars is None else min_chars):
        logger.info("compress skip: %d chunks %d chars < threshold (trace=%s)",
                    len(chunks), total, trace_id)
        return None  # 体量小,不值得一次轻模型调用
    _brk = get_llm_breaker("compress")
    if _brk is not None:
        try:
            _brk.before_call()
        except CircuitOpenError:
            logger.info("compress breaker OPEN, skip (trace=%s)", trace_id)
            return None  # OPEN/放弃期零开销降级:truncate 兜底

    parts = []
    for i, c in enumerate(chunks, 1):
        head = str(c.get("source_stem") or f"片段{i}")
        if c.get("page"):
            head += f" p{c['page']}"
        parts.append(f"[片段{i}] {head}\n"
                     f"{str(c['content'])[:C.CONTEXT_COMPRESS_MAX_INPUT_CHARS]}")
    prompt = _PROMPT.format(question=(question or "").strip()[:300],
                            chunks="\n\n".join(parts))

    t0 = time.time()
    try:
        # 直连单次调用,不走 llm_create_with_retry:压缩是可选增强,重试/failover
        # 会把最坏延迟放大到 2×timeout+(实测 39s)烧穿端到端预算;失败交给熔断器。
        from agent_reasoning.ReAct.support.llm import get_client, no_think_extra
        client = get_client()
        kwargs = dict(
            model=C.CONTEXT_COMPRESS_MODEL or C.TIER_MODEL_SIMPLE,
            messages=[{"role": "user", "content": prompt}],
            temperature=0, max_tokens=500,
            timeout=C.CONTEXT_COMPRESS_TIMEOUT,
            **no_think_extra(),
        )
        fut = _POOL.submit(client.chat.completions.create, **kwargs)
        try:
            resp = fut.result(timeout=C.CONTEXT_COMPRESS_TIMEOUT)
        except FuturesTimeout:
            raise RuntimeError(
                f"compress hard deadline {C.CONTEXT_COMPRESS_TIMEOUT}s exceeded"
            ) from None
        text = (resp.choices[0].message.content or "").strip()
    except Exception as e:  # noqa: BLE001
        if _brk is not None:
            _brk.on_failure("timeout")
        logger.warning("compress call failed, breaker=%s, fallback truncate "
                       "(trace=%s): %s",
                       _brk.state if _brk else "off", trace_id, str(e)[:120])
        return None

    picked = _parse(text, chunks)
    if picked is None:
        if _brk is not None:
            _brk.on_failure("timeout")
        logger.warning("compress parse failed, breaker=%s (trace=%s)",
                       _brk.state if _brk else "off", trace_id)
        return None
    if _brk is not None:
        _brk.on_success()
    logger.info("compress ok: %d chunks %d->%d chars %.2fs breaker=%s (trace=%s)",
                len(chunks), total, sum(len(s) for s in picked),
                time.time() - t0, _brk.state if _brk else "off", trace_id)
    return picked
