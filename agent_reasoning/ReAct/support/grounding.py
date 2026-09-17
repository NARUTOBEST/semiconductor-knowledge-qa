# -*- coding: utf-8 -*-
"""终答 grounding 后置校验 + 置信度门控 + 熔断器(全部以检索文档为唯一标准)。

层1 数字/名称硬校验(纯规则,毫秒级):抽取答案句中的阿拉伯数字并归一化
    (去千分位逗号、万/千换算),逐一与检索资料中的数字集合比对——资料里
    没有该数字的句子直接判 false,不进 LLM。参数知识无法介入,专治
    "资料写12,000答案写4,000"类幻觉。
层2 引文强制校验(LLM,一次调用):对层1放过的句子判"能否被资料蕴含",
    判 true 必须同时给出资料中支撑该句的最短原文片段(quote);拿到结果后
    用字符串匹配验证 quote 真实存在于资料原文——引文给不出或对不上,
    一律按 false 处理。"以文档为准"由机制保证而非由指令嘱咐。
层3 置信度门控:置信度 = 校验通过句占比。
    - 达标(≥ GROUNDING_MIN_CONFIDENCE,默认0.6):删掉无支撑句后下发答案。
    - 不达标(< 线):不下发模型答案,整段替换为"引导人工翻阅手册/联系
      技术支持"的提示文本(info["action"]="guidance")——低置信答案宁可
      不答,也不把可能幻觉的内容交给前端。

熔断状态机(防校验调用本身拖垮链路):
- CLOSED  正常放行:校验调用走超时+重试(复用 llm_create_with_retry 的指数退避
          与备用模型切换);连续失败达阈值 → 熔断 OPEN。
- OPEN    校验不可用 ≠ 低置信:跳过校验放行原文,下发"LLM调用失败"提示事件
          (可用性降级与内容置信度是两回事),冷却期内不再尝试。
- HALF_OPEN 冷却结束:放行一个试探请求,成功 → CLOSED,失败 → 回 OPEN。
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time

import config as C

logger = logging.getLogger("agent.grounding")

# ---- 熔断器参数(config 可覆盖) ----
BREAKER_THRESHOLD = int(getattr(C, "GROUNDING_BREAKER_THRESHOLD", 3))   # 连续失败次数
BREAKER_COOLDOWN = float(getattr(C, "GROUNDING_BREAKER_COOLDOWN", 60))  # OPEN 冷却秒数

# 熔断器进程级单例(线程安全;backend 为多线程并发请求)
_LOCK = threading.Lock()
_STATE = {"state": "closed", "fails": 0, "opened_at": 0.0}


def breaker_allow() -> tuple[bool, str]:
    """是否允许发起校验调用。返回 (allowed, state)。"""
    with _LOCK:
        st = _STATE
        if st["state"] == "open":
            if time.time() - st["opened_at"] >= BREAKER_COOLDOWN:
                st["state"] = "half_open"  # 冷却结束,放一个试探请求
            else:
                return False, "open"
        return True, st["state"]


def breaker_record(success: bool) -> str:
    """记录一次校验调用结果,返回熔断后状态。"""
    with _LOCK:
        st = _STATE
        if success:
            st.update(state="closed", fails=0)
        else:
            st["fails"] += 1
            if st["state"] == "half_open" or st["fails"] >= BREAKER_THRESHOLD:
                st.update(state="open", opened_at=time.time())
        return st["state"]


# ---- 句子拆分 ----

_SPLIT_RE = re.compile(r"(?<=[。!?;；\n])")


def _split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SPLIT_RE.split(text or "") if p.strip()]
    return parts or [text.strip()] if (text or "").strip() else []


# ---- 层1:数字硬校验(纯规则) ----

# 阿拉伯数字 + 可选千分位逗号 + 可选小数 + 可选 万/千 后缀
_NUM_TOKEN_RE = re.compile(r"(\d[\d,，]*(?:\.\d+)?)\s*([万千]?)")
# 引文/比对时的标点与空白归一化
_PUNCT_RE = re.compile(r"[\s，。、;；:：,\.\"'“”‘’「」『』【】\[\]()（）·—\-]")


def _norm_key(s: str) -> str:
    return _PUNCT_RE.sub("", s or "")


def _extract_numbers(text: str) -> set[str]:
    """抽取文本中所有阿拉伯数字表述,归一化为可比对的标准数字串。

    "4,000"→"4000","1.2万"→"12000","3千"→"3000";个位数(0-9)忽略——
    参与价值低("步骤1"/"P1")而误杀成本高。
    """
    out: set[str] = set()
    for m in _NUM_TOKEN_RE.finditer(text or ""):
        raw, unit = m.group(1), m.group(2)
        try:
            val = float(raw.replace(",", "").replace("，", ""))
        except ValueError:
            continue
        if unit == "万":
            val *= 10000
        elif unit == "千":
            val *= 1000
        if val < 10:
            continue
        out.add(str(int(val)) if val == int(val) else str(val))
    return out


def _layer1_check(sents: list[str], src_nums: set[str]) -> tuple[list[bool], int]:
    """逐句判数字是否全部有据。返回 (每句是否层1通过, 层1判死数)。"""
    verdicts, killed = [], 0
    for s in sents:
        bad = _extract_numbers(s) - src_nums
        if bad:
            killed += 1
            logger.info("grounding L1 kill: nums=%s not in sources", sorted(bad))
        verdicts.append(not bad)
    return verdicts, killed


# ---- 主流程 ----

# 诚实拒答/未找到类回答:本身就是"资料没有"的如实陈述,不是事实断言,
# 无需资料一致性校验(用低分杂讯资料校验它必然 0 分,反而被误替换为引导)。
_NOT_FOUND_RE = re.compile(
    r"(未(能|曾|没有)?(找到|检索到|命中|查到|提及|包含|发现|覆盖)"
    r"|没有找到|查不到|暂未?找到|暂无(相关)?(资料|内容|信息|记录)"
    r"|资料(中|里)?未?(提及|包含|涉及|找到)|无法(找到|确认|回答)"
    r"|没有(相关|对应|匹配)的?[一-龥]{0,4}?(资料|内容|信息|记录))")


def grounding_filter(answer: str, sources: list[dict], *, trace_id: str = ""):
    """对终答做三层 grounding 校验,返回 (filtered_answer, info)。

    info: {"enabled","state","checked","removed","elapsed_ms","error",
           "l1_removed","quote_failed","confidence","action"}
    action: "guidance"(低置信整段替换)| "pruned"(删句)| "passthrough"
    - 校验调用失败(重试+备用模型耗尽)→ 熔断记失败;OPEN 期间跳过校验放行原文,
      并由调用方下发"LLM调用失败"降级提示。
    - 解析失败/答案过短(<2 句)等一律放行原文,校验只删句不造句。
    """
    info = {"enabled": True, "state": "closed", "checked": 0, "removed": 0,
            "elapsed_ms": 0, "error": None, "l1_removed": 0, "quote_failed": 0}
    if _NOT_FOUND_RE.search(answer or ""):
        info["enabled"] = False
        info["action"] = "passthrough_notfound"
        logger.info("grounding skip: honest not-found answer, no check needed")
        return answer, info
    sents = _split_sentences(answer)
    if len(sents) < 2:  # 一句话答案没有可删的,别为它多烧一次调用
        info["enabled"] = False
        return answer, info

    contexts = []
    # 低于重排 τ 的杂讯资料不进校验池:模型答"资料未找到/凭会话记忆"时,
    # 用杂讯校验必然全句 unsupported → 0.0 → 被误替换为引导(实测轮2复现)。
    _tau = float(getattr(C, "RERANK_TAU", 0.3))
    for s in (sources or []):
        try:
            if float((s or {}).get("score") or 0.0) < _tau:
                continue
        except (TypeError, ValueError):
            pass
        content = str((s or {}).get("content", "")).strip()
        if content:
            contexts.append(content[:700])
        if len(contexts) >= 3:  # top-3×700 字:控制 prefill,校验调用压在 ~2s 内
            break
    if not contexts:
        info["enabled"] = False
        info["action"] = "passthrough_no_qualified_sources"
        logger.info("grounding skip: no source above tau %.2f (memory/context answer?)",
                    _tau)
        return answer, info

    # ---- 层1:数字硬校验(不调模型,与 LLM 校验独立) ----
    src_nums: set[str] = set()
    for c in contexts:
        src_nums |= _extract_numbers(c)
    l1_ok, l1_killed = _layer1_check(sents, src_nums)
    info["l1_removed"] = l1_killed

    ctx_blob = "\n\n".join(f"[资料{i+1}] {c}" for i, c in enumerate(contexts))
    todo_idx = [i for i in range(1, len(sents) + 1) if l1_ok[i - 1]]

    keep: dict[int, bool] = {}
    if todo_idx:
        # 层1 没有全军覆没:幸存句进 LLM 引文强制校验
        numbered = "\n".join(f"{i}. {sents[i-1]}" for i in todo_idx)
        allowed, st = breaker_allow()
        info["state"] = st
        if not allowed:
            info["error"] = "LLM调用失败"  # OPEN 降级
            return answer, info

        t0 = time.time()
        data = None
        err = None
        try:
            from .llm import get_client, llm_create_with_retry, no_think_extra
            import os
            # 校验模型:默认 GPU main(Qwen3 本地 ~0.4s/次,无 ark 外网 RTT);
            # GROUNDING_MODEL 可覆盖(如切回轻云模型 doubao-seed-2.0-lite,单次 ~2-4s)。
            # 别名解析:"light"/"main" 是网关别名,codingplan 云端不认(404),
            # 先映射为 env 里的真实模型名再发。
            _gmodel = (os.getenv("GROUNDING_MODEL", "").strip()
                       or getattr(C, "MODEL_MAIN", "") or getattr(C, "MODEL_LIGHT", "light"))
            _alias = {"light": str(getattr(C, "MODEL_LIGHT", "") or ""),
                      "main": str(getattr(C, "MODEL_MAIN", "") or "")}
            _gmodel = _alias.get(_gmodel.lower(), _gmodel)
            resp, cerr = llm_create_with_retry(
                get_client(), trace_id=trace_id, retries=1,  # 失败快速让位:熔断器兜底,不拖垮延迟
                model=_gmodel,
                messages=[
                    {"role": "system", "content": "你是事实核查员。只输出 JSON。"},
                    {"role": "user", "content":
                        "下面是【回答】拆成的编号句子与检索【资料】。逐句判断该句能否由"
                        "资料内容蕴含(supported=true/false)。判断标准:同义改写、直接推"
                        "论、合理具体化都算 true;资料无法推出、与资料矛盾、或资料完全"
                        "没提的机理/原因解释/通用建议算 false。"
                        "判 true 必须同时给出资料中支撑该句的最短原文片段(quote),quote"
                        "必须是资料原文的连续片段;判 false 不需要 quote。\n"
                        f"【回答句子】\n{numbered}\n\n【资料】\n{ctx_blob}\n\n"
                        '输出格式:{"verdicts":[{"i":1,"supported":true,'
                        '"quote":"资料原句"}]}'},
                ],
                temperature=0.0, timeout=float(getattr(C, "GROUNDING_TIMEOUT", 12)),
                max_tokens=400, **no_think_extra(),
            )
            if cerr is not None:
                err = cerr
            else:
                text = (resp.choices[0].message.content or "").strip()
                m = re.search(r"\{.*\}", text, re.S)
                data = json.loads(m.group(0) if m else text)
        except Exception as e:  # noqa: BLE001
            err = e

        info["elapsed_ms"] = int((time.time() - t0) * 1000)
        if err is not None or not isinstance(data, dict) \
                or not isinstance(data.get("verdicts"), list):
            new_state = breaker_record(False)
            info["state"] = new_state
            info["error"] = f"LLM调用失败: {str(err)[:120]}" if err else "LLM调用失败: 解析失败"
            logger.warning("grounding check failed, breaker=%s: %s",
                           new_state, str(info["error"]))
            return answer, info

        breaker_record(True)

        # ---- 层2后处理:引文机械验证 ----
        # 判 true 必须给出资料原文中真实存在的 quote;给不出/对不上一律按 false。
        ctx_norm = _norm_key(ctx_blob)
        for v in data["verdicts"]:
            try:
                i = int(v.get("i", 0))
            except (TypeError, ValueError):
                continue
            supported = bool(v.get("supported", True))
            if supported:
                quote = _norm_key(str(v.get("quote") or ""))
                if not quote or quote not in ctx_norm:
                    supported = False
                    info["quote_failed"] += 1
                    logger.info("grounding L2 quote failed (i=%s): %.40s", i, quote)
            keep[i] = supported

    # 汇总每句最终判定:层1判死 → false;层1放过 → 以层2为准(缺省 true)
    supported = [l1_ok[i] and keep.get(i + 1, True) for i in range(len(sents))]
    return _apply_fuse_and_delete(answer, sents, supported, info, sources)


def _guidance_text(sources: list[dict]) -> str:
    """低置信替换文本:引导人工翻阅手册(带来源文档/页码)或联系技术支持。"""
    refs = []
    seen: set[str] = set()
    for s in (sources or []):
        stem = str((s or {}).get("source_stem", "")).strip()
        if not stem or stem in seen:
            continue
        seen.add(stem)
        page = (s or {}).get("page") or (s or {}).get("page_num") \
            or (s or {}).get("page_start")
        refs.append(f"《{stem}》" + (f" 第{page}页" if page else ""))
        if len(refs) >= 3:
            break
    ref_line = ("相关资料:" + "、".join(refs)) if refs else "相关随机资料"
    return (
        "抱歉,该问题的回答未能通过资料一致性校验,置信度不足,为避免误导,"
        "暂不提供自动回答。建议您:\n"
        f"1. 直接翻阅随机资料中的对应章节自行核对({ref_line});\n"
        "2. 或联系设备技术支持(FAE)获取人工确认。"
    )


def _apply_fuse_and_delete(answer: str, sents: list[str],
                           supported: list[bool], info: dict,
                           sources: list[dict]):
    """层3置信度门控 + 删句。

    置信度 = 通过句占比:达标 → 删无支撑句后下发;不达标 → 整段替换为
    人工引导文本(低置信答案不下发前端)。"""
    _MIN_CONF = float(getattr(C, "GROUNDING_MIN_CONFIDENCE", 0.6))
    n = len(sents)
    n_false = n - sum(supported)
    info["checked"] = n
    confidence = (n - n_false) / n if n else 0.0
    info["confidence"] = round(confidence, 3)
    if n and n_false / n > float(getattr(C, "GROUNDING_REMOVAL_CAP", 0.4)):
        # 判 false 句占比超 REMOVAL_CAP ⇔ 置信度低于 1-CAP:无论归因于
        # 校验链路异常还是答案本身大面积无支撑,都不再猜——直接引导人工。
        info["removed"] = 0
        info["cap_hit"] = True
        info["action"] = "guidance"
        logger.warning("grounding confidence %.2f < %.2f (%d/%d unsupported), "
                       "answer replaced with manual guidance",
                       confidence, 1.0 - float(getattr(C, "GROUNDING_REMOVAL_CAP", 0.4)),
                       n_false, n)
        return _guidance_text(sources), info

    out, removed = [], 0
    for s, ok in zip(sents, supported):
        if ok:
            out.append(s)
        else:
            removed += 1
    info["removed"] = removed
    info["action"] = "pruned" if removed else "passthrough"
    return ("".join(out) if removed else answer), info
