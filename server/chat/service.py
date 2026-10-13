# -*- coding: utf-8 -*-
"""知识助手 -- 聊天服务入口(薄封装)。

本文件只做一件事:提供 react_stream(),先做复杂度路由(classify_complexity),
按 tier 分发到 simple(单轮直答)/ react(ReAct 检索循环),并在答案产出后、
``done`` 事件前过共享质检门(quality_gate),按 verdict 放行 / 重做一次 /
simple→react 升级一次。

具体的 ReAct 主循环、LLM 调用、工具调用全部在 agent_reasoning 包内。

为兼容现有调用方(health/service.py 及若干测试以 chat.service.get_client 等
路径导入/打桩),这里重导出 LLM 相关符号。
"""
from __future__ import annotations

import json
import re
import time
import uuid
import logging
from typing import Callable, Optional

from agent_reasoning.ReAct.support.runner import run_agent_graph
from agent_reasoning.simple import run_simple
from agent_reasoning.raglite import run_raglite
from agent_reasoning.ReAct.support.llm import (
    get_client, llm_create_with_retry, LLM_RETRIES, no_think_extra)
from agent_reasoning.router import classify_complexity, classify_and_clarify, _obviously_simple
from agent_reasoning.clarify import check_clarify
from agent_reasoning.quality_gate import check as quality_check
import config as C

logger = logging.getLogger("agent")

__all__ = [
    "react_stream",
    "get_client",
    "llm_create_with_retry",
    "LLM_RETRIES",
]

# 升级链:simple/raglite(快路径) -> react(检索问答);react 已为最高 tier。
_NEXT_TIER = {"simple": "react", "raglite": "react", "react": None}
_MAX_ESCALATIONS = 1   # 整条请求最多升级一次
_MAX_REDOS = 1         # 同一 tier 质检 failed 最多重做一次

# run_simple 不认这些 ReAct 预算/重做参数,分发前剔除。
_NON_SIMPLE_KW = ("max_steps", "max_total_seconds", "hard_deadline",
                  "qc_feedback", "search_future", "scope_stem")


# ---- 轮次9:检索冲突反问 + 会话 scope 锁(轮次12 改版)----
# 机械冲突信号命中后,先经 lite LLM 判定问题是否已指明具体对象:已指明则
# 不反问照常作答(轮次11 dev200 实测"我想查砷中毒的处理流程"这类具体问题
# 被相似手册挤占 top-K 误伤反问);确实没指明才反问,且选项为从 top-4 候选
# 块内容提取的**对象名**(设备/机型/物料等,用户看得懂),绝不出现文档文件名
# ——用户无法判断该选哪份手册。每个选项携带其所属文档族 stem(option_stems)。
# 轮次12:scope 锁退役为纯"会话已消解"标记——锁定期间不再反问,检索一律
# 正常翻译双路(原 query+英文翻译 RRF),不再做文档族强加权(轮次9 的
# scope_stem 注入已全部撤除);应答轮检索用原问题(选项只是消歧信号,
# 不是信息需求;详见 _scope_resolve)。
# key = (username, 会话首条用户消息):前端反问应答轮 history 首条用户消息不变,
# 天然按会话隔离;TTL 兜底防泄漏。
_SCOPE_PENDING: dict = {}   # key -> {"choices": [(label, stem)...], "stems": [...], "ts", "message"}
_SCOPE_LOCKS: dict = {}     # key -> {"stem": base_stem, "ts"}   # 仅标记已消解
_SCOPE_TTL_S = 30 * 60
_CONFLICT_CLARIFY_QUESTION = "你想了解的具体对象是下面哪一个?(也可以直接补充说明)"


def _scope_key(username, history, message):
    """会话键:取 history 首条用户消息(多轮中恒定),无历史用当前消息。"""
    first = ""
    for h in (history or []):
        role = h.get("role") if isinstance(h, dict) else getattr(h, "role", "")
        if role == "user":
            first = (h.get("content") if isinstance(h, dict)
                     else getattr(h, "content", "")) or ""
            break
    return (username or "", (first or message or "")[:200])


def _norm_opt(s):
    return re.sub(r"[\s_\-·・()（）]+", "", (s or "").lower())


def _scope_resolve(username, message, history):
    """反问应答轮:消息与待选选项(对象名,或兼容旧口径的文档族 stem)匹配
    → 转正为锁定并返回 (stem, 原问题);不匹配则清除待选。
    锁定活跃(TTL 内)→ 返回锁定 stem;否则空串。
    轮次12:stem 仅作"本会话已消解、不再反问"标记,不注入检索加权。
    原问题仅在本轮刚转正时非空:应答轮的检索 query 必须用原问题而非选项
    文本——选项(对象名/文档名)不是信息需求,拿它当 query 只会命中标题/
    概览块,金事实全 miss(20261008 轮次11 dev200 实测 34/37 反问题
    recall=0 的根因)。"""
    key = _scope_key(username, history, message)
    pend = _SCOPE_PENDING.pop(key, None)
    if pend and time.time() - pend["ts"] <= _SCOPE_TTL_S:
        m = _norm_opt(message)
        if m:
            # 先按对象名匹配(轮次12 选项口径),再按文档族 stem 兜底
            # (用户直接回文档名的情形)
            for label, stem in (pend.get("choices") or []):
                ns = _norm_opt(label)
                if ns and (ns in m or m in ns):
                    _SCOPE_LOCKS[key] = {"stem": stem, "ts": time.time()}
                    return stem, (pend.get("message") or "")
            for stem in pend.get("stems") or []:
                ns = _norm_opt(stem)
                if ns and (ns in m or m in ns):
                    _SCOPE_LOCKS[key] = {"stem": stem, "ts": time.time()}
                    return stem, (pend.get("message") or "")
    lock = _SCOPE_LOCKS.get(key)
    if lock and time.time() - lock["ts"] <= _SCOPE_TTL_S:
        return lock["stem"], ""
    return "", ""


def _conflict_probe(future, timeout=30.0):
    """投机检索 future → (conflict_stems, 候选块列表)。
    conflict_stems 为检索引擎机械冲突信号;候选块(heading/内容片段/source_stem)
    供反问前置 LLM 提取候选对象。
    不取消 future:超时/失败后 raglite/react 仍可正常消费该结果。"""
    try:
        out = future.result(timeout=timeout)
    except Exception:  # noqa: BLE001  信号层故障不阻断主链路
        return [], []
    if isinstance(out, tuple) and len(out) == 2:
        result, err = out
        if err is not None:
            return [], []
        out = result
    if not isinstance(out, dict):
        return [], []
    scope = out.get("scope") or {}
    stems = [s for s in (scope.get("conflict_stems") or []) if s]
    items = out.get("items") or []
    return stems, items


# 冲突反问前置判定(轮次12):机械冲突信号只说明"检索回了多份相似文档",
# 不代表问题本身有歧义——是否反问必须以"问题是否指明具体对象"为准。
_CONFLICT_OBJECTS_PROMPT = (
    "你是公司内部知识问答系统的澄清判定器。用户提出了一个问题,系统检索回了"
    "若干候选资料块(可能来自不同文档/涉及不同对象)。请依次判断:\n"
    "1) 用户问题是否已明确指出了它所问的**对象**(具体设备/机型/物料/工艺/标准等)?"
    "只要能从问题里看出明确对象就视为 specific——即使检索回了多份相似文档也照样"
    " specific,不得因检索结果歧义而反问具体问题。\n"
    "2) 仅当问题确实没指明对象(如只说'怎么保养''这个参数多少',上下文也无法消解)"
    "时,从候选块中提取最多 4 个用户可能想问的**对象**作为选项。对象名必须具体、"
    "用户一眼看得懂(如'重掺砷硅单晶''F200 键合机');**绝对不要用文档文件名/"
    "手册名作为选项**——用户无法凭文件名判断该选哪个。\n"
    "只输出 JSON,不要解释,格式:\n"
    '{{"specific": true/false, "question": "反问句(specific=true 或提不出对象时为空串)",'
    ' "objects": [{{"name": "对象名", "stem": "该对象所在候选块的文档名原文"}}]}}\n\n'
    "用户问题:{question}\n\n候选块:\n{blocks}"
)


def _conflict_objects(message, items):
    """冲突反问前置 LLM 判定。返回 (specific, question, [(对象名, stem)...])。
    specific=问题已指明对象(不反问);objects=候选对象(含所属文档族 stem,
    供点选后 scope 锁定与评测匹配)。
    任何异常/解析失败 fail-open 返回 (True, "", []):照常作答,绝不因反问
    模块故障阻断对话。"""
    blocks = []
    for i, it in enumerate((items or [])[:4], 1):
        stem = str(it.get("source_stem", "") or "")
        head = str(it.get("heading_path", "") or "")
        content = re.sub(r"\s+", " ", str(it.get("content", "") or ""))[:300]
        blocks.append(f"[{i}] (文档:{stem}) 标题:{head}\n{content}")
    if not blocks:
        return True, "", []
    prompt = _CONFLICT_OBJECTS_PROMPT.format(
        question=message, blocks="\n\n".join(blocks))
    try:
        resp, err = llm_create_with_retry(
            get_client(), trace_id="conflict-clarify",
            model=C.TIER_MODEL_SIMPLE,
            messages=[{"role": "user", "content": prompt}],
            temperature=0, max_tokens=300,
            timeout=max(getattr(C, "CLARIFY_TIMEOUT", 8), C.ROUTER_TIMEOUT),
            **no_think_extra(), retries=1)
        if err is not None:
            logger.warning("conflict-clarify LLM failed, skip clarify: %s", err)
            return True, "", []
        text = (resp.choices[0].message.content or "").strip()
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return True, "", []
        obj = json.loads(text[start:end + 1])
        if not isinstance(obj, dict) or bool(obj.get("specific")):
            return True, "", []
        question = str(obj.get("question", "")).strip()
        choices = []
        for o in (obj.get("objects") or [])[:4]:
            if not isinstance(o, dict):
                continue
            name = str(o.get("name", "")).strip()
            stem = str(o.get("stem", "")).strip()
            if name:
                choices.append((name, stem))
        return False, question, choices
    except Exception:  # noqa: BLE001  fail-open:反问层故障不阻断主链路
        logger.warning("conflict-clarify failed, skip clarify", exc_info=True)
        return True, "", []


def _speculative_search(message: str, scope_stem: str = ""):
    """投机检索:在后台工具池预发一次 search_text,与路由/分类重叠执行。
    返回 future(.result() -> (result, err),与 call_with_resilience 同构);
    无人消费时结果自然丢弃(daemon 池,不阻塞收尾)。
    (原低置信触发 LLM 查询改写已删除——轮次5 dev-200 验证四项指标均劣于
    单路且在噪声带内;跨语言鸿沟由检索服务内的翻译双路承担。)
    with_scope_info=True:响应附带 scope 锁定/冲突信号,供冲突反问判定(轮次9)。"""
    from tools import dispatch, registry
    from agent_reasoning.ReAct.support.tool_pool import get_pool
    from agent_reasoning.ReAct.support.tool_resilience import call_with_resilience

    def _search():
        spec = registry.get("search_text")
        args = {"query": message, "k": C.RAGLITE_SEARCH_K,
                "score_ratio": C.RAGLITE_SEARCH_SCORE_RATIO,
                "with_scope_info": True}
        if scope_stem:
            args["scope_stem"] = scope_stem
        return call_with_resilience("search_text", args, spec, invoke=dispatch)

    return get_pool().submit(_search)


def _qc_feedback_msg(feedback: str) -> dict:
    """质检/升级反馈包装成 user 槽位的消息(维持 user/assistant 交替结构),
    但明确标注这是系统内部自动质检结果、并非用户发言。
    否则模型会误以为用户在纠错,回出"您说得对/感谢指出"这类对不存在的用户发言的回应。"""
    fb = (feedback or "").strip() or "请更严谨地重新生成回答"
    return {"role": "user", "content": (
        "[系统内部质检提示 —— 这不是用户消息,用户并未发言或指出问题。]"
        f"上一版回答未通过自动核查:{fb}。"
        "请据此自检并重新给出完整、准确、表述自然的最终回答;"
        "不要向用户致谢或表示认同(如“您说得对/感谢指出”),直接呈现修正后的答案。"
    )}


def _run_tier(tier: str,
              message: str,
              history: list[dict],
              **kwargs):
    """按 tier 调用对应路径,返回事件生成器。

    simple  -> run_simple(单轮直答,忽略步数/时长/硬截止);
    raglite -> run_raglite(1 次检索 + 1 次作答快路径);
    react   -> run_agent_graph(ReAct 检索循环)。
    """
    if tier == "simple":
        simple_kw = {k: v for k, v in kwargs.items() if k not in _NON_SIMPLE_KW}
        return run_simple(message, history, **simple_kw)
    if tier == "raglite":
        return run_raglite(message, history, **kwargs)
    # react 消费投机检索 future:首轮作为"预检索"注入(见 nodes.agent_node),
    # 模型可直接引用原始问题的检索结果,省一步检索;simple 不消费。
    return run_agent_graph(message, history, **kwargs)


def react_stream(message,
                 history,
                 max_total_seconds: Optional[int] = None,
                 on_event: Optional[Callable[[dict], None]] = None,
                 *,
                 thread_id: Optional[str] = None,
                 username: Optional[str] = None,
                 session_id: Optional[str] = None,
                 max_steps: Optional[int] = None):
    """生成器:yield SSE 事件 dict。

    流程:
      1. classify_complexity 判 tier -> 先发 ``tier`` 事件。
      2. 跑该 tier 路径,实时转发 token 等事件,但**扣留**终端 ``done``。
      3. 拿到答案后过 quality_gate:
         - passed           -> 补发警示(若有)再放行 held ``done``;
         - failed           -> 同 tier 重做一次(带 feedback);
         - needs_escalation -> simple→react 升级重跑(发 ``escalation`` 事件),
                              整条请求最多升一次。
    """
    if thread_id is None:
        thread_id = str(uuid.uuid4())

    # 入口等待门:上一轮的后台记忆链(独立记忆图)没跑完时先等它,保证本轮
    # 构建上下文时长期偏好/摘要/compact 已就位。有界超时(MEM_WAIT_IDLE_TIMEOUT,
    # 默认 10s)兜底放行——主链路永不被记忆链拖死;未完成的幂等重做留给下一轮。
    try:
        from agent_reasoning.ReAct.support.memory_background import wait_previous_turn
        wait_previous_turn(username, thread_id)
    except Exception:
        pass

    # 0+1. 入口判定(澄清 + 路由)。ROUTER_FUSED=1 时合并为单次判定
    #      (规则可判时零 LLM),替代旧的 clarify + router 两次串行调用。
    # 先解析会话 scope 锁:轮次12 起锁定仅作"本会话已消解、不再反问"标记,
    # 不再注入检索加权(投机检索/下游 tier 均不带 scope_stem)。
    scope_stem = ""
    clarified_q = ""   # 反问应答轮刚转正时的原问题(检索用它,见 _scope_resolve)
    if getattr(C, "RETRIEVAL_CONFLICT_CLARIFY", False):
        try:
            scope_stem, clarified_q = _scope_resolve(
                username, message, history or [])
        except Exception:  # noqa: BLE001  信号层故障不阻断主链路
            logger.warning("scope resolve failed", exc_info=True)
    # 反问应答轮:入口判定(澄清/路由)必须用原问题——本轮 message 是选项
    # 文本(对象名,往往短且无领域词面),拿它判定会被误路由到 simple
    # (simple 不检索,recall 归零)或再触发一次澄清(轮次12)。
    entry_q = clarified_q or message

    search_future = None
    if getattr(C, "ROUTER_FUSED", True):
        # 投机检索:在融合分类 LLM 调用前预发 search_text,两路重叠省 1.5~2.5s;
        # 规则短路/rule-routed 场景该 future 就是 raglite 本来要做的检索(零浪费),
        # react 则作为首轮预检索注入。仅在明显闲聊(问候/超短)时跳过——领域
        # 关键词表无法穷举(实测"重掺砷硅单晶/离子发生器"均漏),宁滥勿缺:
        # 一次后台检索成本远低于 react 漏检索导致的零来源作答。
        if (getattr(C, "RAGLITE_SPECULATIVE_SEARCH", True)
                and getattr(C, "RAGLITE_ENABLED", True)
                and (clarified_q or not _obviously_simple(message))):
            try:
                # 轮次12:scope 加权已全面退役——反问应答轮与锁定粘性轮一律
                # 正常翻译双路检索(检索服务内 原query+英文翻译 RRF),
                # query 用原问题(clarified_q)或当前消息;选项文本绝不单独
                # 当 query(丢原问题信息需求,轮次11 dev200 事故)。
                # clarified_q 非空时无视 _obviously_simple:选项文本(对象名)
                # 往往短且无领域词面,会被误判闲聊跳过投机检索。
                search_future = _speculative_search(clarified_q or message)
            except Exception:  # noqa: BLE001  投机失败不影响主链路
                search_future = None
        try:
            entry = classify_and_clarify(entry_q, history or [])
        except Exception:
            logger.warning("classify_and_clarify 异常,兜底 react", exc_info=True)
            entry = {"need_clarify": False, "tier": "react",
                     "confidence": 0.0, "source": "fallback"}
    else:
        try:
            clarify = check_clarify(entry_q, history or [])
        except Exception:
            logger.warning("check_clarify 异常,跳过澄清", exc_info=True)
            clarify = {"need_clarify": False}
        if clarify.get("need_clarify"):
            yield {"type": "clarify",
                   "question": clarify.get("question", ""),
                   "options": clarify.get("options") or [],
                   "source": clarify.get("source", "llm")}
            yield {"type": "done", "trace": {}}
            return
        decision = classify_complexity(entry_q, history or [])
        entry = {"need_clarify": False, "tier": decision["tier"],
                 "confidence": decision.get("confidence", 0.0),
                 "source": decision.get("source", "fallback")}
    if entry.get("need_clarify"):
        yield {"type": "clarify",
               "question": entry.get("question", ""),
               "options": entry.get("options") or [],
               "source": entry.get("source", "llm")}
        yield {"type": "done", "trace": {}}
        return

    tier = entry["tier"] or "react"

    # 轮次9/12:检索冲突反问——投机检索机械信号(top-4 跨文档族且分差小且
    # 词面未确信命中榜首族)只是廉价预筛;命中后先经 _conflict_objects 用
    # lite LLM 判定:问题已指明具体对象 → 不反问照常作答(具体问题被相似
    # 手册误伤反问是轮次11 dev200 的主要伤害面);确实没指明 → 以候选块中
    # 提取的对象名为选项反问(绝不给文档名)。仅首轮问;应答轮经
    # _scope_resolve 转正锁定。
    if (getattr(C, "RETRIEVAL_CONFLICT_CLARIFY", False)
            and not scope_stem and search_future is not None
            and not (history or []) and tier in ("raglite", "react")):
        stems, citems = _conflict_probe(search_future)
        if stems:
            specific, cquestion, choices = _conflict_objects(message, citems)
            if not specific and len(choices) >= 2:
                _SCOPE_PENDING[_scope_key(username, history or [], message)] = {
                    "choices": choices, "stems": stems, "ts": time.time(),
                    "message": message}   # 原问题:应答轮检索要用它(选项只是消歧)
                if len(_SCOPE_PENDING) > 256:
                    _SCOPE_PENDING.clear()
                yield {"type": "clarify",
                       "question": cquestion or _CONFLICT_CLARIFY_QUESTION,
                       "options": [c[0] for c in choices],
                       "option_stems": [c[1] for c in choices],
                       "source": "retrieval-conflict"}
                yield {"type": "done", "trace": {}}
                return
            # specific(问题已指明对象)或提不出 ≥2 个候选对象:不反问,照常作答

    # 2. 告知前端所选 tier。raglite 对外仍报 "react"(eval tier_ok / 前端零改动),
    #    真实路径放 path 字段供观测。
    yield {"type": "tier",
           "tier": "react" if tier == "raglite" else tier,
           "path": tier,
           "confidence": entry.get("confidence", 0.0),
           "source": entry.get("source", "fallback")}

    run_history = list(history or [])
    escalations = 0
    redos = 0
    held_done: Optional[dict] = None
    answer = ""
    qc_feedback: Optional[str] = None

    # 端到端硬预算:整条请求(含升级/重做)共享一个绝对截止墙钟。
    # 初始 tier 预算 + 预留一次升级的 react 预算(raglite 25s + react 25s),
    # 各 tier 自身的 max_total_seconds 仍约束单次执行。
    deadline_cap = (max_total_seconds if max_total_seconds is not None
                    else int(C.TIER_CONFIG.get(tier, C.TIER_CONFIG["react"])
                             ["max_total_seconds"])
                    + (int(C.TIER_CONFIG["react"]["max_total_seconds"])
                       if tier != "react" else 0))
    hard_deadline = time.time() + deadline_cap

    while True:
        held_done = None
        answer = ""
        path_errored = False
        # qc_feedback 不在循环顶复位:升级/重做分支设置后,下一轮迭代消费
        # (普通首轮为 None;重做分支总会覆盖,无残留路径)。

        # 硬预算将尽:不再开启新路径/重做,直接收尾。
        if hard_deadline - time.time() < 3.0:
            yield {"type": "status",
                   "message": "已达整体响应时限,请稍后重试或简化问题。"}
            yield {"type": "done", "trace": {}}
            return

        # 每个 tier 用自己的步数/时长预算;显式入参覆盖配置(测试/调用方可用)。
        tcfg = C.TIER_CONFIG.get(tier, C.TIER_CONFIG["react"])
        tier_max_steps = max_steps if max_steps is not None else tcfg["max_steps"]
        tier_max_total = (max_total_seconds if max_total_seconds is not None
                          else tcfg["max_total_seconds"])

        gen = _run_tier(
            tier, message, run_history,
            thread_id=thread_id, username=username, session_id=session_id,
            max_steps=tier_max_steps, max_total_seconds=tier_max_total,
            hard_deadline=hard_deadline,
            qc_feedback=qc_feedback,
            on_event=on_event,
            # 投机检索 future:raglite 直接消费,react 作为预检索注入;simple 不消费
            search_future=(search_future if tier in ("raglite", "react") else None),
            # 轮次12:不再向下传 scope_stem——锁定仅作"本会话已消解、不再反问"
            # 标记,检索一律正常翻译双路,无文档族加权
        )
        for ev in gen:
            etype = ev.get("type")
            if etype == "assistant_message":
                answer = ev.get("content", "") or answer
            if etype == "error":
                path_errored = True
            if etype == "done":
                # 扣留 done:质检通过/放行后再发;升级/重做时丢弃
                held_done = ev
                continue
            yield ev

        # 路径异常且未产出 done:不再质检/升级(答案为空),但必须补发终态 done
        # (错误事件已由 run_path 发出)。done 是 SSE 契约的唯一终端事件,
        # 无条件补齐,前端不能依赖"连接关闭"这类传输层副作用收尾。
        if held_done is None:
            yield {"type": "done", "trace": {}, "final_reason": "error"}
            return

        # 路径自身报错(final_reason=error)但仍发了 done:不做升级/重做,放行
        if path_errored:
            yield held_done
            return

        # 3. 质检门
        context = {
            "question": message,
            "history": run_history,
            # react 路径在 done 事件带回本轮最高检索相关分/检索次数,供低置信判定;
            # simple 直答不带这两个字段(默认 0,不触发 react 低置信规则)。
            "retrieval_max_score": float(held_done.get("retrieval_max_score") or 0.0),
            "search_count": int(held_done.get("search_count") or 0),
        }
        result = quality_check(answer, context, tier=tier)
        verdict = result.get("verdict", "passed")
        warnings = list(result.get("warnings") or [])
        feedback = result.get("feedback", "") or ""

        if verdict == "needs_escalation":
            nxt = _NEXT_TIER.get(tier)
            if nxt is not None and escalations < _MAX_ESCALATIONS:
                escalations += 1
                redos = 0
                # 带上已有历史 + 失败答案 + 反馈,供升级后路径参考
                if answer:
                    run_history = run_history + [
                        {"role": "assistant", "content": answer},
                        _qc_feedback_msg(feedback or "请更严谨地回答"),
                    ]
                yield {"type": "escalation", "from_tier": tier,
                       "to_tier": nxt, "reason": feedback}
                qc_feedback = _qc_feedback_msg(feedback or "请更严谨地回答")["content"]
                tier = nxt
                continue  # 丢弃当前 held_done,重跑升级路径

        if verdict == "failed" and redos < _MAX_REDOS:
            redos += 1
            # 先发 reflect 让前端清空已上屏的旧答案,否则重做的 token 会追加在旧答案后。
            yield {"type": "reflect", "feedback": feedback or "质检未过,重新生成"}
            yield {"type": "status",
                   "message": "质检未过,正在换关键词重新检索作答…"}
            if answer:
                run_history = run_history + [
                    {"role": "assistant", "content": answer},
                    _qc_feedback_msg(feedback or "请重新生成回答"),
                ]
            # 同一 thread_id 重做,只清空 ReAct 工作记忆(checkpoint messages):
            # 等价旧版"换新 thread_id = 全新 checkpoint"的清理效果,但短期流水/
            # 事实表/摘要/入口等待门仍归原会话键——重做轮问答对下一轮召回可见
            # (修复换 thread 导致的下一轮多轮失忆)。清空后重做轮为冷启动,由
            # build_messages 以 Redis 短期流水为权威重建完整上下文;质检反馈经
            # qc_feedback 显式注入(不再依赖 history 夹带)。
            # 仅 react 有 checkpoint;simple/raglite 无工作记忆可清。
            if tier == "react":
                try:
                    from agent_reasoning.ReAct.support.memory_background import reset_react_memory
                    reset_react_memory(username, thread_id)
                except Exception:
                    logger.warning("质检重做清空工作记忆失败,将带旧上下文重做",
                                   exc_info=True)
            qc_feedback = _qc_feedback_msg(feedback or "请重新生成回答")["content"]
            continue  # 同 tier 干净重做,丢弃 held_done

        # passed,或已无升级/重做预算 -> 放行(有警示则以可见 status 提示,fail-open)
        for w in warnings:
            yield {"type": "status", "message": f"⚠️ {w}"}
        yield held_done
        return
