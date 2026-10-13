# -*- coding: utf-8 -*-
"""RAGAS 式离线 LLM 裁判(不进在线链路/镜像)。

三个经典指标的"轻量自实现"版,复用项目已有的 OpenAI 兼容客户端(judge 默认走
MODEL_LIGHT 省成本)与检索微服务的 BGE-m3 嵌入(/embed_text):

- faithfulness(忠实度):从答案抽取原子陈述,逐条判能否在检索 context 中找到依据,
  返回"可被支撑的陈述比例"。防幻觉核心指标。
- context_precision(上下文精确率):逐块判该 chunk 对回答问题是否有用,返回有用块占比。
  衡量检索是否把槽位浪费在无关块上。
- answer_relevancy(答案切题):BGE-m3 余弦(question↔answer)为主 + LLM 切题判定兜底,
  取两者平均;衡量答非所问。
- context_recall(可选,需参考答案 ground_truth):参考答案的每个要点能否在 context 中
  找到依据。无 ground_truth 的题跳过(返回 None)。

所有 LLM 判定都要求模型只输出 JSON;解析失败时保守回退(不抛异常,记 None/默认值),
保证整批评测不被单题裁判失败拖垮。
"""
from __future__ import annotations

import json
import logging
import re
import time

import config as C

from . import metrics as M

logger = logging.getLogger("eval.judge")

_CLIENT = None
_JUDGE_MODEL = None


def _client():
    global _CLIENT, _JUDGE_MODEL
    if _CLIENT is None:
        import os
        import openai
        _JUDGE_MODEL = (os.getenv("EVAL_JUDGE_MODEL", "").strip()
                        or getattr(C, "MODEL_LIGHT", "") or "light")
        _CLIENT = openai.OpenAI(
            api_key=getattr(C, "EFFECTIVE_LLM_API_KEY", C.OPENAI_API_KEY),
            base_url=getattr(C, "EFFECTIVE_LLM_BASE_URL", C.OPENAI_BASE_URL),
            timeout=40,
        )
    return _CLIENT


def _chat_json(system: str, user: str):
    """调 judge 模型,要求只输出 JSON;返回解析后的对象,失败返回 None。

    Ark coding 网关实测会间歇性(~10-20%)对任意合法请求返回
    400 MissingParameter("missing model parameter")伪错误——与内容无关
    (同一 prompt 重试即可成功),故这里做 3 次短退避重试。"""
    # doubao 系注入 thinking disabled:纯思考模式下非流式调用会思考数分钟,
    # 且逐块到达的 reasoning 字节绕过 httpx read 超时(实测挂 17min+,
    # run23m 裁判卡死根因)。GLM 系纯思考不能带该参数(400),只对 doubao 发。
    kwargs = dict(
        model=_JUDGE_MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    if "doubao" in str(_JUDGE_MODEL or "").lower():
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
    last_e = None
    for attempt in range(4):
        try:
            resp = _client().chat.completions.create(**kwargs)
            text = (resp.choices[0].message.content or "").strip()
            m = re.search(r"\{.*\}", text, re.S)
            return json.loads(m.group(0) if m else text)
        except Exception as e:  # 裁判失败不应中断整批
            last_e = e
            # 伪 400 是秒级突发窗口,0.5s 级退避逃不出窗口(实测同窗口内
            # 3 连败),用 1/3/6s 拉开间隔
            time.sleep(1.0 * (2 ** attempt))
    logger.warning("judge LLM 调用失败(4次): %s", str(last_e)[:160])
    return None


# ---------------- 嵌入(答案切题主信号) ----------------

def _embed(texts: list[str]):
    """调检索微服务 /embed_text 取 BGE-m3 dense 向量;失败返回 None。"""
    try:
        import httpx
        base = getattr(C, "RETRIEVAL_SERVICE_URL", "http://127.0.0.1:8002").rstrip("/")
        tok = getattr(C, "RETRIEVAL_INTERNAL_TOKEN", "")   # 服务间鉴权
        headers = {"X-Internal-Token": tok} if tok else None
        r = httpx.post(base + "/embed_text", json={"texts": list(texts)},
                       headers=headers, timeout=60)
        r.raise_for_status()
        return r.json()["dense"]
    except Exception as e:
        logger.warning("embed_text 调用失败: %s", str(e)[:160])
        return None


# ---------------- 指标 ----------------

def faithfulness(answer: str, contexts: list[str]) -> float | None:
    """答案陈述中可被 context 支撑的比例。"""
    if not (answer or "").strip() or not contexts:
        return None
    ctx_blob = "\n\n".join(f"[资料{i+1}] {c}" for i, c in enumerate(contexts))
    data = _chat_json(
        "你是严格的事实核查员。只输出 JSON,不要输出多余文字。",
        "下面是系统回答与它检索到的内部资料。请:\n"
        "1) 把【回答】拆成若干条可独立核验的原子陈述(claims),短句即可;\n"
        "2) 逐条判断该陈述能否由【资料】内容蕴含(supported=true/false)。\n"
        "判断标准(语义蕴含,不要求逐字对应):与资料含义一致的同义改写、直接推论、"
        "合理具体化(如资料说\"移走基板\",回答说\"手动移走卡住的基板\")都算 supported=true;"
        "只有资料无法推出、或与资料矛盾的陈述才算 supported=false。\n"
        f"【回答】\n{answer}\n\n【资料】\n{ctx_blob}\n\n"
        '输出格式:{"claims":[{"claim":"...","supported":true}]}')
    if not data or not isinstance(data.get("claims"), list) or not data["claims"]:
        return None
    sup = sum(1 for c in data["claims"] if c.get("supported"))
    return round(sup / len(data["claims"]), 4)


def context_precision(question: str, contexts: list[str]) -> float | None:
    """检索块中对回答该问题真正有用的比例。"""
    if not contexts:
        return None
    blocks = "\n".join(f"[资料{i+1}] {c}" for i, c in enumerate(contexts))
    data = _chat_json(
        "你是检索相关性评判员。只输出 JSON。",
        f"问题:{question}\n\n以下是检索回的资料块,请逐块判断它对【回答该问题】"
        "是否有用(相关且能提供依据,relevant=true;无关/跑题=false)。\n"
        f"{blocks}\n\n"
        '输出格式:{"blocks":[{"index":1,"relevant":true}]}')
    if not data or not isinstance(data.get("blocks"), list) or not data["blocks"]:
        return None
    rel = sum(1 for b in data["blocks"] if b.get("relevant"))
    return round(rel / len(data["blocks"]), 4)


def answer_relevancy(question: str, answer: str) -> float | None:
    """答案切题度:BGE-m3 余弦与 LLM 判定的平均(任一失败用另一个;都失败 None)。"""
    if not (answer or "").strip():
        return None
    scores = []
    vecs = _embed([question, answer])
    if vecs and len(vecs) == 2:
        scores.append(max(0.0, M.cosine(vecs[0], vecs[1])))
    data = _chat_json(
        "你评判问答切题度。只输出 JSON。",
        f"问题:{question}\n回答:{answer}\n\n"
        "判断回答是否正面、切题地回应了问题(1=完全切题,0=完全答非所问)。\n"
        '输出格式:{"relevance":0.0到1.0的数}')
    if isinstance(data, dict) and isinstance(data.get("relevance"), (int, float)):
        scores.append(max(0.0, min(1.0, float(data["relevance"]))))
    if not scores:
        return None
    return round(sum(scores) / len(scores), 4)


def context_recall(question: str, ground_truth: str, contexts: list[str]) -> float | None:
    """参考答案要点能在 context 中找到依据的比例(需 ground_truth)。"""
    if not ground_truth or not contexts:
        return None
    ctx_blob = "\n\n".join(f"[资料{i+1}] {c}" for i, c in enumerate(contexts))
    data = _chat_json(
        "你是事实核查员。只输出 JSON。",
        "下面是问题的【参考答案】与系统检索到的【资料】。请把参考答案拆成若干要点,"
        "逐点判断该要点能否由资料支撑(supported=true/false)。\n"
        f"问题:{question}\n参考答案:{ground_truth}\n\n资料:\n{ctx_blob}\n\n"
        '输出格式:{"points":[{"point":"...","supported":true}]}')
    if not data or not isinstance(data.get("points"), list) or not data["points"]:
        return None
    sup = sum(1 for p in data["points"] if p.get("supported"))
    return round(sup / len(data["points"]), 4)


def judge_all(question, answer, sources, ground_truth="") -> dict:
    """对单题跑全部可用指标,返回 {指标: 分数或 None}。

    contexts 每块带出处标签(文档名/页码/标题):答案按引用规则会标注
    "[文档名 pN]",出处不在资料文本里,裁判看不到页码就把引用判成
    unsupported —— 系统性压低忠实度(实测 88% 答案带页码引用)。带上
    元数据后引用类陈述才可核验。"""
    contexts = []
    for s in (sources or []):
        content = str(s.get("content", "")).strip()
        if not content:
            continue
        meta = " ".join(str(s.get(k, "")).strip()
                        for k in ("source_stem", "page", "heading")
                        if s.get(k))
        contexts.append(f"(出处:{meta})\n{content}" if meta else content)
    out = {
        "faithfulness": faithfulness(answer, contexts),
        "context_precision": context_precision(question, contexts),
        "answer_relevancy": answer_relevancy(question, answer),
        "context_recall": context_recall(question, ground_truth, contexts) if ground_truth else None,
    }
    return out
