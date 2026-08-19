# -*- coding: utf-8 -*-
"""Grounding:引用验证(正则) + 忠实度检测(LLM) + 综合校验(从原 chat.service 抽出)。"""
import json
import logging
import re

import config as C

from .llm import get_client, llm_create_with_retry, LLM_TIMEOUT

logger = logging.getLogger("agent")

# 匹配 [文档名 p123] 或 [文档名 p123-125] 或 [文档名 123]
_CITATION_RE = re.compile(
    r"\[([^\[\]]+?)\s+p?(\d+)(?:\s*[-–—]\s*(\d+))?\s*\]"
)


def verify_citations(answer, sources):
    """校验答案中的引用标注是否真实存在于检索结果中。

    返回 (all_valid, invalid_citations)。
    """
    found = _CITATION_RE.findall(answer)
    if not found:
        return True, []

    valid = set()
    for s in sources:
        stem = (s.get("source_stem") or "").strip()
        page = (s.get("page") or "").replace("p", "").strip()
        if stem and page.isdigit():
            valid.add((stem, int(page)))

    invalid = []
    for stem, page_start, _ in found:
        stem = stem.strip()
        page = int(page_start)
        if (stem, page) in valid:
            continue
        fuzzy = any(
            (stem in vs_stem or vs_stem in stem) and vs_page == page
            for vs_stem, vs_page in valid
        )
        if not fuzzy:
            invalid.append(f"[{stem} p{page_start}]")

    return len(invalid) == 0, invalid


def check_faithfulness(answer, sources):
    """用 LLM 检查答案是否忠于检索内容。

    返回 (score, issues) 或 None(检测未能执行)。
    """
    try:
        source_text = "\n---\n".join(
            f"[{s.get('source_stem', '')} {s.get('page', '')}] "
            f"{(s.get('content') or '')[:300]}"
            for s in sources[:5]
        )
        prompt = (
            "请检查以下回答是否忠于检索资料。\n\n"
            f"检索资料:\n{source_text}\n\n"
            f"回答:\n{answer[:2000]}\n\n"
            "检查:\n"
            "1. 回答中的事实性陈述是否能在检索资料中找到支持?\n"
            "2. 回答是否包含检索资料中没有的信息(且未标注为通用知识)?\n\n"
            '只输出 JSON: {"score": 0.0-1.0, "issues": ["问题1", "问题2"]}\n'
            "score: 1.0=完全忠于资料, 0.5=部分存疑, 0.0=大量无法验证"
        )
        client = get_client()
        resp, err = llm_create_with_retry(
            client, trace_id="grounding",
            model=C.OPENAI_TEXT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=50,
            timeout=LLM_TIMEOUT,
        )
        if err is not None:
            raise err
        raw = resp.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        result = json.loads(raw)

        score = float(result.get("score", 1.0))
        issues = result.get("issues", [])
        if not isinstance(issues, list):
            issues = []
        return score, issues
    except Exception as e:
        logger.warning(f"faithfulness check failed: {e}")
        return None


def grounding_check(answer, sources):
    """综合校验:引用验证(全部) + 忠实度检测(仅高分 chunk)。"""
    warnings = []
    if sources:
        ok, invalid = verify_citations(answer, sources)
        if not ok and invalid:
            warnings.append(f"以下引用未在检索结果中找到: {', '.join(invalid[:3])}")

    threshold = getattr(C, "GROUNDING_FAITHFULNESS_THRESHOLD", 0.5)
    high_score_sources = [s for s in sources if s.get("score", 0) > threshold]
    if high_score_sources and answer.strip():
        result = check_faithfulness(answer, high_score_sources)
        if result is None:
            warnings.append("答案忠实度检测因临时异常暂不可用,请自行核实答案")
        else:
            score, _issues = result
            if score < threshold:
                warnings.append("回答的部分内容未能从检索资料中验证,请注意核实")

    return {"passed": len(warnings) == 0, "warnings": warnings}


def yield_grounding_warnings(full_reply, collected_sources, result_box=None):
    """答案完成后做 grounding 检查,yield 警告事件(保留原生成器接口)。"""
    if not full_reply.strip() or not collected_sources:
        if result_box is not None:
            result_box.append({"passed": True, "warnings": []})
        return
    yield {"type": "status", "message": "验证答案来源…"}
    try:
        grounding = grounding_check(full_reply, list(collected_sources.values()))
        if result_box is not None:
            result_box.append(grounding)
        if not grounding["passed"]:
            for w in grounding["warnings"]:
                yield {"type": "status", "message": f"⚠️ {w}"}
    except Exception as e:
        logger.warning(f"grounding check error: {e}")
        if result_box is not None:
            result_box.append({"passed": True, "warnings": []})
