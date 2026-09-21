# -*- coding: utf-8 -*-
"""从工具结果中提取来源条目(供前端显示引用条)。

优先使用 ToolSpec.source_extractor(由各工具自行声明,支持 doc/web 等不同
来源结构);无 spec / 无 extractor 时回退到本地检索块的历史提取逻辑。

另含 ``final_citation_cards``:答案定稿后,把全部检索来源按"书"去重、并以
答案是否实际引用重排,产出与答案引用对齐的来源卡片(供 ReAct finalize 与
Plan-Execute 综合定稿复用)。
"""
import re


def _norm(s):
    """归一化:只保留中英文字与数字,小写拉丁(用于书名宽松匹配)。"""
    return re.sub(r"[^0-9a-z一-鿿]", "", str(s or "").lower())


def _lcs_len(a, b):
    """两个字符串的最长公共子串长度。"""
    best = 0
    prev = [0] * (len(b) + 1)
    for ca in a:
        cur = [0] * (len(b) + 1)
        for j, cb in enumerate(b, 1):
            if ca == cb:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best = cur[j]
        prev = cur
    return best


def _char_cover(short, long):
    """short 去重字符出现在 long 中的比例。作者+书名倒装时公共子串被打断
    (如引"何丹农纳米制造"而书名"纳米制造…何丹农著"),但引用名的字几乎都在
    书名里,用字符覆盖率判定同一本书。"""
    if not short:
        return 0.0
    chars = set(short)
    return sum(1 for ch in chars if ch in long) / len(chars)


def _cited_doc_names(answer):
    """从答案内联引用 [《文档》 pNN] 提取被引用文档名(归一化);跳过图/表引用。"""
    names = []
    for b in re.findall(r"\[([^\[\]]{2,80}?)\]", answer or ""):
        if re.match(r"^\s*(图|表|Figure|Fig|Table|步骤|第.{0,3}步)", b, re.I):
            continue
        # 剥掉末尾页码 token,保留书名本体(含内部数字,如"21世纪""2015年版")
        n = _norm(re.sub(r"[\s，,。:：;；]*[pP]?\s*\d+(\s*[-–—~]\s*\d+)?\s*$", "", b))
        if len(n) >= 4:
            names.append(n)
    return names


def _source_is_cited(source, names, answer):
    """该来源是否被答案实际引用:web 看 url 是否出现在答案;文档看书名宽松匹配
    (双向包含 / 足够长公共子串 / 公共子串≥4 且引用名字符几乎都命中——兼容 LLM
    "作者+书名"倒装,如引"何丹农纳米制造"而卡片为"纳米制造…何丹农著")。"""
    if source.get("source_type") == "web":
        url = str(source.get("url") or "").rstrip("/")
        return bool(url) and url in (answer or "")
    sn = _norm(source.get("source_stem", ""))
    if not sn:
        return False
    for n in names:
        if sn in n or n in sn:
            return True
        l = _lcs_len(sn, n)
        if l >= 6 and l >= 0.5 * len(n):
            return True
        if l >= 4 and _char_cover(n, sn) >= 0.8:
            return True
    return False


def _book_key(s):
    """卡片按"书/来源"聚合的主键:文档用书名 source_stem(同一本书的不同分块
    同名),web 用 url。"""
    if s.get("source_type") == "web":
        return "url:" + str(s.get("url") or "")
    return "doc:" + str(s.get("source_stem") or "")


def final_citation_cards(answer, sources, k=6):
    """答案定稿后的最终来源卡片(按"书"去重 + 被引用优先)。

    答案常会引用多本不同的书;若按分块(chunk)取前 k 条,同一本书的多个高分
    分块会占满卡片、把答案引用的其它书挤出(且按插入/检索顺序取也未必对齐答案)。
    这里先按书聚合(每本保留最高分分块),再把"答案实际引用的书"排在前面、未引用
    的按检索分补位,最终取 k 本不同的书——保证卡片覆盖答案真正依据的来源。
    """
    pool = [s for s in (sources or []) if isinstance(s, dict)]
    # 按书聚合:同主键保留最高分的一个分块作为代表
    best_book = {}
    for s in pool:
        bk = _book_key(s)
        if not bk or bk.endswith(":"):
            continue
        if bk not in best_book or float(s.get("score", 0) or 0) > float(
                best_book[bk].get("score", 0) or 0):
            best_book[bk] = s
    books = list(best_book.values())
    names = _cited_doc_names(answer)
    cited = [s for s in books if _source_is_cited(s, names, answer)]
    uncited = [s for s in books if not _source_is_cited(s, names, answer)]
    cited.sort(key=lambda s: float(s.get("score", 0) or 0), reverse=True)
    uncited.sort(key=lambda s: float(s.get("score", 0) or 0), reverse=True)
    return (cited + uncited)[:k]


def _extract_doc_sources(result):
    """历史逻辑:从检索工具(search_text/search_image)返回里提取来源条目。

    兼容文本与图像块。每条截断 content 到 160 字。
    实现已统一收敛到 tools.mcp_policies.extract_doc_sources(含多媒体透传
    image_url/item_type/description/video_url),此处委托并兜底,避免双份漂移。
    """
    try:
        from tools.mcp_policies import extract_doc_sources
        return extract_doc_sources(result)
    except Exception:  # 兜底:仅基础字段,保证来源提取永不被策略层故障阻断
        out = []
        if not isinstance(result, list):
            return out
        for r in result:
            if not isinstance(r, dict) or not r.get("source_stem"):
                continue
            page = r.get("page_num") or r.get("page_start")
            heading = r.get("heading_path") or r.get("caption") or ""
            out.append({
                "chunk_id": r.get("chunk_id", ""),
                "source_stem": r["source_stem"],
                "page": f"p{page}" if page else "",
                "heading": heading,
                "score": round(float(r.get("score") or 0), 4),
                "content": (r.get("content", "") or "")[:160],
            })
        return out


def sources_from_result(result, *, spec=None):
    """从工具返回里提取来源条目列表。

    :param spec: 可选 ToolSpec;有 source_extractor 时优先用它。
    """
    if spec is not None and getattr(spec, "source_extractor", None) is not None:
        try:
            extracted = spec.source_extractor(result)
        except Exception:
            return []
        return [s for s in extracted if isinstance(s, dict)]
    return _extract_doc_sources(result)
