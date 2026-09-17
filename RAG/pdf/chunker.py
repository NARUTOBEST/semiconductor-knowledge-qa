# -*- coding: utf-8 -*-
"""content_list.json (v1) -> text chunks + image chunks。

确定性主路径:
  1. 按 L1/L2 标题切 section(标题来自 text_level);无 text_level 但匹配编号模式
     (第X章 / 1.2 / 가. / ① 等)的短文本作为虚拟 L3 子节标题。标题文本只进入
     heading_path,不重复进 content,避免三重冗余;只有正文/图/表的 section 才产生 chunk。
  2. section 内逐条 item 累加到文本缓冲,达到 target 时在最后一个句末边界
     (。！？.!? / 换行)收口;未竟句作为下一块起点,不夹断句子。target 内无
     完整句时继续攒到 max_size,仍无句界则硬切兜底;正文块间传递 ≤100 字尾句作为
     下一块前缀(overlap),表格行/脚注不参与;章节切换时 overlap 重置。
  3. 单条 item 超过 max_size 且无句界可切 -> 先调 LLM(按 text hash 缓存),
     LLM 失败/返回 None 时强制硬切兜底,保证不输出超 max_size 的 chunk;
     英文缩写(Fig./et al./e.g./U.S./No. 等)的句点不算句末;
     表格超长时按整行(\n)打包,单行过长在单元格边界(' | ')切,不切断单元格;
  4. 不足 min 的尾块并入同 section 上一 chunk(合并后不超过 max_size);带 overlap
     前缀的块不被合并,避免语义重复;
  5. image/chart 按阅读序挂到所在 text chunk,并各生成一条 image chunk;chart/table
     的 footnote 拼入正文。
header/footer/page_number/aside_text 直接丢弃。list 取 list_items、code 取 code_body;
正文/标题/caption/footnote 中的内联 HTML(<sup>/<sub>/<img>/<a>)清理后入向量。
表格留 HTML(完整存 table_html),并按行转纯文本全部入 content;大表跨块时,真正的列名
行(首行 ≥2 单元格且不少于第二行一半)像 Excel "顶端标题行"一样复制到每个续块开头;
分组标题行(colspan 单单元格)或 OCR 噪声行不当列名,不挂 table_html;公式留 LaTeX。
"""
import os, re, json, sys, hashlib, html

# sys.path:HERE=RAG/pdf(import chunker 自身/同目录),RAG=父(import embed/ark_client),config(import config)
_HERE = os.path.dirname(os.path.abspath(__file__))
_RAG = os.path.dirname(_HERE)
_CONFIG = os.path.join(os.path.dirname(_RAG), "config")
for _p in (_HERE, _RAG, _CONFIG):
    if _p not in sys.path:
        sys.path.insert(0, _p)

FILTER_TYPES = {"header", "footer", "page_number", "aside_text"}
HEADING_LEVELS = {1, 2, 3}
IMAGE_TYPES = {"image", "chart"}

# 正文 chunk 间重叠的最大字符数(取尾句,0=关闭)。表格行不参与重叠。
OVERLAP_CHARS = 100

# 内联 HTML 标签(text/page_footnote/caption 中残留的 <sup>/<sub>/<img>/<a> 等)
_TAG_RE = re.compile(r"<[^>]+>")
_BR_RE = re.compile(r"<br\s*/?>", re.I)

# ---- P5: OCR 单词粘连修复(如 thechamber -> the chamber)----
# 部分扫描/低质量 PDF 的文本层或 OCR 会丢失词间空格,产生超长小写串
# (如 anyofthecomponentsused)。仅切"全小写、>=24 字符、且内嵌常见英文功能词"
# 的串——合法英文词极少超过 24 字符(electromagnetic=15),功能词判据进一步排除
# 化学/德语长词与代码标识符,零误切。URL/邮箱含 . / @ 大写,不匹配纯小写串。
_GLUE_TOKEN_RE = re.compile(r"[a-z]{24,}")
_GLUE_FUNC_RE = re.compile(
    r"(?:the|and|that|with|for|are|was|were|this|which|from|been|have|into|"
    r"using|used|will|than|then|these|those|what|when|where|but|not|can|all)")
_wordninja = None


def _get_wordninja():
    global _wordninja
    if _wordninja is None:
        import wordninja
        _wordninja = wordninja
    return _wordninja


def _deglue(s):
    """把字符串中的 OCR 粘连长串用 wordninja 重新分词;无粘连串时原样返回。"""
    if not s or not _GLUE_TOKEN_RE.search(s):
        return s

    def _repl(m):
        tok = m.group(0)
        if not _GLUE_FUNC_RE.search(tok[3:]):  # 跳过开头被截断的半个词
            return tok
        return " ".join(_get_wordninja().split(tok))

    return _GLUE_TOKEN_RE.sub(_repl, s)


def _strip_html(s):
    """去掉 HTML 标签并解码实体(供表格单元格等结构化场景)。

    先去标签再 unescape:若先 unescape,&lt;img&gt; 会被还原成 <img>,反而引入标签。
    """
    if not s:
        return ""
    s = re.sub(r"<[^>]+>", " ", s)
    s = html.unescape(s)
    s = _deglue(s)
    return re.sub(r"\s+", " ", s).strip()


def _clean_inline(s):
    """清理正文/标题/脚注中的内联 HTML 标签与实体,保留文本内容。

    与 _strip_html 的区别:_strip_html 用于表格单元格等纯结构化场景,
    这里额外把 <br> 转为空格,并折叠空白,供正文入向量前清洗。
    顺序:先去标签,再 unescape 实体(&#x27;->'、&quot;->"、&amp;->&),最后折叠空白。
    """
    if not s:
        return ""
    s = _BR_RE.sub(" ", s)
    s = _TAG_RE.sub(" ", s)
    s = html.unescape(s)
    s = _deglue(s)
    return re.sub(r"\s+", " ", s).strip()


# ---- 编号标题识别(无 text_level 时作为虚拟 L3)----
_CJK_RE = re.compile(r"[一-鿿가-힣]")
_HEADING_CHAPTER_RE = re.compile(r"^第\s*[一二三四五六七八九十百千0-9]+\s*[章节条部分]")
_HEADING_NUMERIC_RE = re.compile(r"^\d{1,2}(?:[．.]\d{1,2}){0,3}[\s．.、]+(\S.*)$")
_HEADING_JAMO_RE = re.compile(r"^[가나다라마바사아자차카타파하][．.、]\s*[一-鿿가-힣]")
_HEADING_CIRCLED_RE = re.compile(r"^[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮]\s*[一-鿿가-힣]")
_INNER_SENT_RE = re.compile(r"[。！？…]|(?<=\s)[.!?]\s")
_MULTI_NUM_RE = re.compile(r"\d{1,2}\s*[.、．].*\d{1,2}\s*[.、．]")


def _match_numbered_heading(text):
    """无 text_level 的短文本若匹配编号标题模式,返回 3(虚拟 L3),否则 None。

    保守约束:长度 2~40、不含句内句末标点(。！？…)、非多个编号项拼一行。
    数字编号标题(1./1.2)额外要求含 CJK,或为 Title Case 且 ≤4 词的英文短标题。
    """
    t = (text or "").strip()
    if not t or len(t) < 2 or len(t) > 40:
        return None
    if t[-1] in "。.!?！？":
        return None
    if _INNER_SENT_RE.search(t):
        return None
    if _MULTI_NUM_RE.search(t):
        return None
    if _HEADING_CHAPTER_RE.match(t) or _HEADING_JAMO_RE.match(t) or _HEADING_CIRCLED_RE.match(t):
        return 3
    m = _HEADING_NUMERIC_RE.match(t)
    if m:
        rest = m.group(1).strip()
        if _CJK_RE.search(rest):
            return 3
        words = rest.split()
        if words and len(words) <= 4 and rest[0].isupper():
            return 3
    return None


# 编号序号(用于判断连续编号序列,把"1./2./3."步骤与真正的编号标题区分开)
_CIRCLED_CHARS = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮"
_JAMO_CHARS = "가나다라마바사아자차카타파하"
_NUM_PREFIX_RE = re.compile(r"^(\d{1,2})")


def _heading_number(text):
    """提取 text 开头的编号序号,返回 (族, 序号),否则 None。

    与 _match_numbered_heading 不同,这里不受 40 字长度限制:操作步骤的文本常超过
    40 字(如 "1. Isolate the reactor from the vacuum pumping...")。本函数仅用于
    _find_list_runs 判断"相邻项编号是否连续",而连续本身即列表的强证据,故放宽长度。
    族为 'numeric'/'circled'/'jamo';序号为从 1 开始的整数。第X章类不参与。
    """
    t = (text or "").strip()
    if not t:
        return None
    if t[0] in _CIRCLED_CHARS:
        return ("circled", _CIRCLED_CHARS.index(t[0]) + 1)
    if t[0] in _JAMO_CHARS and len(t) >= 2 and t[1] in "．.、":
        return ("jamo", _JAMO_CHARS.index(t[0]) + 1)
    # 数字编号:1. / 1) / 1、 / 1． 开头(不要求整条像标题,因为连续编号会相互印证)
    m = re.match(r"^(\d{1,2})\s*[．.、)]\s*\S", t)
    if m:
        return ("numeric", int(m.group(1)))
    return None


def _find_list_runs(items):
    """找出"连续编号序列"内的 text item 下标集合。

    若某编号项与相邻编号项同族且序号相差 1(如 1./2./3.、①②③、가/나/다),
    则这些项属于操作步骤/枚举列表,不应被当作 L3 标题切 section。查找相邻编号项时
    可跨过穿插的 image/chart/table(步骤里常见配图/警示表),但遇到任意 text
    (标题或正文段落)即停止。真正的编号标题(后跟正文段落、非下一个编号)不在此集合中。
    """
    nums = {}  # idx -> (family, value)
    for i, it in enumerate(items):
        if it.get("type") != "text" or it.get("text_level") in HEADING_LEVELS:
            continue
        n = _heading_number(it.get("text", ""))
        if n is not None:
            nums[i] = n

    def nearest_num(start, step):
        """从 start 出发沿 step(+1/-1)找最近的编号项,跳过图/表,遇 text 停。"""
        j = start + step
        while 0 <= j < len(items):
            tj = items[j].get("type")
            if tj in IMAGE_TYPES or tj == "table":
                j += step
                continue
            return nums.get(j)   # text(含标题/正文):是编号则返回,否则 None
        return None

    list_run = set()
    for i, (fam, val) in nums.items():
        prev = nearest_num(i, -1)
        nxt = nearest_num(i, +1)
        if (prev and prev[0] == fam and prev[1] == val - 1) or \
           (nxt and nxt[0] == fam and nxt[1] == val + 1):
            list_run.add(i)
    return list_run


def _table_rows(html):
    """表格 HTML -> 行文本列表:单元格用 " | " 连接,一行一条。

    保留行边界,使切块能按行贪心打包并在跨块时复制列名行,
    从而让整张表(含中后段行)全部进入向量,而不是只截前 400 字。
    合并单元格(colspan/rowspan)不展开,按出现顺序读单元格文本,对检索足够。
    """
    rows = []
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.I | re.S):
        cells = []
        for m in re.finditer(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, re.I | re.S):
            cell = _strip_html(m.group(1))   # 去单元格内嵌套标签 + 解码实体
            if cell:
                cells.append(cell)
        if cells:
            rows.append(" | ".join(cells))
    return rows


def _table_to_text(html, caption=""):
    """表格 HTML -> 按行纯文本(供调试/自测;切块主路径用 _table_rows 逐行入缓冲)。"""
    text = "\n".join(_table_rows(html))
    cap = (caption or "").strip()
    if cap:
        text = cap + ("\n" if text else "") + text
    return text


def _item_text(it):
    """非图非表 item -> 并入 content 的文本。"""
    t = it.get("type")
    if t == "equation":
        return _clean_inline(it.get("text", ""))        # LaTeX
    if t == "text":
        return _clean_inline(it.get("text", ""))
    if t == "page_footnote":
        text = _clean_inline(it.get("text", ""))
        return ("[脚注] " + text) if text else ""
    if t == "list":
        # MinerU: list 类型正文在 list_items(list[str]),text/content 字段通常为空
        items = it.get("list_items") or []
        return "\n".join(_clean_inline(x) for x in items if x)
    if t == "code":
        # MinerU: code 类型正文在 code_body,可能有 code_caption 作前缀
        body = _clean_inline(it.get("code_body", "") or it.get("text", "") or it.get("content", ""))
        cap = it.get("code_caption") or []
        cap_text = " ".join(_clean_inline(x) for x in cap if x).strip()
        return (cap_text + "\n" + body) if cap_text else body
    return _clean_inline(it.get("text", "") or it.get("content", ""))


def _footnote_text(it):
    """chart/table 的 footnote(列表)合并为单行文本,供拼入 chunk。"""
    for k in ("chart_footnote", "table_footnote", "image_footnote"):
        c = it.get(k)
        if c:
            return " ".join(_clean_inline(x) for x in c if x).strip()
    return ""


def _caption(it):
    for k in ("image_caption", "chart_caption", "table_caption"):
        c = it.get(k)
        if c:
            return " ".join(_clean_inline(x) for x in c if x).strip()
    return ""


def _split_by_sentence(text, max_size):
    """按句末标点/换行切,贪心打包到 <= max_size。返回多块列表;无法干净切(有单句>max)返回 None。"""
    # 用 _sent_end_iter 找真正的句界(跳过缩写点),按切分点拆成"句子片段"
    parts = []
    last = 0
    for m in _sent_end_iter(text):
        parts.append(text[last:m.end()])
        last = m.end()
    if last < len(text):
        parts.append(text[last:])
    out, cur = [], ""
    for p in parts:
        if not p.strip():
            continue
        if not cur:
            cur = p.strip()
        elif len(cur) + len(p) + 1 <= max_size:
            cur = cur + " " + p.strip()
        else:
            out.append(cur)
            cur = p.strip()
    if cur:
        out.append(cur)
    if any(len(s) > max_size for s in out):     # 有单句超 max,句界切分失败
        return None
    return out if len(out) > 1 else None        # 只有一块=没切,返回 None


# 句末边界:中文。！？、英文 .!? (后须跟空白或串尾)、换行。标点本身归入前一句。
# 注:英文缩写点(Fig./et al./e.g./U.S. 等)由 _sent_end_iter 过滤,不直接切句。
_SENT_END = re.compile(r"[。！？…]+|\n+|[.!?]+(?=\s|$)")

# 常见英文缩写:这些词后的 '.' 不单独视为句末(但若该缩写确实位于句末,
# 其后面的空白/结尾仍可作为句界——只是切分点不在缩写点之后而已)。
_ABBREV = {
    "fig", "figs", "et", "al", "e.g", "i.e", "u.s", "u.k", "vs", "no",
    "vol", "chap", "etc", "dr", "mr", "mrs", "ms", "prof", "sr", "jr",
    "inc", "ltd", "co", "corp", "dept", "approx", "appt",
    "ed", "eds", "rev", "trans", "pp", "p", "ch", "sec", "eq", "nos",
    "ca", "cf", "viz", "ph.d", "m.d", "b.s", "m.s",
}


def _is_abbrev_dot(text, start, end):
    """判断 _SENT_END 匹配 [start,end) 是否为英文缩写点(而非句末点)。

    读取匹配前的单词(去掉句点后),若其小写形式在 _ABBREV 中,则视为缩写点。
    """
    # 匹配区间 [start,end) 是标点(如 ".");向前找单词字符
    word_start = start
    while word_start > 0 and (text[word_start - 1].isalnum() or text[word_start - 1] == '.'):
        word_start -= 1
    word = text[word_start:start].lower().rstrip('.')
    return word in _ABBREV


def _sent_end_iter(text):
    """遍历 text 中真正的句末边界(跳过英文缩写点)。yield match 对象。"""
    for m in _SENT_END.finditer(text):
        seg = m.group()
        # 中文标点/换行直接放行
        if '.' not in seg:
            yield m
            continue
        # 若匹配到串尾或紧跟换行,即使是缩写点也视为句末
        # (缩写位于真正句末时,该点即句末点)
        if m.end() >= len(text) or text[m.end():m.end() + 1] in ("\n", "\r"):
            yield m
            continue
        if seg == '.' and _is_abbrev_dot(text, m.start(), m.end()):
            # 缩写点后还有正文(如 "Fig. 5"、"et al. said"):不切句
            continue
        yield m



def _cut_at_sentence(text, budget):
    """在 budget 字符内的最后一个句末边界切分。

    返回 (prefix, suffix):prefix 以句末标点结束且 len(prefix) <= budget,
    供当前 chunk 收尾;suffix 是未说完的句子,作为下一块的起点。
    budget 内没有任何完整句子时返回 ("", text),整句留给下一块(不夹断)。
    """
    if not text or budget <= 0:
        return "", text or ""
    if len(text) <= budget:
        return text, ""
    cut = 0
    for m in _sent_end_iter(text):
        if m.end() <= budget:
            cut = m.end()
        else:
            break
    if cut == 0:
        return "", text
    return text[:cut], text[cut:].lstrip()


def _tail_sentence(text, max_chars=OVERLAP_CHARS):
    """从 text 尾部取最后一个完整句子作为下一块的重叠前缀。

    仅在尾部 2*max_chars 窗口内查找,返回 10~max_chars 字的完整句(含句末标点);
    无完整句或句子过长/过短时返回 ""。表格行不调用此函数。
    """
    if not text or max_chars <= 10:
        return ""
    tail = text[-max_chars * 2:]
    ends = list(_sent_end_iter(tail))
    if not ends:
        return ""
    last_end = ends[-1].end()
    start = ends[-2].end() if len(ends) >= 2 else 0
    sent = tail[start:last_end].strip()
    if len(sent) < 10 or len(sent) > max_chars:
        return ""
    return sent


def _hard_split(text, max_size):
    """确定性硬切:优先在空格/标点边界切,实在没有就按字符硬切到 <= max_size。

    用于无句界的超长文本;不在 ' | ' 处切(那是表格单元格分隔符,会切断表格行)。
    """
    out = []
    s = text
    # 防御:非正预算(调用方扣减超长表头后可能出现)会导致 s[0:] 永不前进的死循环。
    # 退化为在半长处硬切,保证一定有进展。
    if max_size <= 0:
        max_size = max(len(s) // 2, 1)
    while len(s) > max_size:
        win = s[:max_size]
        cut = max(win.rfind(" "), win.rfind("，"), win.rfind(","),
                  win.rfind("；"), win.rfind(";"))
        if cut < max_size // 2:                 # 边界太靠前,直接硬切
            cut = max_size
        out.append(s[:cut].strip())
        s = s[cut:].strip()
    if s:
        out.append(s)
    return out if len(out) > 1 else None


def _split_long_line(line, max_size):
    """把一行超长文本切成 <= max_size 的多段,优先在 ' | ' 单元格边界切。

    用于表格中单行过长的兜底:不在单元格内部切断;若单个单元格本身超长,
    才退化为在该单元格内部按空格/逗号/分号切。
    """
    if len(line) <= max_size:
        return [line]
    if " | " not in line:
        return _hard_split(line, max_size) or [line[:max_size]]
    cells = line.split(" | ")
    groups, cur = [], ""
    for cell in cells:
        if not cur:
            cur = cell
        elif len(cur) + 3 + len(cell) <= max_size:
            cur = cur + " | " + cell
        else:
            if cur:
                groups.append(cur)
            if len(cell) > max_size:
                groups.extend(_hard_split(cell, max_size) or [cell[:max_size]])
                cur = ""
            else:
                cur = cell
    if cur:
        groups.append(cur)
    return groups


def _split_table_content(text, max_size):
    """按表格整行切分;超长行在单元格边界(' | ')切,保证不切断单元格。"""
    out, cur = [], ""
    for line in text.split("\n"):
        if not line.strip():
            continue
        if len(line) <= max_size:
            if not cur:
                cur = line
            elif len(cur) + 1 + len(line) <= max_size:
                cur = cur + "\n" + line
            else:
                out.append(cur)
                cur = line
            continue
        if cur:
            out.append(cur)
            cur = ""
        out.extend(_split_long_line(line, max_size))
    if cur:
        out.append(cur)
    return out if len(out) > 1 else None


def chunk_content_list(content_list, *, source_path, source_stem, auto_dir,
                       target=500, max_size=600, min_size=150, llm_split=None):
    """返回 (text_chunks, image_chunks),均为 dict 列表。"""
    items = [it for it in content_list if it.get("type") not in FILTER_TYPES]

    # LLM 切分结果按 (text_hash, target) 缓存,避免同一段长文本反复调用 LLM。
    # 同一 PDF 内常见重复模板/页眉长文本,缓存可显著减少 LLM 调用。
    _llm_cache = {}

    def _llm_split_cached(text, tgt):
        if llm_split is None:
            return None
        key = (hashlib.md5(text.encode("utf-8")).hexdigest(), tgt)
        if key in _llm_cache:
            return _llm_cache[key]
        try:
            result = llm_split(text, tgt)
        except Exception:
            result = None
        # 校验返回值:非空 list、每块为 str、每块非空
        if not isinstance(result, list) or not result:
            result = None
        elif not all(isinstance(s, str) and s.strip() for s in result):
            result = None
        _llm_cache[key] = result
        return result

    # ---- 1. 按 L1/L2/L3 标题切 section ----
    # 标题文本只进入 heading_path,不再作为 content 首行(避免与 heading_path/embed_text 三重重复),
    # 也避免纯标题章节产生碎片块。只有正文/图/表等内容 item 才进入 parts。
    # L3 有两个来源:MinerU text_level=3,或无 text_level 但匹配编号标题正则(가./1./①等)。
    # 连续编号序列(操作步骤 1./2./3.、①②③、가/나/다)是枚举列表而非标题,不切 section。
    list_runs = _find_list_runs(items)
    sections = []
    stack = {}        # {1: 文档标题, 2: 节标题, 3: 子节标题}
    cur = None
    for idx, it in enumerate(items):
        t = it.get("type")
        lvl = None
        heading_text = None
        if t == "text":
            tl = it.get("text_level")
            if tl in HEADING_LEVELS:
                lvl = tl
                heading_text = it.get("text", "")
            elif idx not in list_runs:
                # 连续编号序列内的项是列表步骤,不作为虚拟 L3 标题
                num_lvl = _match_numbered_heading(it.get("text", ""))
                if num_lvl:
                    lvl = num_lvl
                    heading_text = it.get("text", "")
        if lvl is not None:
            if cur and cur["parts"]:
                sections.append(cur)
            stack[lvl] = _clean_inline(heading_text)
            for l in [x for x in stack if x > lvl]:
                del stack[l]
            hp = " > ".join(stack[l] for l in sorted(stack) if stack.get(l)) or source_stem
            cur = {"hp": hp, "parts": []}            # 标题不进 parts
        else:
            if cur is None:
                cur = {"hp": source_stem, "parts": []}
            cur["parts"].append(it)
    if cur and cur["parts"]:
        sections.append(cur)

    # ---- 2. section 内打包 ----
    text_chunks = []
    image_chunks = []
    state = {"tseq": 0, "iseq": 0}

    def emit_visuals(visuals, parent_cid):
        for v in visuals:
            state["iseq"] += 1
            image_chunks.append({
                "chunk_id": f"{source_stem}__i{state['iseq']:05d}",
                "image_path": os.path.join(auto_dir, v.get("img_path", "")),
                "img_relname": v.get("img_path", ""),
                "description": "",
                "source_path": source_path, "source_stem": source_stem,
                "page_num": v.get("page_idx", 0),
                "caption": _caption(v),
                "parent_text_chunk_id": parent_cid,
                "chunk_index": state["iseq"],
                "item_type": v.get("type"),
            })

    def _split_pieces_at(pieces, cut):
        """把 (text,page,kind) 列表按"\\n"拼接后的字符位置 cut 切成前缀/后缀两段。

        切落在哪条 piece 内部就把该条 piece 拆成两半(同页码、同 kind),前缀进当前
        chunk、后缀(未竟句/未竟行)作下一块起点,页码范围随之正确分配。
        """
        pre, suf = [], []
        pos = 0
        for i, (txt, pidx, kind) in enumerate(pieces):
            start = pos
            end = pos + len(txt)
            if end <= cut:
                pre.append((txt, pidx, kind))
            elif start >= cut:
                suf.append((txt, pidx, kind))
            else:
                at = cut - start
                head, tail = txt[:at], txt[at:]
                if head:
                    pre.append((head, pidx, kind))
                if tail:
                    suf.append((tail, pidx, kind))
            pos = end + (1 if i < len(pieces) - 1 else 0)   # 行间 "\n"
        return pre, suf

    def flush_block(pieces, visuals, hp, overlap_prefix=""):
        """把已确定边界的一段文本刷成 chunk(可能因超 max 再切多块)。视觉挂首块。

        pieces 为 (text, page, kind):kind 描述表格片段,用于
          - 把该表完整 HTML 挂到含表头行的首块;
          - 续块以数据行起首时,自动把列名行补到块首(Excel "顶端标题行")。
        overlap_prefix 为上一块尾句(若有),作为本块首行以提供跨块语义重叠。
        返回本次 flush 产出的尾句(供下一块重叠),若不适合重叠返回 ""。
        """
        if not pieces and not visuals:
            return ""
        # 该块是否含某张表的表头行 -> 挂完整 HTML;以及块内首个表格片段是不是数据行
        # (是则说明表头在上一块,需在本块补列名)。
        tbl_html = None
        repeat_header = None
        body_lines = []
        for txt, _, kind in pieces:
            if kind and kind[0] == "hdr":
                if tbl_html is None:
                    tbl_html = kind[2]            # 表头行携带的完整表 HTML
                body_lines.append(txt)
            elif kind and kind[0] == "row":
                if repeat_header is None and tbl_html is None:
                    repeat_header = kind[1]       # 该数据行所属表的列名行
                body_lines.append(txt)
            else:
                body_lines.append(txt)

        # 列名行过长(本身就接近或超过 max_size,常见于 MinerU 把长段落首行误当
        # 两列表头)时,不再作前缀复制到每个续块——否则前缀即超长,补列名后必超
        # max_size 且无法切分。完整 table_html 仍挂在含表头的首块,内容不丢失。
        if repeat_header and len(repeat_header) > max_size // 2:
            repeat_header = None
        if pieces:
            content = "\n".join(body_lines).strip()
        else:
            content = ""
        if not content and visuals:                     # 纯图无文本:用 caption/标题填充
            caps = [_caption(v) for v in visuals]
            content = " ".join(c for c in caps if c).strip() or f"[图: {hp}]"

        pages = [p for _, p, _ in pieces]
        ps = min(pages) if pages else 0
        pe = max(pages) if pages else 0
        img_paths = [v.get("img_path", "") for v in visuals]

        # 判断本块尾部是否为可重叠的正文(非表格行、非脚注),用于给下一块传递尾句
        last_kind = pieces[-1][2] if pieces else None
        last_text = pieces[-1][0].strip() if pieces else ""
        tail_is_prose = (last_kind is None and last_text
                         and not last_text.startswith("[脚注]"))

        # 超长兜底(正常在 target 处已按句/行收口,prefix <= target;
        # 仅当 target 内无句界且累计到 max 强制切、或单行超 max 时才走到这里):
        #   - 含表格行:优先按整行(\n)切分,不在 ' | ' 单元格分隔符处切断;
        #   - 普通正文:先按句界切,再调 LLM(带 hash 缓存),最后硬切兜底;
        # 任何路径都必须切到 <= max_size,不允许超长 content 原样输出。
        # 若需补列名,先按扣除列名后的预算切正文,再给每块补列名,保证不超 max_size。
        sub_texts = None
        inline_header = None
        # 若本块需补列名(repeat_header),则把列名长度计入,避免补列名后超长
        effective_len = len(content) + (len(repeat_header) + 1 if repeat_header else 0)
        if effective_len > max_size:
            budget = max_size
            if repeat_header:
                budget = max(min_size, max_size - len(repeat_header) - 1)
            has_table_row = any(k and k[0] in ("hdr", "row") for _, _, k in pieces)
            if has_table_row:
                # 表格内容:按整行(\n)打包,不在单元格 ' | ' 处切断;
                # 单行超长时在单元格边界切,保证每个续块自解释。
                # 若本次 flush 内含表头行(hdr),则取首行作为列名,续块复制。
                if tbl_html is not None and body_lines:
                    inline_header = body_lines[0]  # 首行即列名行
                # 列名行过长时不作前缀复制(见上方 repeat_header 守卫),否则补列名后必超长
                if inline_header and len(inline_header) > max_size // 2:
                    inline_header = None
                if inline_header and not repeat_header:
                    budget = max(min_size, max_size - len(inline_header) - 1)
                sub_texts = _split_table_content(content, budget)
                if sub_texts is None:
                    sub_texts = _hard_split(content, budget)
                # 续块(不含列名行的)复制列名到块首
                if sub_texts and (inline_header or repeat_header):
                    hdr_line = inline_header or repeat_header
                    sub_texts = [
                        s if s.startswith(hdr_line)
                        else hdr_line + "\n" + s
                        for s in sub_texts
                    ]
            else:
                sub_texts = _split_by_sentence(content, budget)
                if sub_texts is None and llm_split:
                    sub_texts = _llm_split_cached(content, target)
                if sub_texts is None:
                    sub_texts = _hard_split(content, budget)
        if sub_texts is None and repeat_header:
            content = repeat_header + "\n" + content   # 正常单块:补一次列名
        if sub_texts and repeat_header:
            sub_texts = [repeat_header + "\n" + s for s in sub_texts]
        # 最终保险:若 sub_texts 中仍有超 max 的块(补列名后、或意外情况),强制硬切
        if sub_texts:
            fixed = []
            hdr_line = inline_header or repeat_header
            for s in sub_texts:
                if len(s) <= max_size:
                    fixed.append(s)
                else:
                    # 若补了列名导致超长,去掉已补的列名再硬切
                    if hdr_line and s.startswith(hdr_line + "\n"):
                        body = s[len(hdr_line) + 1:]
                        budget = max(min_size, max_size - len(hdr_line) - 1)
                        parts = _hard_split(body, budget)
                        if parts:
                            fixed.extend(hdr_line + "\n" + pt for pt in parts)
                            continue
                    fixed.extend(_hard_split(s, max_size) or [s[:max_size]])
            sub_texts = fixed

        # 正常单块路径注入跨块 overlap(上一块尾句);sub_texts 已按句界切,不注入
        has_overlap = False
        if sub_texts is None and overlap_prefix and content:
            if len(content) + len(overlap_prefix) + 1 <= max_size:
                content = overlap_prefix + "\n" + content
                has_overlap = True

        if sub_texts:                       # 切成多块,视觉/表 HTML 挂第一块
            for i, st in enumerate(sub_texts):
                state["tseq"] += 1
                cid = f"{source_stem}__t{state['tseq']:05d}"
                text_chunks.append({
                    "chunk_id": cid, "content": st, "embed_text": f"{hp}\n{st}",
                    "source_path": source_path, "source_stem": source_stem,
                    "page_start": ps, "page_end": pe, "heading_path": hp,
                    "image_paths": img_paths if i == 0 else [],
                    "image_descriptions": [],
                    "has_table": bool(tbl_html) and i == 0,
                    "table_html": tbl_html if i == 0 else None,
                    "chunk_index": state["tseq"], "char_count": len(st),
                    "_has_overlap": False,
                })
            emit_visuals(visuals, f"{source_stem}__t{state['tseq']-len(sub_texts)+1:05d}")
            # 超长切分路径不向下一块传 overlap(内含表格行风险)
            return ""
        else:                               # 正常单块
            state["tseq"] += 1
            cid = f"{source_stem}__t{state['tseq']:05d}"
            text_chunks.append({
                "chunk_id": cid, "content": content, "embed_text": f"{hp}\n{content}",
                "source_path": source_path, "source_stem": source_stem,
                "page_start": ps, "page_end": pe, "heading_path": hp,
                "image_paths": img_paths, "image_descriptions": [],
                "has_table": bool(tbl_html), "table_html": tbl_html,
                "chunk_index": state["tseq"], "char_count": len(content),
                "_has_overlap": has_overlap,
            })
            emit_visuals(visuals, cid)
            # 仅当尾部为正文时,返回尾句供下一块重叠
            if tail_is_prose:
                real_tail = content
                if has_overlap and overlap_prefix:
                    # 去掉 overlap 前缀,从真实正文取尾句
                    real_tail = content[len(overlap_prefix) + 1:]
                return _tail_sentence(real_tail)
            return ""

    for sec in sections:
        hp = sec["hp"]
        buf = []                 # 待刷文本:[(text, page, kind), ...],按 "\n" 拼接
        pend_vis = []            # 待挂到下一文本块的图
        carry_overlap = ""       # 上一文本块尾部句,作为下一块前缀(跨块语义重叠)
        for it in sec["parts"]:
            t = it.get("type")
            pidx = it.get("page_idx", 0)
            if t in IMAGE_TYPES:
                pend_vis.append(it)
                # chart 的 footnote(数据来源等,如 "Source: McKinsey...")进正文,
                # image 一般无 footnote。放在图后作为普通文本 piece。
                fn = _footnote_text(it)
                if fn:
                    buf.append((fn, pidx, None))
                continue
            if t == "table":
                html = it.get("table_body", "")
                rows = _table_rows(html)
                if rows:
                    cap = _caption(it)
                    if cap:
                        buf.append((cap, pidx, None))
                    # 判定首行是否为真正的列名行:
                    #   - 非空单元格数 >= 2(单单元格多为 colspan 分组标题,如"A.焊頭")
                    #   - 列名行单元格数不少于第二行的一半(避免 OCR 噪声行被误当列名)
                    # 不满足时,所有行按普通文本入缓冲,不挂 table_html、不复制列名。
                    first_cells = [c for c in rows[0].split("|") if c.strip()]
                    is_header = len(first_cells) >= 2
                    if is_header and len(rows) >= 2:
                        second_cells = [c for c in rows[1].split("|") if c.strip()]
                        if second_cells and len(first_cells) < len(second_cells) / 2:
                            is_header = False
                    if is_header:
                        # 列名行取首行(th/td 均可,MinerU 表头多为首行)。逐行入缓冲,
                        # 行间 "\n" 即句界,在 target 处按行收口;表头行 kind=hdr 携带
                        # 完整 HTML,数据行 kind=row 携带列名文本供跨块时复制。
                        # 超长行立即在单元格边界切分,避免主循环在字符位置硬切截断单元格。
                        header_line = rows[0]
                        if len(header_line) > max_size:
                            hdr_parts = _split_long_line(header_line, max_size)
                            buf.append((hdr_parts[0], pidx, ("hdr", header_line, html)))
                            for seg in hdr_parts[1:]:
                                buf.append((seg, pidx, ("row", header_line)))
                        else:
                            buf.append((header_line, pidx, ("hdr", header_line, html)))
                        for row in rows[1:]:
                            if len(row) > max_size:
                                for seg in _split_long_line(row, max_size):
                                    buf.append((seg, pidx, ("row", header_line)))
                            else:
                                buf.append((row, pidx, ("row", header_line)))
                    else:
                        # 分组标题/无列名表:不挂 HTML、不复制列名,逐行作为普通文本
                        for row in rows:
                            if len(row) > max_size:
                                for seg in _split_long_line(row, max_size):
                                    buf.append((seg, pidx, None))
                            else:
                                buf.append((row, pidx, None))
                    fn = _footnote_text(it)
                    if fn:
                        buf.append((fn, pidx, None))
            else:
                txt = _item_text(it)
                if txt:
                    buf.append((txt, pidx, None))

            # 到达 target 时,在最后一个句号/行尾处收口;未竟部分留作下一块起点。
            while True:
                joined = "\n".join(s for s, _, _ in buf)
                if len(joined) < target:
                    break
                prefix, _ = _cut_at_sentence(joined, target)
                if prefix:
                    pre, buf = _split_pieces_at(buf, len(prefix))
                    carry_overlap = flush_block(pre, pend_vis, hp, carry_overlap) or ""
                    pend_vis = []
                    continue
                # target(500)内没有完整句:在 max_size 内继续攒以补完当前句;
                # 一旦达到 max 仍无句号:有句界则在句界收口;完全无句界时把整块
                # 交给 flush_block,由其 LLM 兜底 + 硬切保证 <= max_size。
                if len(joined) >= max_size:
                    prefix, _ = _cut_at_sentence(joined, max_size)
                    if prefix:
                        pre, buf = _split_pieces_at(buf, len(prefix))
                    else:
                        pre, buf = buf, []
                    carry_overlap = flush_block(pre, pend_vis, hp, carry_overlap) or ""
                    pend_vis = []
                    continue
                break
        # 段末剩余(含纯图):章节结束,跨章节不延续 overlap
        if buf or pend_vis:
            flush_block(buf, pend_vis, hp, carry_overlap)
        carry_overlap = ""

    # ---- 3. 合并过小尾块(仅同 section,不跨标题;合并后不超过 max_size)----
    # 带 overlap 前缀的块不参与被合并(它已含上一块尾句,再并入会造成语义重复);
    # 也不把小块并入一个带 overlap 的块之外(维持简单规则)。
    merged = []
    for tc in text_chunks:
        tc_has_overlap = tc.pop("_has_overlap", False)
        merged_len = len(merged[-1]["content"]) if merged else 0
        if (not tc_has_overlap and merged and tc["char_count"] < min_size
                and not tc["image_paths"]
                and merged[-1]["heading_path"] == tc["heading_path"]
                and merged_len + tc["char_count"] + 1 <= max_size):
            p = merged[-1]
            p["content"] = (p["content"] + "\n" + tc["content"]).strip()
            p["embed_text"] = f"{p['heading_path']}\n{p['content']}"
            p["page_end"] = max(p["page_end"], tc["page_end"])
            p["char_count"] = len(p["content"])
            # 被并入块的图(若有)改挂到 p
            if tc["image_paths"]:
                p["image_paths"] = p["image_paths"] + tc["image_paths"]
        else:
            merged.append(tc)
    for i, tc in enumerate(merged, 1):
        tc["chunk_index"] = i
        tc.pop("_has_overlap", None)
        tc.pop("_merged_with_overlap", None)

    return merged, image_chunks


# ---------------------------------------------------------------------------
def chunk_one_pdf(stem, auto_dir, source_path, do_describe=True, **kw):
    """读 <stem>_content_list.json,返回 (text_chunks, image_chunks)。
    若 do_describe=True,对图像块调 Doubao 生成描述。"""
    cl = json.load(open(os.path.join(auto_dir, stem + "_content_list.json"), encoding="utf-8"))
    tc, ic = chunk_content_list(cl, source_path=source_path, source_stem=stem,
                                auto_dir=auto_dir, **kw)
    if do_describe and ic:
        import ark_client
        cache = ark_client.load_desc_cache(auto_dir)
        for v in ic:
            if (v.get("description") or "").strip():
                continue
            try:
                desc, cache = ark_client.describe_with_cache(
                    v["img_relname"], v["image_path"], auto_dir, cache)
                v["description"] = desc
            except Exception as e:
                print(f"    [desc fail] {v['img_relname']}: {str(e)[:80]}")
    return tc, ic


if __name__ == "__main__":
    # 自测:对 Oxford ALD Operation Manual 跑分块,打印样例
    CLEAN = r"D:\清洗文件\pdf\0001 半导体设备资料合集\ALD原子沉积资料\ALD原子沉积资料"
    SRC   = r"D:\180-半导体设备相关资料！\0001 半导体设备资料合集\ALD原子沉积资料\ALD原子沉积资料"
    import config as C  # noqa: 仅取默认参数
    stem = "Oxford ALD Operation Manual"
    auto = os.path.join(CLEAN, stem, "auto")
    src_pdf = os.path.join(SRC, stem + ".pdf")
    tc, ic = chunk_one_pdf(stem, auto, src_pdf,
                           target=C.CHUNK_TARGET, max_size=C.CHUNK_MAX, min_size=C.CHUNK_MIN)
    print(f"text chunks: {len(tc)}   image chunks: {len(ic)}")
    # 字数分布
    lens = sorted(t["char_count"] for t in tc)
    print(f"chunk 字数: min={lens[0]} 中位={lens[len(lens)//2]} max={lens[-1]}")
    over = [t for t in tc if t["char_count"] > C.CHUNK_MAX]
    print(f"超 max({C.CHUNK_MAX}) 的 chunk: {len(over)}(单 item 过长,需 LLM 兜底)")
    for t in over:
        print(f"   [{t['chunk_id']}] {t['char_count']}字 page {t['page_start']}-{t['page_end']} :: {t['content'][:80]}...")
    print("\n=== 样例 text chunk(前3个,完整) ===")
    for t in tc[:3]:
        print(f"\n[{t['chunk_id']}] page {t['page_start']}-{t['page_end']} | {t['char_count']}字 | 图{len(t['image_paths'])} | 表{t['has_table']}")
        print(f"heading: {t['heading_path']}")
        print(f"content:\n{t['content']}")
    print("\n=== image chunks(前3个) ===")
    for v in ic[:3]:
        print(f"[{v['chunk_id']}] p{v['page_num']} {v['item_type']} -> parent {v['parent_text_chunk_id']} | caption: {v['caption'][:60]!r}")
