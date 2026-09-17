# -*- coding: utf-8 -*-
r"""MinerU 清洗结果入库前的质量体检(批量、静态、秒级)。

扫描 <root> 下所有 auto\<stem>_content_list.json,对每份 PDF 跑一组客观红旗检查,
亮 ERROR 的 PDF 必须人工打开同目录 <stem>_layout.pdf 复核/重解析;WARN 建议复核。

用法:
  python -m RAG.pdf.check                       # 扫 C.CLEAN_ROOT,打印汇总
  python RAG/pdf/check.py --root D:\清洗文件\pdf\0001 半导体设备资料合集
  python RAG/pdf/check.py --json report.json --verbose
  python RAG/pdf/check.py --limit 20 --chunks
退出码:任一 PDF 有 ERROR -> 1;否则 0(WARN 不阻断,可接进入库流水线当门禁)。

只依赖标准库 + config(取 CLEAN_ROOT);--chunks 时才导入 chunker,不触碰 embedding/GPU。
"""
import os, sys, json, re, time, argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
_RAG = os.path.dirname(_HERE)
_PROJECT = os.path.dirname(_RAG)
_CONFIG = os.path.join(_PROJECT, "config")
for _p in (_HERE, _CONFIG, _RAG):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import config as C  # noqa: E402

# ---- 阈值(集中在此,便于调)----
LONG_ITEM_CHARS = 3000      # 单个 text/list item 超过此长 -> 疑似双栏串行
GARBLE_REPLACE_ABS = 5      # U+FFFD 替换符绝对数阈值
GARBLE_REPLACE_RATIO = 0.003  # 或占比 > 0.3%
IMG_HEAVY_RATIO = 0.60      # image/chart item 占比
IMG_HEAVY_CHARS_PER_PAGE = 100  # 且每页文本字符 < 此值 -> 扫描件未 OCR
EMPTY_PAGE_RATIO = 0.20     # 无正文内容的页占比(仅当页数 > 5)
EMPTY_PAGE_MIN_PAGES = 5
GLUE_DENSE_PAGE_HITS = 3    # 单页单词粘连命中 > 此值 -> 该页疑似 OCR 缺空格
GLUE_DENSE_PAGES = 2        # 这样的"密集页"达到此数 -> WARN(整本厚书的总量会被稀释,按密集页数判定)
RESIDUAL_HTML = 50          # 残留实体总数阈值(达此数才在 verbose 提示;chunker P4 已切分时解码,不阻断)
LONG_ITEM_LINES = 5         # 超长 item 含 >= 此数短行
LONG_ITEM_NUMEND_RATIO = 0.4  # 且 >= 此比例行以数字结尾 -> 判定为目录/清单,不报双栏串行

# 正则
# 真·OCR 单词粘连:超长小写串(>=24,正常英语/化学词极少更长)且内嵌常见英文功能词。
# 旧规则 [a-z]{4,}[A-Z][a-z] 抓的是驼峰标识符(CoarseHorVert/ACLEntrySelector),属误报;
# 全小写粘连(thechamber)反而匹配不到。新规则两者都修正。
_GLUE_TOKEN_RE = re.compile(r"[a-z]{24,}")
_GLUE_FUNC_RE = re.compile(
    r"(?:the|and|that|with|for|are|was|were|this|which|from|been|have|into|using|used|will|than|then|these|those)")
_ENTITY_RE = re.compile(r"&(?:#x?[0-9a-fA-F]+|[a-zA-Z]+);")
_TAG_RE = re.compile(r"<[a-zA-Z/!][^>]*>")
# 内容 item 类型(判定某页是否有可读内容):图/表也算"有内容",图页(专利图/设备照)不是空页
_CONTENT_TYPES = ("text", "table", "list", "code", "equation", "image", "chart")


def _finding(severity, code, message, page=None):
    """构造一条 finding。page 为可选页码(1 个或多个)。"""
    f = {"severity": severity, "code": code, "message": message}
    if page is not None:
        f["page"] = page
    return f


def _item_all_text(it):
    """汇总一个 item 的全部可读文本(正文/list/code/table_body/caption/footnote)。

    表格单元格里同样可能有 OCR 粘连/乱码/实体(如 FIJI 的 "HeaterLocation"),
    所以体检要把 table_body 也纳入扫描,而不只看 text 字段。
    """
    parts = [it.get("text") or ""]
    t = it.get("type")
    if t == "list":
        parts.append(" ".join(it.get("list_items") or []))
    elif t == "code":
        parts.append(it.get("code_body") or "")
    if t == "table":
        parts.append(it.get("table_body") or "")
    for k in ("image_caption", "chart_caption", "table_caption",
              "image_footnote", "chart_footnote", "table_footnote"):
        v = it.get(k)
        if isinstance(v, list):
            parts.append(" ".join(v))
        elif v:
            parts.append(str(v))
    return " ".join(p for p in parts if p)


def _count_real_glue(text):
    """统计真·OCR 单词粘连(长小写串内嵌常见英文功能词,如 anyofthecomponentsused)。

    排除两类误报:
    - 驼峰标识符(CoarseHorVert/ACLEntrySelector):含大写,本正则不匹配;
    - 化学/德语合法长词(trimethylsilyldiethylamine、Nutzungserscheinungen):
      不含 the/and/with 等英文功能词。
    OCR 把整句空格吃掉时,这些功能词必然出现在长小写串里。
    """
    n = 0
    for tok in _GLUE_TOKEN_RE.findall(text):
        if _GLUE_FUNC_RE.search(tok[3:]):  # 跳过开头被截断的半个词
            n += 1
    return n


def _looks_like_toc(text):
    """超长 text item 是否为目录/索引/图表清单(而非双栏串行正文)。

    判据:带页码引导点(....482),或多行短行且高比例以数字结尾(目录条目)。
    这些是正常的书前件,不是解析缺陷。
    """
    if "..." in text or "．．" in text or "…" in text:
        return True
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if len(lines) >= LONG_ITEM_LINES:
        num_end = sum(1 for ln in lines if re.search(r"\d+\s*$", ln[-8:]))
        if num_end / len(lines) >= LONG_ITEM_NUMEND_RATIO:
            return True
    return False


def check_content_list(cl, auto_dir):
    """对单个已解析的 content_list(对象列表)跑检查,返回 (metrics, findings)。"""
    findings = []
    n = len(cl)

    # 页码范围(部分 PDF 的 page_idx 从 0 开始,可能不连续)
    page_idxs = sorted({it.get("page_idx") for it in cl if "page_idx" in it})
    n_pages = len(page_idxs)

    # 按类型统计
    type_counts = {}
    text_chars = 0
    n_replace = 0          # U+FFFD 真乱码
    n_pua = 0              # 私有区字形(符号字体,正常,仅记录)
    n_long_items = 0
    n_glue = 0
    n_entity = 0
    n_tag = 0
    missing_imgs = []
    long_pages = []
    glue_pages = {}

    # 每页有哪些类型
    page_types = {p: set() for p in page_idxs}
    page_has_any = set(page_idxs)  # 出现过的页都算"有 item"

    for it in cl:
        t = it.get("type")
        type_counts[t] = type_counts.get(t, 0) + 1
        pg = it.get("page_idx")
        if pg in page_types:
            page_types[pg].add(t)

        txt = _item_all_text(it)
        text_chars += len(txt)
        n_replace += txt.count("\ufffd")
        n_pua += sum(1 for ch in txt if 0xE000 <= ord(ch) <= 0xF8FF)
        n_entity += len(_ENTITY_RE.findall(txt))

        # long-text-item \u53ea\u770b\u6b63\u6587(text/list/code):3000 \u5b57\u7684\u6bb5\u843d\u662f\u53cc\u680f\u4e32\u884c\u4fe1\u53f7;
        # \u5927\u8868 table_body \u8d85\u957f\u662f\u6b63\u5e38\u7684(chunker \u6309\u884c\u6253\u5305),\u4e0d\u7b97\u7ea2\u65d7\u3002
        if t in ("text", "list", "code") and len(txt) > LONG_ITEM_CHARS \
                and not _looks_like_toc(txt):
            n_long_items += 1
            long_pages.append(pg)

        # 真·OCR 单词粘连按页统计(排除驼峰标识符与化学/德语长词,见 _count_real_glue)。
        hits = _count_real_glue(txt)
        if hits and pg is not None:
            glue_pages[pg] = glue_pages.get(pg, 0) + hits

        if t in ("image", "chart"):
            rel = it.get("img_path") or ""
            if not rel or not os.path.exists(os.path.join(auto_dir, rel)):
                missing_imgs.append(rel or "<empty img_path>")

    n_glue = sum(glue_pages.values())

    # ---- ERROR ----
    if n and n_replace >= max(GARBLE_REPLACE_ABS, int(text_chars * GARBLE_REPLACE_RATIO)):
        findings.append(_finding("ERROR", "garbled",
                                 f"OCR 乱码替换符(U+FFFD){n_replace} 个", ))
    if missing_imgs:
        findings.append(_finding("ERROR", "missing-image",
                                 f"{len(missing_imgs)} 个图片文件缺失,首个: {missing_imgs[0]!r}"))

    # ---- WARN ----
    n_img = type_counts.get("image", 0) + type_counts.get("chart", 0)
    if n and n_img / n > IMG_HEAVY_RATIO and n_pages and text_chars < n_pages * IMG_HEAVY_CHARS_PER_PAGE:
        findings.append(_finding("WARN", "image-heavy",
                                 f"图片占比 {n_img/n:.0%},文本仅 {text_chars} 字(疑似扫描件未 OCR)"))
    # 注:早期版本对 >3000 字的 text item 报 long-text-item(疑似双栏串行),实测经
    # 目录过滤后剩余的长 item 均为正确阅读顺序的长正文(双栏论文 MinerU 已按左→右栏
    # 正确提取),且 chunker 会强制按句子/行切到 max_size,超长不影响入库,故不再告警。
    # 阅读顺序错位属语义问题,静态规则抓不准,交给人工翻 _layout.pdf。
    # 单词粘连按"密集页数"判定:整本厚书的总量会被页数稀释(171 页里只 6 页有问题),
    glue_dense_pages = sorted(p for p, v in glue_pages.items()
                              if v > GLUE_DENSE_PAGE_HITS)
    if len(glue_dense_pages) >= GLUE_DENSE_PAGES:
        findings.append(_finding("WARN", "word-glue",
                                 f"{len(glue_dense_pages)} 页出现 OCR 单词粘连(如 'thechamber'),"
                                 f"共 {n_glue} 处(chunker P5 切分时用 wordninja 自动补空格,不影响入库)",
                                 page=glue_dense_pages[:6]))
    # 残留 HTML 实体(&#x27;/&quot;/&amp; 等):阈值设较高(50),且 chunker P4 在切分时
    # 已 html.unescape 兜底解码,不影响入库内容,故仅作源脏提示,不阻断。内联标签
    # (<sup>/<img>)与 table_body 的 HTML 是 MinerU 合法标记,不算缺陷,不报。
    if n_entity > RESIDUAL_HTML:
        findings.append(_finding("WARN", "residual-entities",
                                 f"残留 HTML 实体 {n_entity} 处(chunker 切分时已解码,不影响入库)"))

    # 整页只有 header/footer/page_number(既无文本也无图)的页,占比过高才提示——
    # 多为章节分隔页/封面;image/chart 页算有内容,不计入(专利图页/设备照是正常的)。
    if n_pages > EMPTY_PAGE_MIN_PAGES:
        no_content = [p for p, ts in page_types.items()
                      if not (ts & set(_CONTENT_TYPES))]
        if len(no_content) > n_pages * EMPTY_PAGE_RATIO:
            findings.append(_finding("WARN", "empty-content-pages",
                                     f"{len(no_content)}/{n_pages} 页既无文本也无图(章节分隔页或漏解析)"))

    metrics = {
        "items": n, "pages": n_pages, "text_chars": text_chars,
        "images": n_img, "tables": type_counts.get("table", 0),
        "type_counts": type_counts,
        "pua_chars": n_pua, "replace_chars": n_replace,
        "glue_hits": n_glue, "glue_pages": len(glue_pages),
        "long_items": n_long_items,
        "has_layout_pdf": _has_layout_pdf(auto_dir),
    }
    return metrics, findings


def _has_layout_pdf(auto_dir):
    """auto 目录同级或自身是否有 *_layout.pdf 供人工复核。"""
    for d in (auto_dir, os.path.dirname(auto_dir)):
        try:
            if any(f.endswith("_layout.pdf") for f in os.listdir(d)):
                return True
        except OSError:
            pass
    return False


def discover(root):
    r"""复用 ingest.py 的发现逻辑:遍历 auto\<stem>_content_list.json(跳过 _v2)。"""
    out = []
    for dp, _dn, fn in os.walk(root):
        if os.path.basename(dp) != "auto":
            continue
        for f in fn:
            if f.endswith("_content_list.json") and "_v2" not in f:
                stem = f[:-len("_content_list.json")]
                out.append((stem, dp, os.path.join(dp, f)))
    out.sort()
    return out


def check_one(stem, auto_dir, path):
    """检查单个 PDF;返回结果 dict。JSON 解析失败也作为 ERROR 返回,不抛出。"""
    result = {"stem": stem, "auto_dir": auto_dir, "path": path,
              "metrics": None, "findings": []}
    try:
        if os.path.getsize(path) < 10:
            result["findings"].append(_finding("ERROR", "parse-error", "文件为空或过小"))
            return result
        with open(path, encoding="utf-8") as f:
            cl = json.load(f)
        if not isinstance(cl, list):
            result["findings"].append(_finding("ERROR", "parse-error",
                                              f"顶层不是列表(是 {type(cl).__name__})"))
            return result
    except Exception as e:
        result["findings"].append(_finding("ERROR", "parse-error", f"JSON 解析失败: {e}"))
        return result
    metrics, findings = check_content_list(cl, auto_dir)
    result["metrics"] = metrics
    result["findings"] = findings
    return result


def maybe_check_chunks(result):
    """--chunks:跑 chunker,报告 >CHUNK_MAX 的块数(chunker P2 后应恒为 0)。"""
    import chunker
    if any(f["severity"] == "ERROR" and f["code"] == "parse-error"
           for f in result["findings"]):
        return
    try:
        with open(result["path"], encoding="utf-8") as f:
            cl = json.load(f)
        tc, _ic = chunker.chunk_content_list(
            cl, source_path=result["stem"] + ".pdf", source_stem=result["stem"],
            auto_dir=result["auto_dir"], target=C.CHUNK_TARGET,
            max_size=C.CHUNK_MAX, min_size=C.CHUNK_MIN)
        over = [c for c in tc if c["char_count"] > C.CHUNK_MAX]
        if over:
            result["findings"].append(_finding(
                "ERROR", "oversized-chunk",
                f"{len(over)}/{len(tc)} 块 > CHUNK_MAX({C.CHUNK_MAX}),"
                f"最大 {max(c['char_count'] for c in over)}"))
        result.setdefault("metrics", {})["chunks"] = len(tc)
        result["metrics"]["oversized_chunks"] = len(over)
    except Exception as e:
        result["findings"].append(_finding("ERROR", "chunker-error", f"切分失败: {e}"))


def print_report(results, verbose):
    n_err = sum(1 for r in results
                if any(f["severity"] == "ERROR" for f in r["findings"]))
    n_warn = sum(1 for r in results
                 if any(f["severity"] == "WARN" for f in r["findings"]))
    # 按 ERROR 数、WARN 数降序,有问题的排前面
    def sev_count(r, sev):
        return sum(1 for f in r["findings"] if f["severity"] == sev)
    flagged = [r for r in results if r["findings"]]
    flagged.sort(key=lambda r: (sev_count(r, "ERROR"), sev_count(r, "WARN")),
                 reverse=True)

    print("=" * 78)
    print(f"MinerU 清洗体检: {len(results)} 份 PDF | ERROR {n_err} | WARN {n_warn} | "
          f"干净 {len(results) - len(flagged)}")
    print("=" * 78)
    if not flagged:
        print("✅ 无红旗。")
        return

    for r in flagged:
        errs = [f for f in r["findings"] if f["severity"] == "ERROR"]
        warns = [f for f in r["findings"] if f["severity"] == "WARN"]
        tag = "❌" if errs else "⚠️ "
        codes = ",".join(f["code"] for f in (errs + warns))
        m = r.get("metrics") or {}
        print(f"\n{tag} {r['stem'][:60]}  E{len(errs)} W{len(warns)}  "
              f"[{codes}]  (pages={m.get('pages','?')}, items={m.get('items','?')})")
        if verbose:
            for f in r["findings"]:
                pg = f" p{f['page']}" if f.get("page") else ""
                print(f"     {f['severity']:5} {f['code']:18}{pg} {f['message']}")
            if not m.get("has_layout_pdf"):
                print("     (未找到 *_layout.pdf)")


def main():
    ap = argparse.ArgumentParser(description="MinerU 清洗结果入库前质量体检")
    ap.add_argument("--root", default=C.CLEAN_ROOT, help="清洗根目录(默认 C.CLEAN_ROOT)")
    ap.add_argument("--json", dest="json_out", help="把完整报告写入此 JSON 文件")
    ap.add_argument("--limit", type=int, default=0, help="只扫前 N 份(调试抽样)")
    ap.add_argument("--chunks", action="store_true", help="额外跑 chunker 检查超长块")
    ap.add_argument("--verbose", "-v", action="store_true", help="打印每条 finding 详情")
    args = ap.parse_args()

    if not os.path.isdir(args.root):
        print(f"根目录不存在: {args.root}", file=sys.stderr)
        return 2

    t0 = time.time()
    pdfs = discover(args.root)
    if args.limit:
        pdfs = pdfs[:args.limit]
    print(f"扫描根目录: {args.root}\n发现 {len(pdfs)} 份 content_list.json"
          f"{' (--chunks 已开启,会较慢)' if args.chunks else ''} ...")

    results = []
    for i, (stem, auto_dir, path) in enumerate(pdfs, 1):
        r = check_one(stem, auto_dir, path)
        if args.chunks:
            maybe_check_chunks(r)
        results.append(r)
        if i % 100 == 0:
            print(f"  ...{i}/{len(pdfs)}")

    print_report(results, args.verbose)
    dt = time.time() - t0
    n_err = sum(1 for r in results
                if any(f["severity"] == "ERROR" for f in r["findings"]))
    print(f"\n耗时 {dt:.1f}s。" + (" 有 ERROR,退出码 1。" if n_err else " 无 ERROR。"))

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump({"root": args.root, "total": len(results),
                       "error_pdfs": n_err, "results": results},
                      f, ensure_ascii=False, indent=2)
        print(f"完整报告已写入: {args.json_out}")

    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
