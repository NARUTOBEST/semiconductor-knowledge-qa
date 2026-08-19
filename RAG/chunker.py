# -*- coding: utf-8 -*-
"""content_list.json (v1) -> text chunks + image chunks。

确定性主路径:
  1. 按 L1/L2 标题切 section(标题来自 text_level);
  2. section 内按 MinerU text-item 边界贪心打包到 target 字;
  3. 单段超过 max 且无内部边界可切 -> 调 llm_split 兜底(由调用方注入,可为 None);
  4. 不足 min 的尾块并入同 section 上一 chunk;
  5. image/chart 按阅读序挂到所在 text chunk,并各生成一条 image chunk。
header/footer/page_number 直接丢弃。表格留 HTML(并 strip 后入 content 供检索),公式留 LaTeX。
"""
import os, re, json, sys

# sys.path (chunk_one_pdf 里 import ark_client 做图描述)
_CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")
_RAG = os.path.dirname(os.path.abspath(__file__))
for _p in (_CONFIG, _RAG):
    if _p not in sys.path:
        sys.path.insert(0, _p)

FILTER_TYPES = {"header", "footer", "page_number"}
HEADING_LEVELS = {1, 2}
IMAGE_TYPES = {"image", "chart"}


def _strip_html(s):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", s or "")).strip()


def _item_text(it):
    """非图非表 item -> 并入 content 的文本。"""
    t = it.get("type")
    if t == "equation":
        return it.get("text", "")                       # LaTeX
    if t == "text":
        return it.get("text", "")
    if t in ("list", "code"):
        return it.get("text", "") or it.get("content", "")
    return it.get("text", "") or it.get("content", "")


def _caption(it):
    for k in ("image_caption", "chart_caption", "table_caption"):
        c = it.get(k)
        if c:
            return " ".join(c).strip()
    return ""


def _split_by_sentence(text, max_size):
    """按句末标点切,贪心打包到 <= max_size。返回多块列表;无法干净切(有单句>max)返回 None。"""
    parts = re.split(r"(?<=[。！？])|(?<=[.!?])\s+|\n+", text)
    out, cur = [], ""
    for p in parts:
        if not p:
            continue
        if not cur:
            cur = p
        elif len(cur) + len(p) + 1 <= max_size:
            cur = cur + " " + p
        else:
            out.append(cur)
            cur = p
    if cur:
        out.append(cur)
    if any(len(s) > max_size for s in out):     # 有单句超 max,确定切分失败
        return None
    return out if len(out) > 1 else None        # 只有一块=没切,返回 None


def chunk_content_list(content_list, *, source_path, source_stem, auto_dir,
                       target=500, max_size=600, min_size=150, llm_split=None):
    """返回 (text_chunks, image_chunks),均为 dict 列表。"""
    items = [it for it in content_list if it.get("type") not in FILTER_TYPES]

    # ---- 1. 按 L1/L2 标题切 section ----
    sections = []
    stack = {}        # {1: 文档标题, 2: 节标题}
    cur = None
    for it in items:
        t = it.get("type")
        if t == "text" and it.get("text_level") in HEADING_LEVELS:
            if cur:
                sections.append(cur)
            lvl = it["text_level"]
            stack[lvl] = it.get("text", "").strip()
            for l in [x for x in stack if x > lvl]:
                del stack[l]
            hp = " > ".join(stack[l] for l in sorted(stack) if stack.get(l)) or source_stem
            cur = {"hp": hp, "parts": [it]}          # 标题本身作为段首
        else:
            if cur is None:
                cur = {"hp": source_stem, "parts": []}
            cur["parts"].append(it)
    if cur:
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

    def emit(hp, pieces, pages, visuals, tables):
        content = "\n".join(s for s, _ in pieces).strip()
        if not content and not visuals:
            return
        if not content and visuals:                     # 纯图无文本:用 caption/标题填充
            caps = [_caption(v) for v in visuals]
            content = " ".join(c for c in caps if c).strip() or f"[图: {hp}]"
        ps = min(pages) if pages else 0
        pe = max(pages) if pages else 0
        img_paths = [v.get("img_path", "") for v in visuals]
        tbl_html = tables[0] if tables else None

        # 超长:先句界确定切分,失败再 LLM 兜底
        sub_texts = None
        if len(content) > max_size:
            sub_texts = _split_by_sentence(content, max_size)
            if sub_texts is None and llm_split:
                sub_texts = llm_split(content, target)

        if sub_texts:                       # 切成多块,视觉元素挂第一块
            for i, st in enumerate(sub_texts):
                state["tseq"] += 1
                cid = f"{source_stem}__t{state['tseq']:05d}"
                text_chunks.append({
                    "chunk_id": cid, "content": st, "embed_text": f"{hp}\n{st}",
                    "source_path": source_path, "source_stem": source_stem,
                    "page_start": ps, "page_end": pe, "heading_path": hp,
                    "image_paths": img_paths if i == 0 else [],
                    "image_descriptions": [],
                    "has_table": bool(tables) and i == 0,
                    "table_html": tbl_html if i == 0 else None,
                    "chunk_index": state["tseq"], "char_count": len(st),
                })
            emit_visuals(visuals, f"{source_stem}__t{state['tseq']-len(sub_texts)+1:05d}")
        else:                               # 正常单块(含超长但无 llm_split 的情况)
            state["tseq"] += 1
            cid = f"{source_stem}__t{state['tseq']:05d}"
            text_chunks.append({
                "chunk_id": cid, "content": content, "embed_text": f"{hp}\n{content}",
                "source_path": source_path, "source_stem": source_stem,
                "page_start": ps, "page_end": pe, "heading_path": hp,
                "image_paths": img_paths, "image_descriptions": [],
                "has_table": bool(tables), "table_html": tbl_html,
                "chunk_index": state["tseq"], "char_count": len(content),
            })
            emit_visuals(visuals, cid)

    for sec in sections:
        hp = sec["hp"]
        pieces, pages, visuals, tables = [], [], [], []
        for it in sec["parts"]:
            t = it.get("type")
            pidx = it.get("page_idx", 0)
            if t in IMAGE_TYPES:
                visuals.append(it)
                pages.append(pidx)
            elif t == "table":
                html = it.get("table_body", "")
                tables.append(html)
                pages.append(pidx)
                txt = _strip_html(html)                 # 表格文本入 content 供检索
                if len(txt) > 400:                      # 大表格截断,完整内容在 table_html
                    txt = txt[:400] + " ...[表格内容已截断,完整见 table_html]"
                if txt:
                    pieces.append((txt, pidx))
            else:
                txt = _item_text(it)
                if txt:
                    cur_tot = sum(len(s) for s, _ in pieces)
                    # 预检:加上这条会超 max 且当前已有内容 -> 先闭合,避免 chunk 超 max
                    if cur_tot > 0 and cur_tot + len(txt) > max_size:
                        emit(hp, pieces, pages, visuals, tables)
                        pieces, pages, visuals, tables = [], [], [], []
                    pieces.append((txt, pidx))
                    pages.append(pidx)
                    if sum(len(s) for s, _ in pieces) >= target:
                        emit(hp, pieces, pages, visuals, tables)
                        pieces, pages, visuals, tables = [], [], [], []
        emit(hp, pieces, pages, visuals, tables)        # 段末剩余

    # ---- 3. 合并过小尾块(仅同 section,不跨标题)----
    merged = []
    for tc in text_chunks:
        if (merged and tc["char_count"] < min_size and not tc["image_paths"]
                and merged[-1]["heading_path"] == tc["heading_path"]):
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
    CLEAN = r"D:\清洗文件\0001 半导体设备资料合集\ALD原子沉积资料\ALD原子沉积资料"
    SRC   = r"C:\project3\180-半导体设备相关资料！\0001 半导体设备资料合集\ALD原子沉积资料\ALD原子沉积资料"
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
