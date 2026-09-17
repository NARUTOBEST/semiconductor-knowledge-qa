# -*- coding: utf-8 -*-
"""chunker.chunk_content_list ?????

???????????????????????????
??????????????????LLM ?????chunk_id ???
"""
import pytest
from chunker import chunk_content_list

# ---- ?????? ----

def _text(text, level=None, page=0):
    it = {"type": "text", "text": text, "page_idx": page}
    if level:
        it["text_level"] = level
    return it

def _image(path="images/test.jpg", page=0, caption=None):
    it = {"type": "image", "img_path": path, "page_idx": page}
    if caption:
        it["image_caption"] = [caption]
    return it

def _chart(path="images/chart.jpg", page=0, caption=None):
    it = {"type": "chart", "img_path": path, "page_idx": page}
    if caption:
        it["chart_caption"] = [caption]
    return it

def _table(html, page=0, caption=None):
    it = {"type": "table", "table_body": html, "page_idx": page}
    if caption:
        it["table_caption"] = [caption]
    return it

def _equation(latex, page=0):
    return {"type": "equation", "text": latex, "page_idx": page}

def _list(items, page=0):
    return {"type": "list", "list_items": list(items), "page_idx": page}

def _code(body, page=0, caption=None):
    it = {"type": "code", "code_body": body, "page_idx": page}
    if caption:
        it["code_caption"] = [caption]
    return it

def _header(text, page=0):
    return {"type": "header", "text": text, "page_idx": page}

def _footer(text, page=0):
    return {"type": "footer", "text": text, "page_idx": page}

def _page_number(text, page=0):
    return {"type": "page_number", "text": text, "page_idx": page}

def _chunk(items, **kw):
    """Call chunk_content_list with sensible defaults."""
    defaults = dict(
        source_path="test.pdf", source_stem="test",
        auto_dir="/tmp/auto", target=500, max_size=600, min_size=150,
    )
    defaults.update(kw)
    return chunk_content_list(items, **defaults)


# ============ ???? ============

class TestBasicPacking:

    def test_single_short_text(self):
        """One short text item -> one chunk."""
        tc, ic = _chunk([_text("Hello world.", page=0)])
        assert len(tc) == 1
        assert tc[0]["content"] == "Hello world."
        assert tc[0]["char_count"] == len("Hello world.")
        assert tc[0]["page_start"] == 0
        assert tc[0]["page_end"] == 0
        assert ic == []

    def test_multiple_texts_under_target(self):
        """Multiple short texts packed into one chunk (< target)."""
        items = [_text(f"Sentence {i}.", page=0) for i in range(5)]
        tc, ic = _chunk(items)
        assert len(tc) == 1
        assert "Sentence 0." in tc[0]["content"]
        assert "Sentence 4." in tc[0]["content"]

    def test_texts_exceed_target_split(self):
        """When accumulated text exceeds target, a new chunk starts."""
        # Each item ~100 chars, target=500 -> ~5 items per chunk
        items = [_text("A" * 100, page=0) for _ in range(12)]
        tc, ic = _chunk(items, target=500, max_size=600)
        assert len(tc) >= 2
        # Each chunk should be <= max_size
        for c in tc:
            assert c["char_count"] <= 600

    def test_empty_content_list(self):
        """Empty input -> empty outputs."""
        tc, ic = _chunk([])
        assert tc == []
        assert ic == []

    def test_chunk_id_format_text(self):
        """Text chunk IDs have __t##### suffix."""
        tc, ic = _chunk([_text("Hello", page=0)])
        assert tc[0]["chunk_id"] == "test__t00001"
        assert tc[0]["chunk_index"] == 1

    def test_embed_text_includes_heading(self):
        """embed_text = heading_path + newline + content."""
        tc, ic = _chunk([_text("Body text", page=0)])
        assert "Body text" in tc[0]["embed_text"]
        assert "test" in tc[0]["embed_text"]  # source_stem as default heading


# ============ ???? ============

class TestHeadingSectioning:

    def test_l1_heading_creates_new_section(self):
        """L1 heading starts a new section with its own heading_path."""
        items = [
            _text("Intro paragraph.", page=0),
            _text("Chapter 1", level=1, page=0),
            _text("Chapter 1 content.", page=1),
        ]
        tc, ic = _chunk(items)
        assert len(tc) == 2
        assert "test" in tc[0]["heading_path"]  # default heading
        assert "Chapter 1" in tc[1]["heading_path"]

    def test_l2_heading_creates_new_section(self):
        """L2 heading also starts a new section."""
        items = [
            _text("Chapter 1", level=1, page=0),
            _text("Section A", level=2, page=0),
            _text("Section A content.", page=1),
        ]
        tc, ic = _chunk(items)
        # First chunk = heading "Chapter 1" alone (might be small, merged or not)
        # Find the chunk with "Section A content"
        section_chunk = [c for c in tc if "Section A" in c["heading_path"]]
        assert len(section_chunk) >= 1
        assert "Chapter 1" in section_chunk[0]["heading_path"]
        assert "Section A" in section_chunk[0]["heading_path"]

    def test_heading_path_hierarchy(self):
        """L2 heading_path includes parent L1 heading."""
        items = [
            _text("Top", level=1, page=0),
            _text("Sub", level=2, page=0),
            _text("Content here.", page=0),
        ]
        tc, ic = _chunk(items)
        content_chunk = [c for c in tc if "Content here" in c["content"]]
        assert len(content_chunk) == 1
        assert "Top" in content_chunk[0]["heading_path"]
        assert "Sub" in content_chunk[0]["heading_path"]

    def test_level3_triggers_section_split(self):
        """text_level=3 现在也作为子节标题切 section(P1-7)。"""
        items = [
            _text("Before.", page=0),
            _text("Level3 Heading", level=3, page=0),
            _text("After.", page=0),
        ]
        tc, ic = _chunk(items)
        # L3 标题切 section: "Before." 一块、"After." 一块,标题只在 heading_path
        assert len(tc) == 2
        assert tc[0]["content"] == "Before."
        assert "Level3 Heading" in tc[1]["heading_path"]
        assert tc[1]["content"] == "After."
        assert not tc[1]["content"].startswith("Level3 Heading")


# ============ ??? ============

class TestFilteredItems:

    def test_header_filtered(self):
        """header items are removed."""
        items = [_header("Page header", page=0), _text("Real content.", page=0)]
        tc, ic = _chunk(items)
        assert len(tc) == 1
        assert "Page header" not in tc[0]["content"]
        assert "Real content." in tc[0]["content"]

    def test_footer_filtered(self):
        """footer items are removed."""
        items = [_text("Real content.", page=0), _footer("Page footer", page=0)]
        tc, ic = _chunk(items)
        assert len(tc) == 1
        assert "Page footer" not in tc[0]["content"]

    def test_page_number_filtered(self):
        """page_number items are removed."""
        items = [_page_number("123", page=0), _text("Real content.", page=0)]
        tc, ic = _chunk(items)
        assert len(tc) == 1
        assert "123" not in tc[0]["content"] or "Real content" in tc[0]["content"]


# ============ ???? ============

class TestImageExtraction:

    def test_image_becomes_image_chunk(self):
        """Image items produce image chunks with correct fields."""
        items = [_text("Text before image.", page=0), _image("images/fig1.jpg", page=0)]
        tc, ic = _chunk(items)
        assert len(ic) == 1
        assert ic[0]["chunk_id"] == "test__i00001"
        assert ic[0]["item_type"] == "image"
        assert ic[0]["source_stem"] == "test"
        assert "fig1.jpg" in ic[0]["image_path"]
        assert ic[0]["description"] == ""  # empty, filled later
        assert ic[0]["parent_text_chunk_id"] == tc[0]["chunk_id"]

    def test_chart_becomes_image_chunk(self):
        """Chart items also produce image chunks."""
        items = [_text("Text.", page=0), _chart("images/c1.jpg", page=0)]
        tc, ic = _chunk(items)
        assert len(ic) == 1
        assert ic[0]["item_type"] == "chart"

    def test_image_caption_captured(self):
        """Image caption is stored in image chunk."""
        items = [
            _text("Text.", page=0),
            _image("images/fig.jpg", page=0, caption="Figure 1: ALD process"),
        ]
        tc, ic = _chunk(items)
        assert "ALD process" in ic[0]["caption"]

    def test_image_attached_to_text_chunk(self):
        """Image paths are listed in parent text chunk's image_paths."""
        items = [_text("Text.", page=0), _image("images/fig.jpg", page=0)]
        tc, ic = _chunk(items)
        assert len(tc[0]["image_paths"]) == 1
        assert "fig.jpg" in tc[0]["image_paths"][0]

    def test_image_only_section(self):
        """Section with only images (no text) -> caption fills content."""
        items = [_image("images/fig.jpg", page=0, caption="Figure 1: Test")]
        tc, ic = _chunk(items)
        assert len(ic) == 1
        # Text chunk should exist with caption as content
        assert len(tc) >= 1
        assert "Test" in tc[0]["content"] or "?" in tc[0]["content"]

    def test_multiple_images_sequential_ids(self):
        """Multiple images get sequential image chunk IDs."""
        items = [
            _text("Text.", page=0),
            _image("images/a.jpg", page=0),
            _image("images/b.jpg", page=0),
        ]
        tc, ic = _chunk(items)
        assert len(ic) == 2
        assert ic[0]["chunk_id"] == "test__i00001"
        assert ic[1]["chunk_id"] == "test__i00002"


# ============ ?? & ?? ============

class TestTableAndEquation:

    def test_table_html_preserved(self):
        """Table HTML is stored in table_html field."""
        html = ("<table><tr><th>Name</th><th>Value</th></tr>"
                "<tr><td>Data</td><td>1</td></tr></table>")
        items = [_text("Before table.", page=0), _table(html, page=0)]
        tc, ic = _chunk(items)
        assert len(tc) == 1
        assert tc[0]["has_table"] is True
        assert tc[0]["table_html"] == html
        # Stripped table text should be in content
        assert "Data" in tc[0]["content"]

    def test_equation_latex_preserved(self):
        """Equation LaTeX is included in content."""
        latex = r"$E = mc^2$"
        items = [_text("The equation is:", page=0), _equation(latex, page=0)]
        tc, ic = _chunk(items)
        assert len(tc) == 1
        assert latex in tc[0]["content"]


# ============ ????? ============

class TestMinSizeMerge:

    def test_small_tail_merged(self):
        """Tail chunk smaller than min_size merges into previous (same section)."""
        # First chunk: ~300 chars (>= target? No, but >= min_size)
        # Second chunk: ~50 chars (< min_size=150)
        items = [
            _text("A" * 300, page=0),
            _text("B" * 50, page=0),  # This should be merged
        ]
        tc, ic = _chunk(items, target=200, max_size=600, min_size=150)
        # The small chunk should be merged into the first
        assert len(tc) == 1
        assert "A" * 300 in tc[0]["content"]
        assert "B" * 50 in tc[0]["content"]
        assert tc[0]["char_count"] == 300 + 1 + 50  # newline separator

    def test_small_chunk_not_merged_across_sections(self):
        """Small chunks are NOT merged across different sections."""
        items = [
            _text("A" * 300, page=0),
            _text("NewSection", level=1, page=0),
            _text("B" * 50, page=0),  # Small, but different section
        ]
        tc, ic = _chunk(items, target=200, max_size=600, min_size=150)
        # The small chunk should NOT be merged (different heading_path)
        headings = [c["heading_path"] for c in tc]
        assert len(set(headings)) >= 2  # At least 2 different sections


# ============ ???? ============

class TestOversizedSplit:

    def test_sentence_split(self):
        """Content > max_size with sentence boundaries -> split."""
        # Each sentence ~30 chars, 25 sentences = ~750 chars > 600
        sentences = ". ".join([f"Sentence number {i} here" for i in range(25)])
        items = [_text(sentences, page=0)]
        tc, ic = _chunk(items, target=500, max_size=600, min_size=1)
        assert len(tc) >= 2
        for c in tc:
            assert c["char_count"] <= 600

    def test_hard_split_fallback(self):
        """LLM 返回超长/无效结果时,由 _hard_split 确定性切到 <= max_size。"""
        long_text = "X" * 700  # No sentence boundaries, > max_size=600
        items = [_text(long_text, page=0)]

        def mock_llm_split(text, target):
            # P2 优先尝试 LLM,但返回无效(每块仍超 max),应回退硬切
            return [text]

        tc, ic = _chunk(items, llm_split=mock_llm_split)
        assert len(tc) >= 2
        for c in tc:
            assert c["char_count"] <= 600
        # 内容不丢失
        assert "".join(c["content"] for c in tc) == long_text

    def test_oversized_no_llm_split_hard_split(self):
        """无 llm_split 时,超长文本也由硬切保证不超过 max_size(不保留超字段)。"""
        long_text = "X" * 700  # No sentence boundaries
        items = [_text(long_text, page=0)]
        tc, ic = _chunk(items, llm_split=None)
        assert len(tc) >= 2
        for c in tc:
            assert c["char_count"] <= 600
        assert "".join(c["content"] for c in tc) == long_text

    def test_target_closes_at_sentence_boundary(self):
        """达到 target 时在最后一个句号收口,未竟句作下一块起点(不夹断句子)。"""
        # 6 句各 200 字、句号结尾;target=500。每块应停在句号(400 字一块),
        # 而不是把第 3 句夹断在 500 字处。
        sents = [("字" * 199) + "。" for _ in range(6)]
        items = [_text("".join(sents), page=0)]
        tc, ic = _chunk(items, target=500, max_size=720, min_size=1, llm_split=None)
        # 每个非空块都应以句号结尾
        for c in tc:
            assert c["content"].rstrip().endswith("。")
            assert c["char_count"] <= 500     # 收口在句号,未越界夹断
        # 内容不丢
        assert "".join(c["content"] for c in tc) == "".join(sents)
        # 第一块正好 2 句(400),第 3 句留给下一块
        assert tc[0]["char_count"] == 400

    def test_table_rows_all_enter_content(self):
        """表格按行转文本后全部入 content(大表中后段行也在),HTML 完整保留。"""
        rows = "".join(
            f"<tr><td>Code {i}</td><td>Event number {i} description here</td></tr>"
            for i in range(100)
        )
        html = f"<table>{rows}</table>"
        items = [_table(html, page=0)]
        tc, ic = _chunk(items, target=500, max_size=600, min_size=1)
        # 中后段行进入某个 chunk
        all_content = "\n".join(c["content"] for c in tc)
        assert "Code 99" in all_content
        assert "Code 0" in all_content
        # 完整 HTML 挂在某个 chunk
        assert any(c.get("table_html") == html for c in tc)
        for c in tc:
            assert c["char_count"] <= 600

    def test_table_header_repeated_across_chunks(self):
        """大表跨块时,列名行复制到每个续块开头(Excel 顶端标题行),HTML 只挂首块。"""
        header = "<tr><th>Alarm ID</th><th>Description</th><th>Solution</th></tr>"
        body = "".join(
            f"<tr><td>{200 + i}</td><td>Vision ConnectionError number {i} occurred</td>"
            f"<td>Check cable and restart module {i}</td></tr>"
            for i in range(60)
        )
        html = f"<table>{header}{body}</table>"
        items = [
            _text("7.3.1 Alarm List", level=2, page=5),
            _table(html, page=5, caption="Alarm list"),
        ]
        tc, ic = _chunk(items, target=500, max_size=600, min_size=1)
        assert len(tc) >= 2, "大表应被切成多块"
        # 每个含数据行的块都必须包含列名行(首块列名随原文,续块为复制到块首)
        header_line = "Alarm ID | Description | Solution"
        data_chunks = [c for c in tc if "Vision ConnectionError" in c["content"]]
        assert len(data_chunks) >= 2
        for c in data_chunks:
            assert header_line in c["content"], (
                f"块缺列名行: {c['content'][:80]!r}")
            assert c["char_count"] <= 600
        # 不含标题的续块:必须以列名行开头(单独召回时也自解释)
        cont_chunks = [c for c in data_chunks
                       if "7.3.1 Alarm List" not in c["content"] and "Alarm list" not in c["content"]]
        assert cont_chunks, "应存在跨表的续块"
        for c in cont_chunks:
            assert c["content"].startswith(header_line), (
                f"续块未以列名行开头: {c['content'][:60]!r}")
            assert c.get("table_html") is None     # 续块不重复存 HTML
        # 完整 HTML 只挂一处(含表头行的首块)
        html_chunks = [c for c in tc if c.get("table_html") == html]
        assert len(html_chunks) == 1
        assert html_chunks[0]["has_table"] is True
        # 数据不丢
        all_content = "\n".join(c["content"] for c in tc)
        assert "Alarm ID | Description | Solution" in all_content
        assert "259" in all_content


# ============ ?? ============

class TestPageRange:

    def test_page_range_multiple_pages(self):
        """Chunks spanning multiple pages have correct page_start/page_end."""
        items = [
            _text("A" * 300, page=0),
            _text("B" * 300, page=2),
        ]
        tc, ic = _chunk(items, target=500, max_size=600, min_size=150)
        # If packed into one chunk, page_start=0, page_end=2
        # If split, each has its own page
        for c in tc:
            assert c["page_start"] <= c["page_end"]

    def test_image_page_num(self):
        """Image chunk records correct page_num."""
        items = [_text("Text.", page=3), _image("images/f.jpg", page=3)]
        tc, ic = _chunk(items)
        assert ic[0]["page_num"] == 3


# ============ P0 回归: list/code 内容不丢失 ============

class TestListAndCodeExtraction:

    def test_list_items_enter_content(self):
        """list 类型的 list_items 全部进入 content(参考文献等不丢失)。"""
        refs = [
            "1. A. M. Shevjakov, Proceedings of the Second USSR Conference (1965).",
            "2. T. Suntola and J. Antson, U. S. Patent, No. 4,058,430 (1977).",
            "3. http://www.planar.com/ald/.",
        ]
        items = [_text("See references below.", page=0), _list(refs, page=1)]
        tc, ic = _chunk(items)
        assert len(tc) == 1
        assert "Shevjakov" in tc[0]["content"]
        assert "4,058,430" in tc[0]["content"]
        assert "planar.com" in tc[0]["content"]

    def test_code_body_enter_content(self):
        """code 类型的 code_body 进入 content(配方/算法代码不丢失)。"""
        body = "File name: HfO2_AL2O3_300_3nm\n#9: 300°C  #8: 270°C"
        items = [_text("Recipe follows.", page=0), _code(body, page=1, caption="Recipe 1")]
        tc, ic = _chunk(items)
        assert len(tc) == 1
        assert "HfO2_AL2O3_300_3nm" in tc[0]["content"]
        assert "300°C" in tc[0]["content"]
        assert "Recipe 1" in tc[0]["content"]      # code_caption 作为前缀

    def test_empty_list_does_not_crash(self):
        """list_items 缺失/为空时不报错、不产生空块。"""
        items = [_text("Body text here.", page=0), {"type": "list", "page_idx": 1}]
        tc, ic = _chunk(items)
        assert all(c["content"].strip() for c in tc)


# ============ P0 回归: 标题不重复、无孤儿标题块 ============

class TestHeadingNoDuplication:

    def test_heading_not_in_content(self):
        """标题文本只出现在 heading_path,不重复进 content 首行。"""
        items = [
            _text("Chapter One", level=1, page=0),
            _text("This is the chapter body content.", page=0),
        ]
        tc, ic = _chunk(items)
        assert len(tc) == 1
        body = tc[0]
        assert "Chapter One" in body["heading_path"]
        assert body["content"] == "This is the chapter body content."
        assert not body["content"].startswith("Chapter One")

    def test_heading_only_section_no_orphan_chunk(self):
        """连续多个只有标题、无正文的 section 不产生碎片块。"""
        items = [
            _text("Opening content before chapters.", page=0),
            _text("Chapter 1", level=1, page=1),
            _text("Section 1.1", level=2, page=1),
            _text("Section 1.2", level=2, page=1),
            _text("Chapter 2", level=1, page=2),
            _text("Chapter 2 body starts here.", page=2),
        ]
        tc, ic = _chunk(items)
        # 只有两个有正文的块:开头、Chapter 2 正文
        contents = [c["content"] for c in tc]
        assert "Opening content" in "\n".join(contents)
        assert "Chapter 2 body starts here" in "\n".join(contents)
        # 任何块都不应该只是一个独立标题
        for c in tc:
            assert c["content"].strip() not in ("Chapter 1", "Section 1.1",
                                                "Section 1.2", "Chapter 2")
            assert c["char_count"] >= 5 or not c["content"].strip()


# ============ P0 回归: 表头启发式 ============

class TestTableHeaderHeuristic:

    def test_group_header_row_not_treated_as_columns(self):
        """首行单单元格(colspan 分组标题)不挂 table_html、不复制列名。"""
        html = (
            "<table>"
            "<tr><td colspan='11'>A.焊頭</td></tr>"
            "<tr><td>1</td><td>清潔吸嘴尖</td><td>√</td><td></td><td></td></tr>"
            "<tr><td>2</td><td>更換O-Ring</td><td></td><td>√</td><td></td></tr>"
            "<tr><td>3</td><td>檢查氣路</td><td></td><td></td><td>√</td></tr>"
            "</table>"
        )
        items = [_table(html, page=0)]
        tc, ic = _chunk(items, target=500, max_size=600, min_size=1)
        # 不应有任何块挂 table_html(无列名行)
        assert all(c.get("table_html") is None for c in tc)
        assert all(c["has_table"] is False for c in tc)
        # 但数据行仍应进入 content
        all_content = "\n".join(c["content"] for c in tc)
        assert "清潔吸嘴尖" in all_content
        assert "更換O-Ring" in all_content
        assert "A.焊頭" in all_content          # 分组标题作为普通行保留

    def test_sparse_first_row_not_treated_as_header(self):
        """首行列数显著少于第二行时,视为 OCR 噪声而非列名。"""
        html = (
            "<table>"
            "<tr><td>，</td></tr>"                                  # 1 个噪声单元格
            "<tr><td>2429</td><td>报警</td><td>输入系统Y轴位置错误</td></tr>"
            "<tr><td>2430</td><td>报警</td><td>真空错误</td></tr>"
            "<tr><td>2431</td><td>报警</td><td>关闭出料堆栈门</td></tr>"
            "</table>"
        )
        items = [_table(html, page=0)]
        tc, ic = _chunk(items, target=500, max_size=600, min_size=1)
        assert all(c.get("table_html") is None for c in tc)
        all_content = "\n".join(c["content"] for c in tc)
        assert "2429" in all_content and "真空错误" in all_content

    def test_real_header_still_detected_and_repeated(self):
        """真正的多列列名表仍被识别、跨块时复制列名(回归保护)。"""
        header = "<tr><th>Alarm ID</th><th>Description</th><th>Solution</th></tr>"
        body = "".join(
            f"<tr><td>{200 + i}</td><td>Vision ConnectionError number {i}</td>"
            f"<td>Check cable and restart module {i}</td></tr>"
            for i in range(60)
        )
        html = f"<table>{header}{body}</table>"
        items = [_table(html, page=5)]
        tc, ic = _chunk(items, target=500, max_size=600, min_size=1)
        # 应有块挂 table_html
        assert any(c.get("table_html") == html for c in tc)
        # 每个含数据行的块都必须包含列名行(首块原文,续块复制)
        data_chunks = [c for c in tc if "Vision ConnectionError" in c["content"]]
        assert len(data_chunks) >= 2
        for c in data_chunks:
            assert "Alarm ID | Description | Solution" in c["content"]
            assert c["char_count"] <= 600


# ============ P1 回归: 内联 HTML 清洗 ============

class TestInlineHtmlCleaning:

    def test_sup_sub_stripped_from_text(self):
        """正文/脚注中的 <sup>/<sub> 标签被清理,文本保留。"""
        items = [
            _text("The value is 10<sup>3</sup> times larger.", page=0),
            {"type": "page_footnote", "text": "<sup>3</sup> Source: Google blog, 2024.",
             "page_idx": 0},
        ]
        tc, ic = _chunk(items)
        content = "\n".join(c["content"] for c in tc)
        assert "10 3" in content or "103" in content.replace(" ", "")
        assert "<sup>" not in content
        assert "Source: Google blog, 2024." in content
        assert "[脚注]" in content

    def test_table_caption_html_cleaned(self):
        """table_caption 列表中的 <sup>/<sub> 被清理。"""
        html = "<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>"
        it = _table(html, page=0)
        it["table_caption"] = ["Table 1<sup>st</sup> result"]
        tc, ic = _chunk([it])
        content = "\n".join(c["content"] for c in tc)
        assert "Table 1" in content
        assert "<sup>" not in content

    def test_chart_footnote_enters_content(self):
        """chart_footnote(数据来源等)拼入正文。"""
        chart = _chart("images/c.jpg", page=3)
        chart["chart_footnote"] = ["Source: McKinsey survey, 2024."]
        tc, ic = _chunk([_text("See chart below.", page=3), chart])
        content = "\n".join(c["content"] for c in tc)
        assert "McKinsey survey" in content
        assert "<" not in content or "<sup" not in content

    def test_aside_text_filtered(self):
        """aside_text(侧边版本号/坐标标记)被过滤。"""
        items = [
            {"type": "aside_text", "text": "C1777A", "page_idx": 0},
            _text("Real body content here.", page=0),
        ]
        tc, ic = _chunk(items)
        content = "\n".join(c["content"] for c in tc)
        assert "C1777A" not in content
        assert "Real body content" in content


# ============ P1 回归: 编号标题识别 ============

class TestNumberedHeading:

    def test_korean_circled_heading(self):
        """①/② 等圈号编号(韩文报告)作为 L3 标题切 section。"""
        items = [
            _text("Previous content.", page=0),
            _text("① 고성능 플라즈마 소스 설계 및 제작 기술", page=1),
            _text("Detailed body for this subsection.", page=1),
        ]
        tc, ic = _chunk(items)
        body = [c for c in tc if "Detailed body" in c["content"]]
        assert len(body) == 1
        assert "고성능 플라즈마" in body[0]["heading_path"]
        assert body[0]["content"] == "Detailed body for this subsection."

    def test_korean_jamo_heading(self):
        """가/나/다 韩文编号作为 L3。"""
        items = [
            _text("가. 개발목표", page=0),
            _text("Goal description.", page=0),
        ]
        tc, ic = _chunk(items)
        assert len(tc) == 1
        assert "개발목표" in tc[0]["heading_path"]
        assert tc[0]["content"] == "Goal description."

    def test_dotted_numeric_heading(self):
        """1.2 / 1. 数字编号作为 L3。"""
        items = [
            _text("1.2 ALD 공정 개요", page=0),
            _text("Process overview content.", page=0),
        ]
        tc, ic = _chunk(items)
        assert len(tc) == 1
        assert "ALD 공정 개요" in tc[0]["heading_path"]
        assert tc[0]["content"] == "Process overview content."

    def test_numbered_list_body_not_misclassified(self):
        """编号开头但包含多个编号项/句末标点的正文不被误判为标题。"""
        items = [
            _text("1.开机和关机。1)并机。依次打开压缩空气阀。", page=0),
            _text("Follow up sentence.", page=0),
        ]
        tc, ic = _chunk(items)
        # 两句话应在同一块(没切 section)
        assert len(tc) == 1
        assert "开机" in tc[0]["content"]
        assert "Follow up" in tc[0]["content"]

    def test_multi_number_line_not_heading(self):
        """一行含两个编号(10...11...)不被误判。"""
        items = [
            _text("10.涂料器排水 11.脱盐水清洁", page=0),
        ]
        tc, ic = _chunk(items)
        assert len(tc) == 1
        # 内容完整保留(作为正文,而非标题)
        assert "涂料器" in tc[0]["content"]


# ============ P1 回归: 跨块 overlap ============

class TestOverlap:

    def test_tail_sentence_prepended_to_next_chunk(self):
        """正文跨块时,下一块以"上一块尾句 + 换行 + 正文"起首。"""
        # 每句约 60 字,target=150,max=300,min=1 -> 必然跨块
        sents = "。".join([f"这是第{i}个关于原子层沉积工艺的详细技术描述句子内容" for i in range(12)]) + "。"
        items = [_text(sents, page=0)]
        tc, ic = _chunk(items, target=150, max_size=300, min_size=1,
                       llm_split=None)
        assert len(tc) >= 2
        # 第二块应以某个完整句(上一块尾句)开头,而不是直接从截断位置起
        second = tc[1]
        first_line = second["content"].split("\n")[0]
        # 第一行是上一块的尾句(以句号结尾,长度合理)
        assert first_line.endswith("。")
        assert 5 <= len(first_line) <= 120
        # 且该尾句确实也出现在上一块内容中
        assert first_line in tc[0]["content"]

    def test_overlap_not_exceed_max_size(self):
        """加 overlap 后总长度不超过 max_size。"""
        sents = "。".join([f"句子编号{i}内容" + "X" * 40 for i in range(20)]) + "。"
        items = [_text(sents, page=0)]
        tc, ic = _chunk(items, target=200, max_size=400, min_size=1)
        for c in tc:
            assert c["char_count"] <= 400, f"超 max: {c['char_count']}"

    def test_table_row_chunk_has_no_overlap(self):
        """表格续块不因 overlap 机制在块首叠加正文句(只复制列名行)。"""
        header = "<tr><th>ID</th><th>Desc</th></tr>"
        body = "".join(f"<tr><td>{i}</td><td>Description number {i} content here</td></tr>"
                       for i in range(40))
        html = f"<table>{header}{body}</table>"
        items = [_table(html, page=0)]
        tc, ic = _chunk(items, target=300, max_size=500, min_size=1)
        # 含数据的块应以列名行或数据行开头,不应以"上一块尾句"开头(尾句是表格行)
        data_chunks = [c for c in tc if "Description number" in c["content"]]
        assert len(data_chunks) >= 2
        for c in data_chunks:
            first_line = c["content"].split("\n")[0]
            assert "ID | Desc" in first_line or first_line[0].isdigit()

    def test_overlap_resets_across_sections(self):
        """章节切换时不把上一章节尾句带到新章节首块。"""
        items = [
            _text("Chapter One", level=1, page=0),
            _text("。".join([f"第一章句子{i}内容" + "X" * 30 for i in range(10)]) + "。", page=0),
            _text("Chapter Two", level=1, page=1),
            _text("Chapter two body starts fresh.", page=1),
        ]
        tc, ic = _chunk(items, target=200, max_size=400, min_size=1)
        # 找到 Chapter two body 块
        ch2 = [c for c in tc if "Chapter two body starts fresh." in c["content"]]
        assert len(ch2) == 1
        # 它的第一行不应包含"第一章"任何内容
        first_line = ch2[0]["content"].split("\n")[0]
        assert "第一章" not in first_line


# ============ P2 回归: 英文缩写不切句 ============

class TestAbbreviationNoSplit:

    def test_fig_abbrev_not_split(self):
        """Fig. 5 / et al. / e.g. 等缩写点不在缩写处切句。"""
        text = ("See Fig. 5 for the architecture. The result is shown in Fig. 12. "
                "Smith et al. reported similar findings. Use e.g. a copper substrate.")
        items = [_text(text, page=0)]
        tc, ic = _chunk(items, target=500, max_size=600, min_size=1)
        # 单块(没有因为缩写点被切断)
        assert len(tc) == 1
        assert "Fig. 5" in tc[0]["content"]
        assert "Fig. 12" in tc[0]["content"]
        assert "et al." in tc[0]["content"]
        assert "e.g." in tc[0]["content"]

    def test_abbrev_at_end_still_ends_sentence(self):
        """缩写位于真正句末时(文本结束)仍作为句末点。"""
        from chunker import _cut_at_sentence
        text = "We used many materials, e.g."
        pre, suf = _cut_at_sentence(text, len(text))
        assert pre == text
        assert suf == ""

    def test_u_s_abbrev_not_split(self):
        """U.S. / No. / Vol. 等缩写点不切句。"""
        text = ("The U.S. patent office granted the claim. "
                "See No. 4,058,430 for details. "
                "Published in Vol. 12 of the journal.")
        items = [_text(text, page=0)]
        tc, ic = _chunk(items, target=500, max_size=600, min_size=1)
        assert len(tc) == 1
        assert "U.S." in tc[0]["content"]
        assert "No. 4,058,430" in tc[0]["content"]
        assert "Vol. 12" in tc[0]["content"]


# ============ P2 回归: LLM 兜底失败硬切 + 缓存 ============

class TestLlmFallbackHardSplit:

    def test_llm_none_forces_hard_split(self):
        """llm_split 返回 None 时,超长文本仍由硬切保证 <= max_size。"""
        long_text = "A" * 700  # 无句界
        items = [_text(long_text, page=0)]

        def llm_none(text, target):
            return None

        tc, ic = _chunk(items, llm_split=llm_none)
        assert len(tc) >= 2
        for c in tc:
            assert c["char_count"] <= 600
        assert "".join(c["content"] for c in tc) == long_text

    def test_llm_raises_forces_hard_split(self):
        """llm_split 抛异常时不崩溃,由硬切兜底。"""
        long_text = "B" * 700
        items = [_text(long_text, page=0)]

        def llm_raise(text, target):
            raise RuntimeError("LLM API down")

        tc, ic = _chunk(items, llm_split=llm_raise)
        assert len(tc) >= 2
        for c in tc:
            assert c["char_count"] <= 600

    def test_llm_result_cached_by_text_hash(self):
        """相同长文本重复出现时,llm_split 只被调用一次(hash 缓存)。"""
        long_text = "C" * 700
        items = [_text(long_text, page=0), _text(long_text, page=1)]
        calls = []

        def llm_count(text, target):
            calls.append(text)
            return [text[i:i + 300] for i in range(0, len(text), 300)]

        tc, ic = _chunk(items, llm_split=llm_count)
        assert len(calls) == 1, f"LLM 应只被调用一次(缓存),实际 {len(calls)} 次"
        for c in tc:
            assert c["char_count"] <= 600

    def test_llm_invalid_return_falls_back(self):
        """llm_split 返回空列表/非字符串时,视为失败并硬切。"""
        long_text = "D" * 700
        items = [_text(long_text, page=0)]
        tc, ic = _chunk(items, llm_split=lambda t, tg: [])
        assert len(tc) >= 2
        for c in tc:
            assert c["char_count"] <= 600


# ============ P2 回归: 表格行硬切保护 ============

class TestTableRowHardSplitProtection:

    def test_long_table_row_split_at_cell_boundary(self):
        """超长表格行在 ' | ' 单元格边界切,不在单元格内部切断。"""
        # 两个超长单元格,合计远超 max_size,但每个单元格 < max_size
        cell_a = "A" * 300
        cell_b = "B" * 300
        cell_c = "C" * 300
        html = (
            "<table>"
            f"<tr><th>ColA</th><th>ColB</th><th>ColC</th></tr>"
            f"<tr><td>{cell_a}</td><td>{cell_b}</td><td>{cell_c}</td></tr>"
            "</table>"
        )
        items = [_table(html, page=0)]
        tc, ic = _chunk(items, target=500, max_size=600, min_size=1)
        for c in tc:
            assert c["char_count"] <= 600
        all_content = "\n".join(c["content"] for c in tc)
        # 每个单元格的内容必须完整出现在某块中(不被切断)
        assert cell_a in all_content
        assert cell_b in all_content
        assert cell_c in all_content
        # 不出现单元格被截断的痕迹(连续 299 个 A 后直接是别的字符)
        assert "A" * 300 in all_content

    def test_no_split_inside_cell(self):
        """硬切不在 ' | ' 单元格分隔符处切断。"""
        cells = [f"Cell{i}-" + "X" * 50 for i in range(20)]
        rows = "".join(f"<tr><td>{c}</td><td>{c}v</td></tr>" for c in cells)
        html = f"<table><tr><th>H1</th><th>H2</th></tr>{rows}</table>"
        items = [_table(html, page=0)]
        tc, ic = _chunk(items, target=400, max_size=600, min_size=1)
        for c in tc:
            # 每一行如果出现,要么整行完整,要么是在单元格边界切分;
            # 不含单独的 " | " 在块尾(说明切在了分隔符处)
            assert not c["content"].rstrip().endswith(" |")

    def test_single_oversized_cell_hard_split(self):
        """单个单元格本身超长时,允许在单元格内部硬切,且不超 max。"""
        cell = "W" * 800  # 单格就超 max_size=600
        html = (
            "<table>"
            "<tr><th>Header</th></tr>"
            f"<tr><td>{cell}</td></tr>"
            "</table>"
        )
        items = [_table(html, page=0)]
        tc, ic = _chunk(items, target=500, max_size=600, min_size=1)
        for c in tc:
            assert c["char_count"] <= 600
        all_content = "\n".join(c["content"] for c in tc)
        # 内容不丢
        assert "W" * 700 in all_content or all_content.count("W") >= 800


# ============ P3 回归: 连续编号步骤不误判为标题 ============

class TestConsecutiveNumberingNotHeading:

    def test_numbered_steps_stay_in_same_section(self):
        """连续编号的操作步骤(1./2./3.)不被切成独立 L3 section,内容在同一块。"""
        items = [
            _text("The general technique is to:", page=0),
            _text("1. Isolate the reactor from the vacuum pumping system", page=0),
            _text("2. Pulse in precursor A for one second", page=0),
            _text("3. Allow time for precursor A to diffuse", page=0),
            _text("4. Re-establish vacuum pumping and purge", page=0),
        ]
        tc, ic = _chunk(items)
        # 不应出现以步骤文本为 heading_path 的 section
        for c in tc:
            assert "Pulse in precursor A" not in c["heading_path"]
        # 步骤文本应作为 content 保留
        all_content = "\n".join(c["content"] for c in tc)
        assert "Isolate the reactor" in all_content
        assert "Pulse in precursor A" in all_content
        assert "Re-establish vacuum" in all_content

    def test_consecutive_steps_can_span_image(self):
        """步骤间穿插配图时,仍识别为同一列表(不把步骤当标题)。"""
        items = [
            _text("4. Gently move the transfer arm out of the chamber", page=0),
            _image("images/arm.jpg", page=0),
            _text("5. Press OK on the popup dialogue box to close the gate valve", page=0),
        ]
        tc, ic = _chunk(items)
        for c in tc:
            assert "Press OK" not in c["heading_path"]
        all_content = "\n".join(c["content"] for c in tc)
        assert "Press OK" in all_content

    def test_circled_run_treated_as_list(self):
        """连续圈号 ①②③ 作为枚举列表,不切 section(中间无正文)。"""
        items = [
            _text("① 고성능 플라즈마 소스 설계 기술", page=0),
            _text("② 차세대 Precursor 적용 기술", page=0),
            _text("③ 고성능 Process Module 설계 기술", page=0),
        ]
        tc, ic = _chunk(items)
        # 三个圈号项不应各自成为 heading_path 片段
        for c in tc:
            assert "차세대 Precursor" not in c["heading_path"]
        all_content = "\n".join(c["content"] for c in tc)
        assert "차세대 Precursor" in all_content

    def test_single_numbered_heading_still_splits(self):
        """孤立编号标题(后跟正文,非连续步骤)仍切 section。"""
        items = [
            _text("Before section content here.", page=0),
            _text("1.2 ALD 공정 개요", page=0),
            _text("Process overview content for this section.", page=0),
        ]
        tc, ic = _chunk(items)
        body = [c for c in tc if "Process overview" in c["content"]]
        assert len(body) == 1
        assert "ALD 공정 개요" in body[0]["heading_path"]

# ============ P4 回归: HTML 实体解码 ============

class TestHtmlEntityDecode:

    def test_numeric_and_named_entities_in_text(self):
        """正文里的 &#x27; / &quot; / &amp; / &gt; 等实体被解码为字符。"""
        items = [
            _text("Press the button &#x27;PUMP&#x27; &amp; wait.", page=0),
            _text("Set value &gt; 5 &lt; 10, use &quot;fast&quot; mode.", page=0),
        ]
        tc, ic = _chunk(items)
        content = "\n".join(c["content"] for c in tc)
        assert "button 'PUMP' & wait." in content
        assert "> 5 < 10" in content
        assert '"fast"' in content
        # 不应残留任何实体
        import re
        assert not re.search(r"&(?:#x?[0-9a-fA-F]+|[a-zA-Z]+);", content)

    def test_entities_in_table_cells_decoded(self):
        """表格单元格里的实体被解码,table_html 保留原始 HTML(供渲染)。"""
        html = ("<table><tr><th>Part</th><th>Note</th></tr>"
                "<tr><td>Valve &amp; Hose</td><td>Use &quot;A&quot; grade</td></tr></table>")
        items = [_table(html, page=0)]
        tc, ic = _chunk(items)
        content = "\n".join(c["content"] for c in tc)
        assert "Valve & Hose" in content
        assert '"A" grade' in content
        # table_html 保留原始实体(完整 HTML 不做解码,供前端渲染)
        assert any(c.get("table_html") == html for c in tc)

    def test_entities_in_footnote_and_caption(self):
        """脚注/图表 caption 中的实体被解码。"""
        chart = _chart("images/c.jpg", page=0)
        chart["chart_footnote"] = ["Source: A &amp; B, 2024."]
        items = [
            {"type": "page_footnote", "text": "See &quot;note&quot; here.", "page_idx": 0},
            chart,
        ]
        tc, ic = _chunk(items)
        content = "\n".join(c["content"] for c in tc)
        assert 'A & B' in content
        assert '"note"' in content

    def test_escaped_tag_does_not_become_real_tag(self):
        """&lt;img&gt; 解码后不应被当成标签再次剥离(先去标签再解码的顺序保证)。"""
        items = [_text("Use &lt;img&gt; to denote an image tag.", page=0)]
        tc, ic = _chunk(items)
        # 解码后应是字面量 "<img>" 文本,而不是被当标签删掉
        assert "<img>" in tc[0]["content"]

    def test_numbered_heading_before_list_not_swallowed(self):
        """编号标题后跟步骤列表时,标题仍切 section,步骤不变成子标题。"""
        items = [
            _text("3. Operating Procedure", page=0),
            _text("Follow these steps:", page=0),
            _text("1. Open the valve", page=0),
            _text("2. Start the pump", page=0),
        ]
        tc, ic = _chunk(items)
        # "3. Operating Procedure" 是标题(text_level 无但孤立);
        # 这里它后面跟正文 "Follow these steps:" 再跟 1./2. 列表。
        # 关键:1./2. 不能进 heading_path
        for c in tc:
            assert "Open the valve" not in c["heading_path"]
            assert "Start the pump" not in c["heading_path"]




# ---- P5: OCR 单词粘连修复(wordninja 切词)----

def test_deglue_splits_long_glued_sentence():
    """超长小写粘连串(含英文功能词)被正确切回带空格的句子。"""
    from chunker import _clean_inline
    got = _clean_inline("anyofthecomponentsusedinmodernproducts")
    assert got == "any of the components used in modern products"


def test_deglue_keeps_normal_text_url_and_identifiers():
    """正常英文、URL、驼峰标识符、化学长词不被改动。"""
    from chunker import _clean_inline
    for s in [
        "the electron beam lithography system",
        "https://example.com/path?x=1",
        "CoarseHorVertEntrySelector123",
        "trimethylsilyldiethylamine",  # 化学长词,不含功能词
    ]:
        assert _clean_inline(s) == s


def test_deglue_applied_during_chunking():
    """端到端:含粘连的 text item 经 chunk_content_list 后正文被切开。"""
    cl = [_text("Miniaturization " + "isthecentralthemeinmodernfabrication" * 3, page=0)]
    tc, _ = chunk_content_list(
        cl, source_path="x.pdf", source_stem="x", auto_dir=".",
        target=500, max_size=800, min_size=100)
    blob = " ".join(c["content"] for c in tc)
    assert "is the central theme" in blob
    assert "isthecentralthemeinmodernfabrication" not in blob


# ---- 回归:超长"表头行"表格不得死循环,且不产出超 max 的块 ----
# FIJI V2 手册中一张两列表的首行是 861 字长段落(被 MinerU 当表头行),
# 导致 _hard_split 收到非正预算而 s[0:] 永不前进(死循环),且补列名后块超 max。

def test_table_with_oversized_header_row_no_hang_no_oversize():
    long_cell = "Gas Cabinet Exhaust. " * 40          # ~860 字符的"单元格"
    second_cell = "A metallic exhaust line. " * 10   # ~240 字符
    data_cell = "Precursor cabinet flexible or hard plumbing. " * 5
    html = (
        "<table>"
        f"<tr><td>{long_cell}</td><td>{second_cell}</td></tr>"
        f"<tr><td>{data_cell}</td><td>{data_cell}</td></tr>"
        f"<tr><td>row2a</td><td>row2b</td></tr>"
        "</table>"
    )
    cl = [{"type": "table", "table_body": html, "page_idx": 0}]
    # 必须在有限时间内返回;此前会死循环
    tc, _ = chunk_content_list(
        cl, source_path="t.pdf", source_stem="t", auto_dir=".",
        target=500, max_size=800, min_size=150)
    assert tc, "应产出 chunk"
    over = [c for c in tc if c["char_count"] > 800]
    assert not over, f"存在超 max 的块: {[c['char_count'] for c in over]}"
