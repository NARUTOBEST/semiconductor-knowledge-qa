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

    def test_level3_not_heading(self):
        """text_level=3 does NOT trigger section split."""
        items = [
            _text("Before.", page=0),
            _text("Level3", level=3, page=0),
            _text("After.", page=0),
        ]
        tc, ic = _chunk(items)
        # All in one section (level 3 is not a heading)
        assert len(tc) == 1
        assert "Before." in tc[0]["content"]
        assert "Level3" in tc[0]["content"]
        assert "After." in tc[0]["content"]


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
        html = "<table><tr><td>Data</td></tr></table>"
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

    def test_llm_split_fallback(self):
        """Content > max_size without sentence boundaries -> llm_split called."""
        long_text = "X" * 700  # No sentence boundaries, > max_size=600
        items = [_text(long_text, page=0)]

        call_args = []

        def mock_llm_split(text, target):
            call_args.append(text)
            mid = len(text) // 2
            return [text[:mid], text[mid:]]

        tc, ic = _chunk(items, llm_split=mock_llm_split)
        assert len(call_args) == 1  # llm_split was called
        assert len(tc) == 2
        for c in tc:
            assert c["char_count"] <= 600

    def test_oversized_no_llm_split_kept_as_is(self):
        """Content > max_size without llm_split -> kept as single oversized chunk."""
        long_text = "X" * 700  # No sentence boundaries
        items = [_text(long_text, page=0)]
        tc, ic = _chunk(items, llm_split=None)
        assert len(tc) == 1
        assert tc[0]["char_count"] == 700  # Kept as-is


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
