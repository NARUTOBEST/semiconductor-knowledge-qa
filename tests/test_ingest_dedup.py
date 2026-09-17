# -*- coding: utf-8 -*-
"""ingest MD5 去重:同内容不同名的副本只入库一次。"""
import os, sys, json, importlib
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "RAG"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "config"))

import ingest


@pytest.fixture
def md5map_path(tmp_path, monkeypatch):
    p = tmp_path / "_ingest_md5.json"
    monkeypatch.setattr(ingest, "MD5MAP", str(p))
    return p


def test_md5_file_detects_identical(tmp_path):
    a = tmp_path / "a.pdf"; b = tmp_path / "b.pdf"; c = tmp_path / "c.pdf"
    a.write_bytes(b"same content")
    b.write_bytes(b"same content")
    c.write_bytes(b"different")
    ma, mb, mc = ingest.md5_file(str(a)), ingest.md5_file(str(b)), ingest.md5_file(str(c))
    assert ma == mb and ma != mc


def test_md5_file_missing_returns_none(tmp_path):
    assert ingest.md5_file(str(tmp_path / "nope.pdf")) is None


def test_content_md5_prefers_source_pdf(tmp_path):
    # 源 PDF 与 content_list 都存在时用源 PDF 的 MD5(最稳)
    auto = tmp_path / "auto"; auto.mkdir()
    stem = "doc"
    src = tmp_path / "doc.pdf"; src.write_bytes(b"PDF BYTES")
    cl = auto / "doc_content_list.json"; cl.write_bytes(b"[]")
    m, src_kind = ingest.content_md5(str(auto), stem, str(src))
    assert m == ingest.md5_file(str(src))
    assert src_kind == "src"


def test_content_md5_falls_back_to_content_list(tmp_path):
    auto = tmp_path / "auto"; auto.mkdir()
    stem = "doc"
    cl = auto / "doc_content_list.json"; cl.write_bytes(b"[]")
    m, src_kind = ingest.content_md5(str(auto), stem, str(tmp_path / "missing.pdf"))
    assert m == ingest.md5_file(str(cl))
    assert src_kind == "cl"


def test_md5map_roundtrip(md5map_path):
    assert ingest.load_md5map() == {}
    ingest.save_md5map({"abc123": "docA"})
    assert ingest.load_md5map() == {"abc123": "docA"}


def test_duplicate_skipped_in_ingest_dir(tmp_path, monkeypatch, md5map_path):
    """两个不同 stem、同 MD5 的副本:第一个入库,第二个被跳过。"""
    # 构造两个 auto 目录,源 PDF 内容相同但文件名不同
    root = tmp_path / "clean"; src_root = tmp_path / "src"
    docs = []
    for stem in ("docA", "docB"):
        auto = root / stem / "auto"; auto.mkdir(parents=True)
        (auto / f"{stem}_content_list.json").write_text("[]", encoding="utf-8")
        # 同字节内容 => 同 MD5
        (src_root / stem).mkdir(parents=True)
        (src_root / stem / f"{stem}.pdf").write_bytes(b"IDENTICAL PDF")
        docs.append((stem, str(auto)))

    # 让 relpath 反推源 PDF 的逻辑成立:CLEAN_ROOT 指向 root
    monkeypatch.setattr(ingest.C, "CLEAN_ROOT", str(root))
    monkeypatch.setattr(ingest, "CKPT", str(tmp_path / "ckpt.json"))

    # mock process_pdf:记录被处理的 stem;client/encoders 不真正使用
    processed = []
    def fake_process(stem, auto, src, client, do_describe=True):
        processed.append(stem)
        return 1, 0
    monkeypatch.setattr(ingest, "process_pdf", fake_process)
    monkeypatch.setattr(ingest, "get_client", lambda: object())
    monkeypatch.setattr(ingest, "ensure_collections", lambda c: None)
    monkeypatch.setattr(ingest, "TE", object())  # 跳过 embed.get_text_encoder()
    monkeypatch.setattr(ingest, "IE", object())  # 跳过 embed.get_image_encoder()

    ingest.ingest_dir(str(root), str(src_root), do_describe=False)

    assert processed == ["docA"], "只有首个副本应被处理"
    m = ingest.load_md5map()
    assert len(m) == 1 and list(m.values())[0] == "docA"
    # checkpoint 把 docB 也记为已处理(避免重跑重复提示)
    done = json.load(open(ingest.CKPT, encoding="utf-8"))
    assert "docB" in done
