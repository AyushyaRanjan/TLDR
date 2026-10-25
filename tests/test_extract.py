"""Cleanup, text loading, and PDF extraction."""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from tests.conftest import make_settings
from tldr_everything.extract import (
    DocumentError,
    Page,
    cleanup_text,
    decode_bytes,
    load_file,
    load_pdf,
    load_text,
    rough_minutes,
    strip_running_elements,
)


def test_cleanup_dehyphenates_and_keeps_fenced_code() -> None:
    text = (
        "This is a sen-\n"
        "tence about tools.\n\n"
        "```python\n"
        "def add(a, b):\n"
        "    return a + b\n"
        "```\n\n"
        "A hy-\n"
        "phen stays out of code.\n"
        "```\n"
        "self-\n"
        "value\n"
        "```\n"
    )
    cleaned = cleanup_text(text)
    assert "sentence about tools." in cleaned
    assert "sen-" not in cleaned.split("```")[0]
    assert "def add(a, b):" in cleaned
    assert "    return a + b" in cleaned
    assert "hyphen stays" in cleaned
    assert "self-\nvalue" in cleaned


def test_cleanup_keeps_indented_code() -> None:
    text = "Intro line.\n\n    def add(a, b):\n        return a + b\n\nOutro line."
    cleaned = cleanup_text(text)
    assert "def add(a, b):" in cleaned
    assert "return a + b" in cleaned
    assert "Intro line." in cleaned
    assert "Outro line." in cleaned


def test_strip_repeated_headers_and_page_numbers() -> None:
    pages = [
        Page(1, "My Book\n\nAlpha content here about widgets and more words.\n\n1"),
        Page(2, "My Book\n\nBeta content here about gadgets and more words.\n\n2"),
        Page(3, "My Book\n\nGamma content here about doodads and more words.\n\n3"),
    ]
    cleaned = strip_running_elements(pages)
    joined = "\n".join(page.text for page in cleaned)
    assert "My Book" not in joined
    assert "Alpha content" in joined
    assert "Beta content" in joined
    assert "Gamma content" in joined
    for page in cleaned:
        assert not page.text.strip().endswith("1") or "Alpha" in page.text


def test_decode_bytes_handles_bom_cp1252_and_latin1() -> None:
    assert decode_bytes(b"\xef\xbb\xbfhello") == "hello"
    assert "é" in decode_bytes("Café notes about reading.".encode("cp1252"))
    decoded = decode_bytes(b"ok\x81tail")
    assert decoded.startswith("ok")
    assert decoded.endswith("tail")


def test_empty_text_and_too_large() -> None:
    settings = make_settings()
    with pytest.raises(DocumentError) as empty:
        load_text("   ", "paste", settings)
    assert empty.value.code == "empty"

    limited = make_settings(max_chars=40)
    with pytest.raises(DocumentError) as large:
        load_text("This document is definitely longer than forty characters.", "paste", limited)
    assert large.value.code == "too_large"
    assert "minutes" in str(large.value)


def test_rough_minutes_is_at_least_one() -> None:
    assert rough_minutes(10_000, 1500) >= 1


def test_load_text_and_markdown_file(tmp_path: Path) -> None:
    path = tmp_path / "note.md"
    path.write_text("Markdown notes about a small local model and a long article.", encoding="utf-8")
    document = load_file(path, make_settings())
    assert document.pages[0].number == 1
    assert "Markdown notes" in document.text
    assert "[p. 1]" in document.text


def test_unsupported_suffix(tmp_path: Path) -> None:
    path = tmp_path / "notes.docx"
    path.write_text("hello", encoding="utf-8")
    with pytest.raises(DocumentError) as exc:
        load_file(path, make_settings())
    assert exc.value.code == "unsupported"


def test_pdf_round_trip_and_empty_and_encrypted(tmp_path: Path) -> None:
    settings = make_settings()
    readable = tmp_path / "hello.pdf"
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "Hello from the tiny fixture about docker on a VPS today.")
    pdf.save(readable)
    pdf.close()
    document = load_pdf(readable.read_bytes(), "hello.pdf", settings)
    assert document.pages[0].number == 1
    assert "docker" in document.text.lower()

    blank = tmp_path / "blank.pdf"
    empty_pdf = pymupdf.open()
    empty_pdf.new_page()
    empty_pdf.save(blank)
    empty_pdf.close()
    with pytest.raises(DocumentError) as scanned:
        load_pdf(blank.read_bytes(), "blank.pdf", settings)
    assert scanned.value.code == "empty"
    assert "OCR" in str(scanned.value)

    secret = tmp_path / "secret.pdf"
    locked = pymupdf.open()
    locked.new_page()
    locked.save(
        secret,
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
        owner_pw="owner-password",
        user_pw="user-password",
    )
    locked.close()
    with pytest.raises(DocumentError) as encrypted:
        load_pdf(secret.read_bytes(), "secret.pdf", settings)
    assert encrypted.value.code == "encrypted"


def test_pdf_page_limit(tmp_path: Path) -> None:
    path = tmp_path / "three.pdf"
    pdf = pymupdf.open()
    for index in range(3):
        page = pdf.new_page()
        page.insert_text((72, 72), f"Page content number {index} is long enough to count as real text.")
    pdf.save(path)
    pdf.close()
    with pytest.raises(DocumentError) as exc:
        load_pdf(path.read_bytes(), "three.pdf", make_settings(max_pages=2))
    assert exc.value.code == "too_large"
    assert "minutes" in str(exc.value)


def test_non_utf8_text_file(tmp_path: Path) -> None:
    path = tmp_path / "cafe.txt"
    path.write_bytes("Café notes about a long article and a quiet afternoon.".encode("cp1252"))
    document = load_file(path, make_settings())
    assert "notes about a long article" in document.text
