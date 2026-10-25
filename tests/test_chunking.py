"""Chunk size, overlap, and page ranges."""

from __future__ import annotations

from tests.conftest import make_settings

from tldr_everything.chunking import chunk_document
from tldr_everything.config import estimate_tokens
from tldr_everything.extract import Document, Page


def _document(pages: list[Page]) -> Document:
    text = "\n\n".join(page.text for page in pages)
    return Document(source_name="fixture", pages=pages, text=text, word_count=10)


def test_overlap_repeats_the_tail_and_moves_forward() -> None:
    def block(index: int) -> str:
        return f"MARK{index} " + " ".join(f"w{index}x{n}" for n in range(20))

    sample = block(0)
    unit_tokens = estimate_tokens(sample)
    settings = make_settings(chunk_tokens=unit_tokens * 2 + 5, chunk_overlap_tokens=unit_tokens)
    pages = [Page(1, "\n\n".join(block(index) for index in range(6)))]
    chunks = chunk_document(_document(pages), settings)
    assert len(chunks) >= 3
    assert "MARK0" in chunks[0].text
    assert "MARK1" in chunks[0].text
    assert "MARK1" in chunks[1].text
    assert "MARK0" not in chunks[1].text
    assert chunks[0].token_estimate > 0


def test_page_ranges_follow_the_source_pages() -> None:
    settings = make_settings(chunk_tokens=120, chunk_overlap_tokens=0)
    pages = [Page(1, "A" * 400), Page(2, "B" * 400)]
    chunks = chunk_document(_document(pages), settings)
    assert len(chunks) == 2
    assert (chunks[0].start_page, chunks[0].end_page) == (1, 1)
    assert (chunks[1].start_page, chunks[1].end_page) == (2, 2)
    assert "[p. 1]" in chunks[0].text
    assert "[p. 2]" in chunks[1].text


def test_large_block_is_split_on_line_boundaries() -> None:
    line = "docker build .\n"
    text = line * 40
    settings = make_settings(chunk_tokens=50, chunk_overlap_tokens=0)
    chunks = chunk_document(_document([Page(3, text)]), settings)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.start_page == 3
        assert chunk.end_page == 3
        assert "docker build ." in chunk.text
        assert chunk.token_estimate <= settings.chunk_tokens + 30


def test_fenced_block_stays_together() -> None:
    fence = "```bash\ndocker compose up -d\n```"
    prose = "A short intro to the command."
    settings = make_settings(chunk_tokens=200, chunk_overlap_tokens=0)
    chunks = chunk_document(_document([Page(1, prose + "\n\n" + fence)]), settings)
    holders = [chunk.text for chunk in chunks if "docker compose up -d" in chunk.text]
    assert len(holders) == 1
    assert "```bash" in holders[0]
    assert holders[0].count("```") >= 2
