"""Split a cleaned document into overlapping chunks with page ranges."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from tldr_everything.config import Settings, estimate_tokens
from tldr_everything.extract import Document, Page

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """One model-sized slice of a document."""

    index: int
    text: str
    start_page: int
    end_page: int
    token_estimate: int


@dataclass
class _Unit:
    page: int
    text: str
    tokens: int


def chunk_document(document: Document, settings: Settings) -> list[Chunk]:
    """Pack page paragraphs into chunks of about ``CHUNK_TOKENS``.

    Overlap is best-effort and follows paragraph boundaries. A chunk always
    makes forward progress, so a paragraph larger than the overlap cannot
    stall the splitter. Code fences are kept intact unless a single fence is
    itself larger than the chunk, in which case it is split on line boundaries.
    """
    units = _units(document.pages, settings.chunk_tokens)
    if not units:
        return []

    chunks: list[Chunk] = []
    start = 0
    limit = max(50, settings.chunk_tokens)
    overlap_target = max(0, settings.chunk_overlap_tokens)
    while start < len(units):
        end = start
        tokens = 0
        while end < len(units) and tokens + units[end].tokens <= limit:
            tokens += units[end].tokens
            end += 1
        if end == start:
            end = start + 1
        group = units[start:end]
        body = "\n\n".join(f"[p. {unit.page}]\n{unit.text}" for unit in group)
        chunks.append(
            Chunk(
                index=len(chunks),
                text=body,
                start_page=group[0].page,
                end_page=group[-1].page,
                token_estimate=estimate_tokens(body),
            )
        )
        if end >= len(units):
            break
        new_start = _overlap_start(units, start, end, overlap_target)
        start = new_start

    logger.info("chunked document chunks=%s", len(chunks))
    return chunks


def _units(pages: list[Page], chunk_tokens: int) -> list[_Unit]:
    units: list[_Unit] = []
    for page in pages:
        for block in _blocks(page.text):
            for piece in _split_large(block, chunk_tokens):
                units.append(_Unit(page=page.number, text=piece, tokens=estimate_tokens(piece)))
    return units


def _blocks(text: str) -> list[str]:
    """Split on blank lines, but keep a fenced block together."""
    lines = text.split("\n")
    blocks: list[str] = []
    buffer: list[str] = []
    in_fence = False
    for line in lines:
        if line.strip().startswith("```"):
            in_fence = not in_fence
            buffer.append(line)
            continue
        if in_fence:
            buffer.append(line)
            continue
        if line.strip() == "" and buffer:
            blocks.append("\n".join(buffer).strip("\n"))
            buffer = []
            continue
        if line.strip() == "":
            continue
        buffer.append(line)
    if buffer:
        blocks.append("\n".join(buffer).strip("\n"))
    return [block for block in blocks if block.strip()]


def _split_large(text: str, chunk_tokens: int) -> list[str]:
    max_chars = max(200, chunk_tokens * 4)
    if len(text) <= max_chars:
        return [text]
    pieces: list[str] = []
    buffer = ""

    def push(segment: str) -> None:
        nonlocal buffer
        if len(segment) > max_chars:
            if buffer:
                pieces.append(buffer.rstrip())
                buffer = ""
            for start in range(0, len(segment), max_chars):
                part = segment[start : start + max_chars]
                if len(part) == max_chars:
                    pieces.append(part.rstrip())
                else:
                    buffer = part
            return
        if buffer and len(buffer) + len(segment) > max_chars:
            pieces.append(buffer.rstrip())
            buffer = segment
            return
        buffer += segment

    lines = text.splitlines(keepends=True) or [text]
    for line in lines:
        push(line)
    if buffer.strip():
        pieces.append(buffer.rstrip())
    return pieces or [text[:max_chars]]


def _overlap_start(units: list[_Unit], start: int, end: int, overlap_tokens: int) -> int:
    if overlap_tokens <= 0:
        return end
    new_start = end
    overlap = 0
    while new_start > start and overlap < overlap_tokens:
        new_start -= 1
        overlap += units[new_start].tokens
    if new_start <= start:
        return start + 1
    return new_start
