"""Load PDF, text, and markdown documents and lightly clean them."""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from tldr_everything.config import ASSUMED_SECONDS_PER_CALL, Settings
from tldr_everything.postprocess import word_count

logger = logging.getLogger(__name__)

_TEXT_SUFFIXES = {".txt", ".md", ".markdown"}
_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_PAGE_NUM_RE = re.compile(r"^\s*(?:page\s*)?\d{1,4}\s*$", re.IGNORECASE)
_CODE_START_RE = re.compile(
    r"^(?:    |\t)|^\s*(?:def |class |import |from |return |sudo |docker |npm |pip3? |git |curl |\$ )"
)


class DocumentError(Exception):
    """A problem with the input the user can act on."""

    def __init__(self, message: str, code: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass
class Page:
    """One page of extracted text. Pasted text is a single page numbered 1."""

    number: int
    text: str


@dataclass
class Document:
    """A cleaned document plus the text the model and the grounding checks see."""

    source_name: str
    pages: list[Page]
    text: str
    word_count: int


def rough_minutes(chars: int, chunk_tokens: int, calls_per_chunk: int = 2) -> int:
    """Rough wall-clock estimate before any chunk has been timed.

    Assumes about 30 seconds per model call on a small CPU model. The live
    progress bar replaces this with observed latency once calls start.
    """
    tokens = max(1, chars // 4)
    chunks = max(1, math.ceil(tokens / max(chunk_tokens, 1)))
    calls = chunks * calls_per_chunk + 4
    return max(1, math.ceil(calls * ASSUMED_SECONDS_PER_CALL / 60))


def decode_bytes(data: bytes) -> str:
    """Decode text bytes, tolerating a BOM and common non-UTF-8 encodings.

    Latin-1 is the last step and accepts every byte, so this always returns text.
    """
    if data.startswith(b"\xef\xbb\xbf"):
        data = data[3:]
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        pass
    try:
        return data.decode("cp1252")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def cleanup_text(text: str) -> str:
    """De-hyphenate and collapse prose whitespace without touching code.

    Fenced blocks and runs of indented or command-like lines are held aside
    and restored unchanged.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    protected: list[str] = []

    def hold(block: str) -> str:
        protected.append(block.strip("\n"))
        return f"\n\n\x00CODE{len(protected) - 1}\x00\n\n"

    text = _FENCE_RE.sub(lambda match: hold(match.group(0)), text)
    text = _protect_code_lines(text, hold)
    text = _clean_prose(text)
    for index, block in enumerate(protected):
        text = text.replace(f"\x00CODE{index}\x00", block)
    return text.strip()


def strip_running_elements(pages: list[Page]) -> list[Page]:
    """Drop repeated headers, footers, and standalone page numbers.

    A line is treated as a running header only when it is short and shows up
    on at least 60% of pages (and on at least 3 pages). Body text that happens
    to repeat is left alone.
    """
    if len(pages) < 3:
        return [Page(page.number, _strip_page_numbers(page.text)) for page in pages]

    counts: Counter[str] = Counter()
    for page in pages:
        seen: set[str] = set()
        for line in page.text.splitlines():
            key = line.strip()
            if not key or key in seen or not _is_running_line(key):
                continue
            seen.add(key)
            counts[key] += 1
    threshold = max(3, math.ceil(len(pages) * 0.6))
    repeated = {line for line, count in counts.items() if count >= threshold}

    cleaned: list[Page] = []
    for page in pages:
        kept = [
            line for line in page.text.splitlines() if line.strip() not in repeated and not _PAGE_NUM_RE.match(line)
        ]
        cleaned.append(Page(page.number, "\n".join(kept)))
    return cleaned


def load_file(path: Path, settings: Settings) -> Document:
    """Load a ``.pdf``, ``.txt``, or ``.md`` file."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path.read_bytes(), path.name, settings)
    if suffix in _TEXT_SUFFIXES:
        return load_text(decode_bytes(path.read_bytes()), path.name, settings)
    raise DocumentError(
        "Unsupported file type. Use a PDF, .txt, or .md file.",
        "unsupported",
    )


def load_text(text: str, source_name: str, settings: Settings) -> Document:
    """Load pasted or decoded text as a one-page document."""
    return _finish([Page(1, text)], source_name, settings, kind="text")


def load_pdf(data: bytes, source_name: str, settings: Settings) -> Document:
    """Extract a PDF page by page with PyMuPDF.

    Encrypted files and pages with almost no text (typical of scans) raise
    :class:`DocumentError`. OCR is intentionally not attempted.
    """
    try:
        pdf = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:
        lowered = str(exc).lower()
        if "encrypt" in lowered or "password" in lowered:
            raise DocumentError(
                "This PDF is encrypted or password-protected. Export a copy without a password and try again.",
                "encrypted",
            ) from exc
        raise DocumentError(f"Could not read this PDF ({exc}).", "invalid") from exc

    try:
        if bool(getattr(pdf, "needs_pass", False)):
            raise DocumentError(
                "This PDF is encrypted or password-protected. Export a copy without a password and try again.",
                "encrypted",
            )
        page_count = int(pdf.page_count)
        if page_count > settings.max_pages:
            assumed_chars = page_count * 2000
            minutes = rough_minutes(assumed_chars, settings.chunk_tokens)
            raise DocumentError(
                (
                    f"This PDF has {page_count} pages (limit {settings.max_pages}). "
                    f"On a small model it would take about {minutes} minutes. "
                    "Raise MAX_PAGES if you still want to run it."
                ),
                "too_large",
            )
        pages = []
        for index in range(page_count):
            raw = pdf[index].get_text("text")
            text = raw if isinstance(raw, str) else str(raw or "")
            pages.append(Page(index + 1, text))
    finally:
        pdf.close()

    return _finish(pages, source_name, settings, kind="pdf")


def _finish(pages: list[Page], source_name: str, settings: Settings, kind: str) -> Document:
    raw_chars = sum(len(page.text) for page in pages)
    non_ws = sum(len(re.sub(r"\s+", "", page.text)) for page in pages)
    if non_ws < 20:
        if kind == "pdf":
            message = "This PDF has almost no extractable text. It may be a scan. OCR is not supported."
        else:
            message = "This document is empty."
        raise DocumentError(message, "empty")
    if raw_chars > settings.max_chars or len(pages) > settings.max_pages:
        minutes = rough_minutes(raw_chars, settings.chunk_tokens)
        raise DocumentError(
            (
                f"This document is too large ({len(pages)} pages, {raw_chars} characters). "
                f"Limits are {settings.max_pages} pages and {settings.max_chars} characters. "
                f"On a small model it would take about {minutes} minutes. "
                "Raise MAX_PAGES or MAX_CHARS if you still want to run it."
            ),
            "too_large",
        )

    pages = strip_running_elements(pages)
    pages = [Page(page.number, cleanup_text(page.text)) for page in pages]
    text = _assemble(pages)
    if len(re.sub(r"\s+", "", text)) < 20:
        if kind == "pdf":
            message = "This PDF has almost no extractable text. It may be a scan. OCR is not supported."
        else:
            message = "This document is empty."
        raise DocumentError(message, "empty")
    logger.info(
        "loaded document pages=%s chars=%s words=%s",
        len(pages),
        len(text),
        word_count(text),
    )
    return Document(
        source_name=source_name,
        pages=pages,
        text=text,
        word_count=word_count(text),
    )


def _assemble(pages: list[Page]) -> str:
    parts: list[str] = []
    for page in pages:
        body = page.text.strip()
        if not body:
            continue
        parts.append(f"[p. {page.number}]\n{body}")
    return "\n\n".join(parts)


def _strip_page_numbers(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if not _PAGE_NUM_RE.match(line))


def _is_running_line(line: str) -> bool:
    if _PAGE_NUM_RE.match(line):
        return True
    return len(line) <= 80 and len(line.split()) <= 10


def _protect_code_lines(text: str, hold: Callable[[str], str]) -> str:
    lines = text.split("\n")
    output: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        nonlocal buffer
        if len(buffer) >= 2:
            output.append(hold("\n".join(buffer)))
        else:
            output.extend(buffer)
        buffer = []

    for line in lines:
        if _CODE_START_RE.match(line) and "\x00CODE" not in line:
            buffer.append(line)
        else:
            flush()
            output.append(line)
    flush()
    return "\n".join(output)


def _clean_prose(text: str) -> str:
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    paragraphs = re.split(r"\n\s*\n", text)
    cleaned: list[str] = []
    for paragraph in paragraphs:
        if "\x00CODE" in paragraph:
            cleaned.append(paragraph.strip())
            continue
        joined = re.sub(r"[ \t]*\n[ \t]*", " ", paragraph).strip()
        joined = re.sub(r"[ \t]{2,}", " ", joined)
        if joined:
            cleaned.append(joined)
    return "\n\n".join(cleaned)
