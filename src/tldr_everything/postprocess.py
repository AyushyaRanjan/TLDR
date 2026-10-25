"""Checks and cleanup applied to model output. The model is not trusted for these."""

from __future__ import annotations

import re

from tldr_everything.models import WarningFlag

_LEADING_RE = re.compile(
    r"^(?:\s*(?:sure|certainly|of course|absolutely|here(?:'s| is| are)|"
    r"below is|the following is|glad to|happy to|okay|ok)[^\n]*\n)+",
    re.IGNORECASE,
)
_TRAILING_RE = re.compile(
    r"\n+\s*(?:let me know|hope this helps|i hope this|if you need|feel free to ask|"
    r"is there anything else)[^\n]*\s*$",
    re.IGNORECASE,
)
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", re.UNICODE)
_FENCE_RE = re.compile(r"```([^\n`]*)\n(.*?)```", re.DOTALL)
_INLINE_RE = re.compile(r"`([^`\n]+)`")
_VERSION_RE = re.compile(r"\b\d+\.\d+(?:\.\d+)*\b")
_LONG_NUMBER_RE = re.compile(r"\b\d{3,}\b")
_COMMAND_RE = re.compile(
    r"^(?:sudo|docker|npm|npx|pip3?|git|curl|wget|ssh|apt|apt-get|systemctl|"
    r"make|ollama|python3?|pytest|yarn|pnpm|cargo|go)\b"
)


def word_count(text: str) -> int:
    """Count words. Numbers and identifiers count as words."""
    if not text:
        return 0
    return len(_WORD_RE.findall(text))


def strip_chatter(text: str) -> str:
    """Remove a chatty opening line and a trailing offer of more help."""
    cleaned = text.replace("\r\n", "\n").strip()
    cleaned = _LEADING_RE.sub("", cleaned).strip()
    cleaned = _TRAILING_RE.sub("", cleaned).strip()
    return cleaned


def parse_score(text: str) -> int:
    """Parse ``SCORE: N``. Garbage input returns 0 so the chunk is dropped."""
    if not text:
        return 0
    match = re.search(r"SCORE\s*:\s*([0-5])\b", text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    match = re.search(r"\b([0-5])\s*/\s*5\b", text)
    if match:
        return int(match.group(1))
    return 0


def length_in_range(text: str, low: int, high: int) -> bool:
    """Return whether the word count sits inside ``low`` and ``high`` inclusive."""
    count = word_count(text)
    return low <= count <= high


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _is_command(body: str) -> bool:
    stripped = body.strip()
    if not stripped:
        return False
    if _COMMAND_RE.match(stripped):
        return True
    return " " in stripped and not stripped.endswith(".")


def ground_practical_steps(text: str, source: str) -> tuple[str, list[WarningFlag]]:
    """Drop fenced code that is not in the source, and mark unknown inline commands.

    Comparison ignores differences in whitespace only. Numbers and versions that
    never appear in the source are flagged and left in the text.
    """
    flags: list[WarningFlag] = []
    normalized_source = _normalize(source)
    dropped = 0

    def replace_fence(match: re.Match[str]) -> str:
        nonlocal dropped
        body = match.group(2)
        if body.strip() and _normalize(body) in normalized_source:
            return match.group(0)
        dropped += 1
        return ""

    grounded = _FENCE_RE.sub(replace_fence, text)
    if dropped:
        flags.append(
            WarningFlag(
                code="grounding",
                message=(f"Dropped {dropped} code block(s) that do not appear in the source."),
            )
        )

    def replace_inline(match: re.Match[str]) -> str:
        body = match.group(1)
        if not _is_command(body) or _normalize(body) in normalized_source:
            return match.group(0)
        flags.append(WarningFlag(code="grounding", message=f"Unverified command: {body.strip()}"))
        return f"{match.group(0)} [unverified]"

    grounded = _INLINE_RE.sub(replace_inline, grounded)
    flags.extend(flag_unsupported_numbers(grounded, source))
    return grounded.strip(), flags


def flag_unsupported_numbers(text: str, source: str) -> list[WarningFlag]:
    """Flag versions and longer numbers that the source never states."""
    missing: list[str] = []
    for number in _VERSION_RE.findall(text):
        if number not in source:
            missing.append(number)
    for number in _LONG_NUMBER_RE.findall(text):
        if number not in source and number not in missing:
            missing.append(number)
    if not missing:
        return []
    unique = list(dict.fromkeys(missing))
    return [
        WarningFlag(
            code="numbers",
            message="Numbers or versions not found in the source: " + ", ".join(unique),
        )
    ]


def dedupe_lines(text: str) -> str:
    """Drop duplicate lines, ignoring case and repeated whitespace."""
    seen: set[str] = set()
    output: list[str] = []
    for line in text.splitlines():
        key = re.sub(r"\s+", " ", line).strip().lower()
        if not key:
            if output and output[-1] != "":
                output.append("")
            continue
        if key in seen:
            continue
        seen.add(key)
        output.append(line.rstrip())
    return "\n".join(output).strip()
