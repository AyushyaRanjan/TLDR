"""Map, recursive reduce, and goal-mode orchestration.

Each model call does one job. Length, grounding, and prompt size are checked
in this module rather than trusted to the model.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from tldr_everything import prompts
from tldr_everything.chunking import Chunk, chunk_document
from tldr_everything.config import READING_WPM, Settings, estimate_tokens
from tldr_everything.extract import Document
from tldr_everything.llm import LLMClient
from tldr_everything.models import (
    ClosestSection,
    DocumentStats,
    GoalResult,
    LayerResult,
    ProgressUpdate,
    StandardResult,
    WarningFlag,
)
from tldr_everything.postprocess import (
    dedupe_lines,
    flag_unsupported_numbers,
    ground_practical_steps,
    length_in_range,
    parse_score,
    strip_chatter,
    word_count,
)
from tldr_everything.prompts import Prompt

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[ProgressUpdate], None]
TokenCallback = Callable[[str], None]

_GOAL_SECTIONS: tuple[tuple[str, str], ...] = (
    ("goal_prerequisites", "prerequisites"),
    ("goal_steps", "steps"),
    ("goal_relevant_parts", "relevant_parts"),
    ("goal_skip", "skip"),
    ("goal_gotchas", "gotchas"),
)


@dataclass(frozen=True)
class LengthTarget:
    """Ideal word range, plus a wider range that still counts as acceptable."""

    low: int
    high: int
    tolerant_low: int
    tolerant_high: int


_DETAIL_TARGETS: dict[str, dict[str, LengthTarget]] = {
    "brief": {
        "30s": LengthTarget(40, 70, 25, 100),
        "3m": LengthTarget(200, 320, 150, 420),
    },
    "normal": {
        "30s": LengthTarget(60, 90, 40, 130),
        "3m": LengthTarget(350, 500, 250, 650),
    },
    "deep": {
        "30s": LengthTarget(90, 130, 60, 180),
        "3m": LengthTarget(500, 700, 350, 900),
    },
}


def normalize_detail(detail: str) -> str:
    """Return a known detail level. Unknown values fall back to normal."""
    value = detail.strip().lower()
    if value not in _DETAIL_TARGETS:
        return "normal"
    return value


def normalize_language(language: str | None, settings: Settings) -> str:
    """Pick the output language. English is the default for a 3B model."""
    if language and language.strip():
        return language.strip()
    if settings.language and settings.language.strip():
        return settings.language.strip()
    return "English"


def estimate_eta(latencies: list[float], remaining: int) -> float | None:
    """ETA from the mean observed call latency. ``None`` until one call finishes."""
    if not latencies or remaining <= 0:
        return None
    return (sum(latencies) / len(latencies)) * remaining


class Pipeline:
    """Run Standard or Goal mode against an :class:`LLMClient`."""

    def __init__(self, llm: LLMClient, settings: Settings) -> None:
        self.llm = llm
        self.settings = settings
        self._progress: ProgressCallback | None = None
        self._on_token: TokenCallback | None = None
        self._on_reset: Callable[[], None] | None = None
        self._done = 0
        self._total = 1
        self._latencies: list[float] = []

    def summarize(
        self,
        document: Document,
        *,
        layers: set[str] | None = None,
        language: str | None = None,
        detail: str = "normal",
        progress: ProgressCallback | None = None,
        on_token: TokenCallback | None = None,
        on_reset: Callable[[], None] | None = None,
    ) -> StandardResult:
        """Build the requested layers. ``layers`` is a subset of 30s, 3m, and notes."""
        selected = layers or {"30s", "3m", "notes"}
        language = normalize_language(language, self.settings)
        detail = normalize_detail(detail)
        self._bind(progress, on_token, on_reset)
        chunks = chunk_document(document, self.settings)
        want_summary = bool(selected & {"30s", "3m"})
        want_notes = "notes" in selected
        final_calls = (
            "summary_30s" if "30s" in selected else None,
            "explain_3m" if "3m" in selected else None,
        )
        source_calls = [name for name in final_calls if name]
        single_pass = want_summary and self._fits_all(document.text, source_calls, language, detail)
        notes_single = want_notes and self._fits(document.text, "chunk_notes", language, detail)
        map_calls = (len(chunks) if want_summary and not single_pass else 0) + (
            len(chunks) if want_notes and not notes_single else 0
        )
        tail = len(source_calls) + (2 if want_notes else 0) + (1 if want_notes and notes_single else 0)
        self._total = max(1, map_calls + tail)
        self._done = 0
        self._latencies = []

        reduced = document.text
        if want_summary and not single_pass:
            summaries = self._map(
                chunks,
                "chunk_summary",
                lambda chunk: prompts.chunk_summary(chunk.text, language=language),
                "Summarizing chunks",
            )
            budget_call = source_calls[0] if source_calls else "summary_30s"
            budget = self._input_budget(budget_call, language, detail)
            reduced, truncated = self.reduce_pieces(summaries, budget, language)
            if truncated:
                logger.warning("chunk summaries were shortened to fit the context window")

        summary_30s = None
        explain_3m = None
        if "30s" in selected:
            summary_30s = self._length_checked(
                "summary_30s",
                reduced,
                document.text,
                language,
                detail,
                "30s",
                "Writing the 30-second summary",
            )
        if "3m" in selected:
            explain_3m = self._length_checked(
                "explain_3m",
                reduced,
                document.text,
                language,
                detail,
                "3m",
                "Writing the 3-minute explanation",
            )

        notes = None
        if want_notes:
            notes = self._build_notes(
                document,
                chunks,
                language,
                notes_single=notes_single,
                reduced=reduced,
            )

        stats = self._stats(document, chunks, summary_30s, explain_3m, notes)
        return StandardResult(
            summary_30s=summary_30s,
            explain_3m=explain_3m,
            notes=notes,
            stats=stats,
            single_pass=single_pass if want_summary else notes_single,
        )

    def goal(
        self,
        document: Document,
        goal: str,
        *,
        language: str | None = None,
        progress: ProgressCallback | None = None,
        on_token: TokenCallback | None = None,
        on_reset: Callable[[], None] | None = None,
    ) -> GoalResult:
        """Run the three-stage goal pipeline for one stated goal."""
        language = normalize_language(language, self.settings)
        self._bind(progress, on_token, on_reset)
        chunks = chunk_document(document, self.settings)
        self._total = max(1, len(chunks))
        self._done = 0
        self._latencies = []
        scores = self._score_chunks(chunks, goal, language)
        threshold = self.settings.goal_relevance_threshold
        kept = [(chunk, score) for chunk, score in scores if score >= threshold]
        stats = self._stats(document, chunks, None, None, None)
        if not kept:
            ranked = sorted(scores, key=lambda item: item[1], reverse=True)[:3]
            closest = [
                ClosestSection(
                    start_page=chunk.start_page,
                    end_page=chunk.end_page,
                    score=score,
                    excerpt=_excerpt(chunk.text),
                )
                for chunk, score in ranked
            ]
            return GoalResult(
                goal=goal,
                covered=False,
                message=(
                    "This document does not appear to cover that goal. "
                    "Nothing was invented. The closest sections are below."
                ),
                closest=closest,
                stats=stats,
            )

        self._total += len(kept) * len(_GOAL_SECTIONS)
        extracted: dict[str, list[str]] = {field: [] for _, field in _GOAL_SECTIONS}
        for chunk, _score in kept:
            for call_type, field in _GOAL_SECTIONS:
                prompt = _goal_prompt(call_type, chunk.text, goal, language)
                text = self._timed_call(call_type, prompt, stream=False, label="Extracting relevant parts")
                extracted[field].append(text)

        sections: dict[str, LayerResult] = {}
        for call_type, field in _GOAL_SECTIONS:
            merged = self._merge_section(extracted[field], call_type, goal, language)
            if field == "steps":
                grounded, flags = ground_practical_steps(merged, document.text)
                sections[field] = LayerResult(text=grounded, warnings=flags)
            else:
                sections[field] = LayerResult(
                    text=merged,
                    warnings=flag_unsupported_numbers(merged, document.text),
                )
        summary_words = word_count(sections["steps"].text)
        stats = stats.model_copy(
            update={
                "summary_words": summary_words,
                "summary_minutes": summary_words / READING_WPM,
            }
        )
        return GoalResult(
            goal=goal,
            covered=True,
            prerequisites=sections["prerequisites"],
            steps=sections["steps"],
            relevant_parts=sections["relevant_parts"],
            skip=sections["skip"],
            gotchas=sections["gotchas"],
            stats=stats,
        )

    def reduce_pieces(
        self,
        pieces: list[str],
        budget_tokens: int,
        language: str,
    ) -> tuple[str, bool]:
        """Hierarchically shorten pieces until they fit ``budget_tokens``.

        Returns the joined text and whether a hard truncation was required.
        Groups are reduced with one call each. A round that cannot pack more
        than one piece per group forces pairs so the round still shrinks.
        """
        budget = max(1, budget_tokens)
        current = [piece.strip() for piece in pieces if piece.strip()]
        if not current:
            return "", False
        truncated = False
        for _round in range(6):
            if estimate_tokens("\n\n".join(current)) <= budget:
                return "\n\n".join(current), truncated
            if len(current) == 1:
                shortened = _hard_truncate(current[0], budget)
                return shortened, shortened != current[0]
            groups = _pack(current, budget)
            if len(groups) == len(current):
                groups = [current[index : index + 2] for index in range(0, len(current), 2)]
            reduced: list[str] = []
            for group in groups:
                if len(group) == 1 and estimate_tokens(group[0]) <= budget and len(groups) == 1:
                    reduced.append(group[0])
                    continue
                self._total += 1
                prompt = prompts.reduce_summary("\n\n".join(group), language=language)
                reduced.append(self._timed_call("reduce_summary", prompt, stream=False, label="Combining summaries"))
            if len(reduced) >= len(current) and estimate_tokens("\n\n".join(reduced)) >= estimate_tokens(
                "\n\n".join(current)
            ):
                truncated = True
                break
            current = reduced
        joined = "\n\n".join(current)
        if estimate_tokens(joined) > budget:
            return _hard_truncate(joined, budget), True
        return joined, truncated

    def _build_notes(
        self,
        document: Document,
        chunks: list[Chunk],
        language: str,
        *,
        notes_single: bool,
        reduced: str,
    ) -> LayerResult:
        warnings: list[WarningFlag] = []
        if notes_single:
            prompt = prompts.chunk_notes(document.text, language=language)
            body = self._timed_call("chunk_notes", prompt, stream=False, label="Extracting notes")
            assembled = f"### Pages {chunks[0].start_page}-{chunks[-1].end_page}\n\n{body}"
        else:
            notes = self._map(
                chunks,
                "chunk_notes",
                lambda chunk: prompts.chunk_notes(chunk.text, language=language),
                "Extracting notes",
            )
            blocks = []
            for chunk, note in zip(chunks, notes, strict=True):
                blocks.append(f"### Pages {chunk.start_page}-{chunk.end_page}\n\n{note}")
            assembled = "\n\n".join(blocks)
        takeaways = self._timed_call(
            "key_takeaways",
            prompts.key_takeaways(reduced if reduced.strip() else assembled, language=language),
            stream=True,
            label="Listing key takeaways",
        )
        questions = self._timed_call(
            "open_questions",
            prompts.open_questions(reduced if reduced.strip() else assembled, language=language),
            stream=True,
            label="Listing open questions",
        )
        text = (
            f"{assembled}\n\n## Key takeaways\n\n{takeaways}\n\n## Questions this document leaves open\n\n{questions}"
        )
        warnings.extend(flag_unsupported_numbers(text, document.text))
        return LayerResult(text=text.strip(), warnings=warnings)

    def _length_checked(
        self,
        call_type: str,
        source: str,
        reference: str,
        language: str,
        detail: str,
        key: str,
        label: str,
    ) -> LayerResult:
        target = _DETAIL_TARGETS[detail][key]
        prompt = _layer_prompt(call_type, source, language, target, strict=False)
        text = self._timed_call(call_type, prompt, stream=True, label=label)
        warnings: list[WarningFlag] = []
        if not length_in_range(text, target.tolerant_low, target.tolerant_high):
            if self._on_reset:
                self._on_reset()
            self._total += 1
            retry = _layer_prompt(call_type, source, language, target, strict=True)
            text = self._timed_call(call_type, retry, stream=True, label=label + " (length retry)")
            count = word_count(text)
            if not length_in_range(text, target.tolerant_low, target.tolerant_high):
                warnings.append(
                    WarningFlag(
                        code="length",
                        message=(
                            f"The {label.lower()} is outside the target range "
                            f"({count} words; expected {target.low}-{target.high})."
                        ),
                    )
                )
        warnings.extend(flag_unsupported_numbers(text, reference))
        return LayerResult(text=text, warnings=warnings)

    def _score_chunks(self, chunks: list[Chunk], goal: str, language: str) -> list[tuple[Chunk, int]]:
        raw = self._map(
            chunks,
            "goal_relevance",
            lambda chunk: prompts.goal_relevance(chunk.text, goal=goal, language=language),
            "Scoring chunks",
        )
        return [(chunk, parse_score(text)) for chunk, text in zip(chunks, raw, strict=True)]

    def _merge_section(self, parts: list[str], call_type: str, goal: str, language: str) -> str:
        merged = dedupe_lines("\n".join(part for part in parts if part.strip()))
        if not merged:
            return "Not covered in the document."
        budget = self._input_budget(call_type, language, "normal")
        if estimate_tokens(merged) <= budget:
            return merged
        self._total += 1
        prompt = _goal_prompt(call_type, merged, goal, language)
        rewritten = self._timed_call(call_type, prompt, stream=True, label="Merging a section")
        if estimate_tokens(rewritten) > budget:
            logger.warning("merged section still over budget for %s", call_type)
            return _hard_truncate(rewritten, budget)
        return rewritten

    def _map(
        self,
        chunks: list[Chunk],
        call_type: str,
        builder: Callable[[Chunk], Prompt],
        label: str,
    ) -> list[str]:
        if not chunks:
            return []
        results = [""] * len(chunks)
        workers = max(1, min(self.settings.llm_concurrency, len(chunks)))

        def job(index: int, chunk: Chunk) -> tuple[int, str, float]:
            prompt = builder(chunk)
            started = time.perf_counter()
            text = self._call(call_type, prompt, stream=False)
            return index, text, time.perf_counter() - started

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(job, index, chunk) for index, chunk in enumerate(chunks)]
            for future in as_completed(futures):
                index, text, elapsed = future.result()
                results[index] = text
                self._latencies.append(elapsed)
                self._done += 1
                self._emit(label)
        return results

    def _timed_call(self, call_type: str, prompt: Prompt, *, stream: bool, label: str) -> str:
        started = time.perf_counter()
        text = self._call(call_type, prompt, stream=stream)
        self._latencies.append(time.perf_counter() - started)
        self._done += 1
        self._emit(label)
        return text

    def _call(self, call_type: str, prompt: Prompt, *, stream: bool) -> str:
        system, user = self._fit(call_type, prompt.system, prompt.user)
        max_tokens = self.settings.max_tokens_for(call_type)
        logger.info(
            "llm call type=%s max_tokens=%s est_prompt_tokens=%s",
            call_type,
            max_tokens,
            estimate_tokens(system) + estimate_tokens(user),
        )
        if stream and self._on_token is not None:
            parts: list[str] = []
            for piece in self.llm.stream(system=system, user=user, max_tokens=max_tokens, call_type=call_type):
                parts.append(piece)
                self._on_token(piece)
            return strip_chatter("".join(parts))
        return strip_chatter(self.llm.complete(system=system, user=user, max_tokens=max_tokens, call_type=call_type))

    def _fit(self, call_type: str, system: str, user: str) -> tuple[str, str]:
        max_tokens = self.settings.max_tokens_for(call_type)
        limit = self.settings.llm_max_context_tokens - self.settings.safety_margin_tokens - max_tokens
        if estimate_tokens(system) + estimate_tokens(user) <= limit:
            return system, user
        logger.warning("truncating prompt for call_type=%s", call_type)
        start_tag = "<document>"
        end_tag = "</document>"
        if start_tag in user and end_tag in user:
            start = user.index(start_tag) + len(start_tag)
            end = user.rindex(end_tag)
            prefix = user[:start]
            suffix = user[end:]
            body = user[start:end]
            fitted = prefix + body + "\n" + suffix
            while body and estimate_tokens(system) + estimate_tokens(fitted) > limit:
                body = body[: max(0, len(body) - 400)]
                fitted = prefix + body + "\n" + suffix
            if estimate_tokens(system) + estimate_tokens(fitted) <= limit:
                return system, fitted
        room = max(0, limit - estimate_tokens(system))
        return system, user[: room * 4]

    def _within_budget(self, call_type: str, system: str, user: str) -> bool:
        limit = (
            self.settings.llm_max_context_tokens
            - self.settings.safety_margin_tokens
            - self.settings.max_tokens_for(call_type)
        )
        return estimate_tokens(system) + estimate_tokens(user) <= limit

    def _fits(self, text: str, call_type: str, language: str, detail: str) -> bool:
        prompt = _layer_prompt(
            call_type,
            text,
            language,
            _target_for(call_type, detail),
            strict=False,
        )
        return self._within_budget(call_type, prompt.system, prompt.user)

    def _fits_all(self, text: str, call_types: list[str], language: str, detail: str) -> bool:
        if not call_types:
            return True
        return all(self._fits(text, call_type, language, detail) for call_type in call_types)

    def _input_budget(self, call_type: str, language: str, detail: str) -> int:
        sample = _layer_prompt(call_type, "PLACEHOLDER", language, _target_for(call_type, detail), strict=False)
        overhead = estimate_tokens(sample.system) + estimate_tokens(sample.user.replace("PLACEHOLDER", ""))
        room = (
            self.settings.llm_max_context_tokens
            - self.settings.safety_margin_tokens
            - self.settings.max_tokens_for(call_type)
            - overhead
        )
        return max(100, room)

    def _bind(
        self,
        progress: ProgressCallback | None,
        on_token: TokenCallback | None,
        on_reset: Callable[[], None] | None,
    ) -> None:
        self._progress = progress
        self._on_token = on_token
        self._on_reset = on_reset

    def _emit(self, label: str) -> None:
        if self._progress is None:
            return
        remaining = max(0, self._total - self._done)
        self._progress(
            ProgressUpdate(
                done=min(self._done, self._total),
                total=max(self._total, 1),
                label=label,
                eta_seconds=estimate_eta(self._latencies, remaining),
            )
        )

    def _stats(
        self,
        document: Document,
        chunks: list[Chunk],
        summary_30s: LayerResult | None,
        explain_3m: LayerResult | None,
        notes: LayerResult | None,
    ) -> DocumentStats:
        primary = summary_30s or explain_3m or notes
        summary_words = word_count(primary.text) if primary else 0
        return DocumentStats(
            source_name=document.source_name,
            pages=len(document.pages),
            words=document.word_count,
            chunks=len(chunks),
            reading_minutes=document.word_count / READING_WPM,
            summary_words=summary_words,
            summary_minutes=summary_words / READING_WPM,
        )


def _target_for(call_type: str, detail: str) -> LengthTarget:
    table = _DETAIL_TARGETS[normalize_detail(detail)]
    if call_type == "explain_3m":
        return table["3m"]
    return table["30s"]


def _layer_prompt(
    call_type: str,
    text: str,
    language: str,
    target: LengthTarget,
    *,
    strict: bool,
) -> Prompt:
    if call_type == "summary_30s":
        return prompts.summary_30s(
            text,
            language=language,
            min_words=target.low,
            max_words=target.high,
            strict=strict,
        )
    if call_type == "explain_3m":
        return prompts.explain_3m(
            text,
            language=language,
            min_words=target.low,
            max_words=target.high,
            strict=strict,
        )
    if call_type == "chunk_notes":
        return prompts.chunk_notes(text, language=language)
    if call_type == "chunk_summary":
        return prompts.chunk_summary(text, language=language)
    if call_type == "reduce_summary":
        return prompts.reduce_summary(text, language=language)
    if call_type == "key_takeaways":
        return prompts.key_takeaways(text, language=language)
    if call_type == "open_questions":
        return prompts.open_questions(text, language=language)
    return _goal_prompt(call_type, text, "", language)


def _goal_prompt(call_type: str, text: str, goal: str, language: str) -> Prompt:
    builders = {
        "goal_relevance": prompts.goal_relevance,
        "goal_prerequisites": prompts.goal_prerequisites,
        "goal_steps": prompts.goal_steps,
        "goal_relevant_parts": prompts.goal_relevant_parts,
        "goal_skip": prompts.goal_skip,
        "goal_gotchas": prompts.goal_gotchas,
    }
    builder = builders[call_type]
    return builder(text, goal=goal, language=language)


def _pack(pieces: list[str], budget_tokens: int) -> list[list[str]]:
    groups: list[list[str]] = []
    current: list[str] = []
    tokens = 0
    for piece in pieces:
        piece_tokens = estimate_tokens(piece)
        separator = 2 if current else 0
        if current and tokens + separator + piece_tokens > budget_tokens:
            groups.append(current)
            current = [piece]
            tokens = piece_tokens
        else:
            current.append(piece)
            tokens += separator + piece_tokens
    if current:
        groups.append(current)
    return groups


def _hard_truncate(text: str, budget_tokens: int) -> str:
    return text[: max(1, budget_tokens) * 4].rstrip()


def _excerpt(text: str, limit: int = 500) -> str:
    flattened = " ".join(text.split())
    if len(flattened) <= limit:
        return flattened
    return flattened[: limit - 3].rstrip() + "..."
