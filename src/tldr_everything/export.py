"""Markdown and JSON exports of pipeline results."""

from __future__ import annotations

import json

from tldr_everything.models import GoalResult, LayerResult, Result, StandardResult, WarningFlag


def to_json(result: Result) -> str:
    """Serialize a result, including warning flags."""
    return json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n"


def to_markdown(result: Result) -> str:
    """Render a result as Markdown suitable for saving next to the source."""
    if isinstance(result, StandardResult):
        return _standard_markdown(result)
    return _goal_markdown(result)


def _standard_markdown(result: StandardResult) -> str:
    lines = [
        _header(
            result.stats.source_name,
            result.stats.pages,
            result.stats.words,
            result.stats.reading_minutes,
        )
    ]
    if result.summary_30s is not None:
        lines.append("## 30-second summary\n")
        lines.append(_body(result.summary_30s))
    if result.explain_3m is not None:
        lines.append("## 3-minute explanation\n")
        lines.append(_body(result.explain_3m))
    if result.notes is not None:
        lines.append("## Detailed notes\n")
        lines.append(_body(result.notes))
    return "\n".join(lines).strip() + "\n"


def _goal_markdown(result: GoalResult) -> str:
    lines = [
        _header(
            result.stats.source_name,
            result.stats.pages,
            result.stats.words,
            result.stats.reading_minutes,
        ),
        f"## Goal\n\n{result.goal}\n",
    ]
    if not result.covered:
        lines.append(result.message + "\n")
        if result.closest:
            lines.append("## Closest sections\n")
            for section in result.closest:
                pages = (
                    f"p. {section.start_page}"
                    if section.start_page == section.end_page
                    else (f"pp. {section.start_page}-{section.end_page}")
                )
                lines.append(f"### {pages} (score {section.score})\n\n{section.excerpt}\n")
        return "\n".join(lines).strip() + "\n"
    sections = [
        ("Prerequisites", result.prerequisites),
        ("Practical steps", result.steps),
        ("Relevant parts only", result.relevant_parts),
        ("What you can safely skip", result.skip),
        ("Gotchas and warnings", result.gotchas),
    ]
    for title, layer in sections:
        lines.append(f"## {title}\n")
        lines.append(_body(layer))
    return "\n".join(lines).strip() + "\n"


def _header(name: str, pages: int, words: int, minutes: float) -> str:
    return (
        f"# tldr-everything\n\n"
        f"Source: {name} · {pages} page(s) · {words} words · "
        f"about {minutes:.1f} min to read the source\n"
    )


def _body(layer: LayerResult) -> str:
    parts = []
    for warning in layer.warnings:
        parts.append(_warning(warning))
    parts.append(layer.text.strip() + "\n")
    return "\n".join(parts)


def _warning(flag: WarningFlag) -> str:
    return f"> Warning ({flag.code}): {flag.message}\n"
