"""In-memory LLM client for tests. No network and no model."""

from __future__ import annotations

from collections.abc import Callable

Handler = Callable[[str, str, str], str]


def words(count: int, token: str = "point") -> str:
    """Return ``count`` whitespace-separated tokens."""
    return " ".join(f"{token}{index}" for index in range(count))


DEFAULTS: dict[str, str] = {
    "chunk_summary": words(40, "chunk"),
    "chunk_notes": (
        "TAKEAWAYS:\n- Keep the main idea (p. 1).\n"
        "DEFINITIONS:\n- term: a word from the text.\n"
        "NUMBERS:\n- No extra figures."
    ),
    "reduce_summary": words(30, "reduced"),
    "summary_30s": words(75, "brief"),
    "explain_3m": words(400, "detail"),
    "key_takeaways": "\n".join(f"- Takeaway {index} stays in the notes." for index in range(1, 7)),
    "open_questions": "- How are backups handled?",
    "goal_relevance": "SCORE: 0",
    "goal_prerequisites": "- A machine you can log in to.",
    "goal_steps": "1. Read the section that matches the goal.",
    "goal_relevant_parts": "- The opening section (p. 1).",
    "goal_skip": "- Skip background that does not mention the goal.",
    "goal_gotchas": "Not covered in the document.",
}


class FakeLLM:
    """Records every call and returns scripted or default text."""

    def __init__(
        self,
        handler: Handler | None = None,
        scripts: dict[str, list[str]] | None = None,
    ) -> None:
        self.handler = handler
        self.scripts: dict[str, list[str]] = {key: list(value) for key, value in (scripts or {}).items()}
        self.calls: list[dict[str, object]] = []

    def complete(self, *, system: str, user: str, max_tokens: int, call_type: str) -> str:
        self.calls.append(
            {
                "system": system,
                "user": user,
                "max_tokens": max_tokens,
                "call_type": call_type,
            }
        )
        queued = self.scripts.get(call_type)
        if queued:
            return queued.pop(0)
        if self.handler is not None:
            return self.handler(call_type, system, user)
        return DEFAULTS.get(call_type, "Not covered in the document.")

    def stream(self, *, system: str, user: str, max_tokens: int, call_type: str) -> object:
        text = self.complete(system=system, user=user, max_tokens=max_tokens, call_type=call_type)
        midpoint = max(1, len(text) // 2)
        yield text[:midpoint]
        if text[midpoint:]:
            yield text[midpoint:]

    def types(self) -> list[str]:
        return [str(call["call_type"]) for call in self.calls]
