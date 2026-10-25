"""Runtime configuration loaded from the environment."""

from __future__ import annotations

from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict

# Output caps are intentionally small. Qwen2.5-3B follows a short format more
# reliably than an open-ended one, and the caps keep each call inside an 8192
# context window together with a chunk.
MAX_TOKENS: dict[str, int] = {
    "chunk_summary": 280,
    "chunk_notes": 400,
    "reduce_summary": 500,
    "summary_30s": 220,
    "explain_3m": 900,
    "key_takeaways": 350,
    "open_questions": 300,
    "goal_relevance": 32,
    "goal_prerequisites": 400,
    "goal_steps": 500,
    "goal_relevant_parts": 400,
    "goal_skip": 300,
    "goal_gotchas": 300,
}

READING_WPM = 220
# Used only for the pre-run "this will take about N minutes" message, before
# any real latency has been observed.
ASSUMED_SECONDS_PER_CALL = 30


def estimate_tokens(text: str) -> int:
    """Approximate tokens as characters / 4.

    A real tokenizer would be more accurate and heavier. This estimate is
    stable, dependency-free, and slightly conservative for English prose.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


class Settings(BaseSettings):
    """Settings read from the process environment and an optional ``.env`` file.

    Constructor arguments win over environment variables, which win over the
    dotenv file. That lets the CLI and tests override a single field.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str = "ollama"
    llm_model: str = "qwen2.5:3b"
    llm_max_context_tokens: int = 8192
    llm_temperature: float = 0.2
    llm_top_p: float = 0.9
    llm_concurrency: int = 2
    chunk_tokens: int = 1500
    chunk_overlap_tokens: int = 150
    max_pages: int = 80
    max_chars: int = 400_000
    goal_relevance_threshold: int = 2
    language: str = "English"
    safety_margin_tokens: int = 256

    def max_tokens_for(self, call_type: str) -> int:
        """Return the output cap for one call type."""
        try:
            return MAX_TOKENS[call_type]
        except KeyError as exc:
            raise KeyError(f"Unknown LLM call type: {call_type}") from exc


def load_settings(**overrides: Any) -> Settings:
    """Build settings, applying explicit overrides on top of the environment."""
    if not overrides:
        return Settings()
    return Settings(**overrides)
