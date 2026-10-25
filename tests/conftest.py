"""Shared settings builder. Explicit values win over the developer environment."""

from __future__ import annotations

from typing import Any

import pytest

from tldr_everything.config import Settings


def make_settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "llm_base_url": "http://localhost:11434/v1",
        "llm_api_key": "ollama",
        "llm_model": "qwen2.5:3b",
        "llm_max_context_tokens": 8192,
        "llm_temperature": 0.2,
        "llm_top_p": 0.9,
        "llm_concurrency": 1,
        "chunk_tokens": 1500,
        "chunk_overlap_tokens": 150,
        "max_pages": 80,
        "max_chars": 400_000,
        "goal_relevance_threshold": 2,
        "language": "English",
        "safety_margin_tokens": 256,
    }
    values.update(overrides)
    return Settings(**values)


@pytest.fixture
def settings() -> Settings:
    return make_settings()
