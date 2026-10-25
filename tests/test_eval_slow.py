"""Optional live-model check. Deselected by default and in CI."""

from __future__ import annotations

import pytest

from tldr_everything.llm import LLMConnectionError, LLMModelMissingError


@pytest.mark.slow
def test_live_eval_smoke() -> None:
    from evals.run_eval import main

    try:
        report = main()
    except (LLMConnectionError, LLMModelMissingError) as exc:
        pytest.skip(str(exc))
    assert report
