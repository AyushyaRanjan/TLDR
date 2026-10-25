"""Markdown/JSON export and the Typer CLI, still with a fake model client."""

from __future__ import annotations

import json
from pathlib import Path

import pymupdf
from typer.testing import CliRunner

from tests.conftest import make_settings
from tests.fakes import FakeLLM
from tldr_everything.cli import app
from tldr_everything.export import to_json, to_markdown
from tldr_everything.extract import load_text
from tldr_everything.models import WarningFlag
from tldr_everything.pipeline import Pipeline

runner = CliRunner()
NOTE = "A short note about reading debt, local models, and a PDF you saved months ago."


def _patch(monkeypatch: object) -> FakeLLM:
    fake = FakeLLM()
    monkeypatch.setattr("tldr_everything.cli.build_client", lambda _settings: fake)  # type: ignore[attr-defined]
    monkeypatch.setattr(  # type: ignore[attr-defined]
        "tldr_everything.cli.collect_startup_warnings",
        lambda _settings: ["context looks small"],
    )
    return fake


def test_export_includes_warnings_and_goal_sections() -> None:
    settings = make_settings()
    document = load_text(NOTE, "note.md", settings)
    result = Pipeline(FakeLLM(), settings).summarize(document, layers={"30s"})
    assert result.summary_30s is not None
    result.summary_30s.warnings.append(WarningFlag(code="length", message="too short in this fixture"))
    markdown = to_markdown(result)
    assert "30-second summary" in markdown
    assert "too short in this fixture" in markdown
    payload = json.loads(to_json(result))
    assert payload["mode"] == "standard"
    assert payload["summary_30s"]["warnings"][0]["code"] == "length"

    goal = Pipeline(FakeLLM(), settings).goal(document, "deploy with Docker")
    goal_markdown = to_markdown(goal)
    assert "does not appear to cover" in goal_markdown
    assert "Closest sections" in goal_markdown


def test_cli_summarize_json_and_file(tmp_path: Path, monkeypatch: object) -> None:
    _patch(monkeypatch)
    text_result = runner.invoke(app, ["summarize", "--text", NOTE, "--layer", "30s", "--json"])
    assert text_result.exit_code == 0, text_result.output
    payload = json.loads(text_result.stdout)
    assert payload["summary_30s"]["text"]
    assert "context looks small" in (text_result.stderr or "")

    pdf_path = tmp_path / "note.pdf"
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text((72, 72), NOTE)
    pdf.save(pdf_path)
    pdf.close()
    out = tmp_path / "out.md"
    file_result = runner.invoke(app, ["summarize", str(pdf_path), "--layer", "all", "--out", str(out)])
    assert file_result.exit_code == 0, file_result.output
    saved = out.read_text(encoding="utf-8")
    assert "30-second summary" in saved
    assert "3-minute explanation" in saved
    assert "Detailed notes" in saved


def test_cli_goal_and_usage_errors(monkeypatch: object) -> None:
    _patch(monkeypatch)
    goal_result = runner.invoke(
        app,
        [
            "goal",
            "--text",
            "Ducks live in the park and eat bread near the pond.",
            "--goal",
            "deploy with Docker",
        ],
    )
    assert goal_result.exit_code == 0, goal_result.output
    assert "does not appear to cover" in goal_result.stdout

    missing = runner.invoke(app, ["summarize"])
    assert missing.exit_code == 1
    both = runner.invoke(app, ["summarize", "file.pdf", "--text", NOTE])
    assert both.exit_code == 1
    empty = runner.invoke(app, ["summarize", "--text", "   "])
    assert empty.exit_code == 1
    assert "empty" in (empty.stderr or empty.output).lower()
    bad_layer = runner.invoke(app, ["summarize", "--text", NOTE, "--layer", "nope"])
    assert bad_layer.exit_code != 0
    no_goal = runner.invoke(app, ["goal", "--text", NOTE])
    assert no_goal.exit_code != 0
    version = runner.invoke(app, ["--version"])
    assert version.exit_code == 0
    assert version.stdout.strip()
    help_result = runner.invoke(app, ["--help"])
    assert "summarize" in help_result.stdout
    assert "goal" in help_result.stdout


def test_config_defaults_and_unknown_call_type() -> None:
    from tldr_everything.config import Settings

    assert Settings.model_fields["llm_model"].default == "qwen2.5:3b"
    assert Settings.model_fields["llm_base_url"].default == "http://localhost:11434/v1"
    assert Settings.model_fields["llm_max_context_tokens"].default == 8192
    settings = make_settings()
    try:
        settings.max_tokens_for("not-a-call")
    except KeyError as exc:
        assert "Unknown" in str(exc)
    else:
        raise AssertionError("expected an unknown call type to fail")
