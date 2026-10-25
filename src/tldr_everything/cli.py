"""Command-line interface for tldr-everything."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer

from tldr_everything import __version__
from tldr_everything.config import Settings, load_settings
from tldr_everything.export import to_json, to_markdown
from tldr_everything.extract import Document, DocumentError, load_file, load_text
from tldr_everything.llm import LLMError, build_client, collect_startup_warnings
from tldr_everything.models import ProgressUpdate, Result
from tldr_everything.pipeline import Pipeline

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Turn a PDF or pasted text into layered summaries.",
)

_LAYERS = {"30s", "3m", "notes", "all"}


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def _root(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version_callback, is_eager=True),
    ] = False,
) -> None:
    """The Reading Debt Killer."""


@app.command()
def summarize(
    file: Annotated[Path | None, typer.Argument(help="PDF, TXT, or Markdown file")] = None,
    text: Annotated[str | None, typer.Option("--text", help="Summarize this text instead of a file")] = None,
    layer: Annotated[str, typer.Option("--layer", help="30s, 3m, notes, or all")] = "all",
    out: Annotated[Path | None, typer.Option("--out", help="Write the result to this file")] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON instead of Markdown")] = False,
    model: Annotated[str | None, typer.Option("--model", help="Override LLM_MODEL")] = None,
    base_url: Annotated[str | None, typer.Option("--base-url", help="Override LLM_BASE_URL")] = None,
    language: Annotated[str | None, typer.Option("--language", help="Output language")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", help="Log call types and timings")] = False,
) -> None:
    """Summarize a document in the standard three layers."""
    _configure_logging(verbose)
    if layer not in _LAYERS:
        raise typer.BadParameter("layer must be 30s, 3m, notes, or all")
    settings = _settings(model, base_url)
    _warn(settings)
    document = _load(file, text, settings)
    layers = {"30s", "3m", "notes"} if layer == "all" else {layer}
    pipeline = Pipeline(build_client(settings), settings)
    try:
        result = pipeline.summarize(
            document,
            layers=layers,
            language=language,
            progress=_progress,
        )
    except (LLMError, DocumentError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    _emit(result, out, json_output)


@app.command()
def goal(
    file: Annotated[Path | None, typer.Argument(help="PDF, TXT, or Markdown file")] = None,
    goal_text: Annotated[str, typer.Option("--goal", help="What you need from the document")] = "",
    text: Annotated[str | None, typer.Option("--text", help="Read this text instead of a file")] = None,
    out: Annotated[Path | None, typer.Option("--out", help="Write the result to this file")] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON instead of Markdown")] = False,
    model: Annotated[str | None, typer.Option("--model", help="Override LLM_MODEL")] = None,
    base_url: Annotated[str | None, typer.Option("--base-url", help="Override LLM_BASE_URL")] = None,
    language: Annotated[str | None, typer.Option("--language", help="Output language")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", help="Log call types and timings")] = False,
) -> None:
    """Tell me only what I need for a stated goal."""
    _configure_logging(verbose)
    if not goal_text.strip():
        raise typer.BadParameter("Pass --goal with what you need from the document.")
    settings = _settings(model, base_url)
    _warn(settings)
    document = _load(file, text, settings)
    pipeline = Pipeline(build_client(settings), settings)
    try:
        result = pipeline.goal(
            document,
            goal_text.strip(),
            language=language,
            progress=_progress,
        )
    except (LLMError, DocumentError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    _emit(result, out, json_output)


@app.command()
def ui() -> None:
    """Launch the Streamlit app."""
    import sys

    from streamlit.web import cli as stcli

    path = Path(__file__).resolve().parent / "ui.py"
    sys.argv = ["streamlit", "run", str(path)]
    raise SystemExit(stcli.main())


def _configure_logging(verbose: bool) -> None:
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


def _settings(model: str | None, base_url: str | None) -> Settings:
    overrides: dict[str, object] = {}
    if model:
        overrides["llm_model"] = model
    if base_url:
        overrides["llm_base_url"] = base_url
    return load_settings(**overrides)


def _warn(settings: Settings) -> None:
    for warning in collect_startup_warnings(settings):
        typer.echo(f"Warning: {warning}", err=True)


def _load(file: Path | None, text: str | None, settings: Settings) -> Document:
    if file is not None and text is not None:
        typer.echo("Pass a file or --text, not both.", err=True)
        raise typer.Exit(code=1)
    try:
        if file is not None:
            return load_file(file, settings)
        if text is not None:
            return load_text(text, "pasted text", settings)
    except DocumentError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo("Pass a file path or --text.", err=True)
    raise typer.Exit(code=1)


def _progress(update: ProgressUpdate) -> None:
    eta = ""
    if update.eta_seconds is not None:
        eta = f", about {max(1, round(update.eta_seconds))}s left"
    typer.echo(f"[{update.done}/{update.total}] {update.label}{eta}", err=True)


def _emit(result: Result, out: Path | None, json_output: bool) -> None:
    payload = to_json(result) if json_output else to_markdown(result)
    if out is not None:
        out.write_text(payload, encoding="utf-8")
        typer.echo(f"Wrote {out}")
        return
    typer.echo(payload)
