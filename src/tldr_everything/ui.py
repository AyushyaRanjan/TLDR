"""Streamlit interface for tldr-everything.

Launch with ``tldr ui`` or ``streamlit run src/tldr_everything/ui.py``.
The app imports Streamlit only when this module is loaded, which is what
``streamlit run`` and the ``tldr ui`` command both do.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from tldr_everything.config import Settings, load_settings
from tldr_everything.export import to_json, to_markdown
from tldr_everything.extract import Document, DocumentError, decode_bytes, load_pdf, load_text
from tldr_everything.llm import LLMError, build_client, collect_startup_warnings, endpoint_is_local
from tldr_everything.models import GoalResult, ProgressUpdate, StandardResult, WarningFlag
from tldr_everything.pipeline import Pipeline

_LANGUAGES = ("English", "Spanish", "French", "German", "Hindi", "Chinese")


def main() -> None:
    """Render the sidebar, the input area, and the layered result."""
    import streamlit as st

    st.set_page_config(page_title="tldr-everything", layout="wide")
    st.title("tldr-everything")
    st.caption("The Reading Debt Killer. Summaries stay on this machine unless you set a remote URL.")

    defaults = load_settings()
    with st.sidebar:
        mode = st.radio("Mode", ["Standard", "Tell me only what I need"])
        goal = st.text_area(
            "Goal",
            placeholder="I want to deploy this with Docker on a VPS",
            help="Used only in goal mode.",
        )
        model = st.text_input("Model", value=defaults.llm_model)
        base_url = st.text_input("Base URL", value=defaults.llm_base_url)
        detail = st.selectbox("Detail", ["brief", "normal", "deep"], index=1)
        language = st.selectbox("Language", list(_LANGUAGES), index=0)
        st.caption("A 3B model is weaker in languages other than English.")

    settings = load_settings(
        llm_model=model.strip() or defaults.llm_model,
        llm_base_url=base_url.strip() or defaults.llm_base_url,
        language=language,
    )
    warn_key = (settings.llm_model, settings.llm_base_url)
    if st.session_state.get("warn_key") != warn_key:
        st.session_state.warn_key = warn_key
        st.session_state.startup_warnings = collect_startup_warnings(settings)
    for warning in st.session_state.get("startup_warnings", []):
        st.warning(warning)
    if not endpoint_is_local(settings.llm_base_url):
        st.warning("This base URL is remote. The document text will leave this machine.")

    uploaded = st.file_uploader("PDF, TXT, or Markdown", type=["pdf", "txt", "md", "markdown"])
    pasted = st.text_area("Or paste text", height=220)
    run = st.button("Summarize", type="primary")

    if run:
        try:
            document = _load_input(uploaded, pasted, settings)
        except DocumentError as exc:
            st.error(str(exc))
            return
        _show_stats_preview(document.word_count, len(document.pages))
        progress = st.progress(0.0)
        status = st.empty()
        live = st.empty()
        buffers: dict[str, list[str]] = {}
        current = {"label": "Working"}

        def on_progress(update: ProgressUpdate) -> None:
            current["label"] = update.label
            fraction = 0.0 if update.total <= 0 else min(1.0, update.done / update.total)
            progress.progress(fraction)
            eta = ""
            if update.eta_seconds is not None:
                eta = f" — about {max(1, round(update.eta_seconds))}s left"
            status.write(f"{update.label} ({update.done}/{update.total}){eta}")

        def on_token(piece: str) -> None:
            buffers.setdefault(current["label"], []).append(piece)
            live.markdown(f"**{current['label']}**\n\n" + "".join(buffers[current["label"]]))

        def on_reset() -> None:
            buffers[current["label"]] = []
            live.empty()

        pipeline = Pipeline(build_client(settings), settings)
        result: StandardResult | GoalResult
        try:
            if mode == "Standard":
                result = pipeline.summarize(
                    document,
                    language=language,
                    detail=str(detail),
                    progress=on_progress,
                    on_token=on_token,
                    on_reset=on_reset,
                )
            else:
                if not goal.strip():
                    st.error("Write a goal first, for example: I want to deploy this with Docker on a VPS.")
                    return
                result = pipeline.goal(
                    document,
                    goal.strip(),
                    language=language,
                    progress=on_progress,
                    on_token=on_token,
                    on_reset=on_reset,
                )
        except (LLMError, DocumentError) as exc:
            st.error(str(exc))
            return
        progress.progress(1.0)
        live.empty()
        st.session_state.result = result
        st.session_state.result_md = to_markdown(result)
        st.session_state.result_json = to_json(result)

    stored = st.session_state.get("result")
    if isinstance(stored, StandardResult):
        _render_standard(stored)
    elif isinstance(stored, GoalResult):
        _render_goal(stored)


def _load_input(uploaded: Any, pasted: str, settings: Settings) -> Document:
    if uploaded is not None:
        name = str(getattr(uploaded, "name", "upload"))
        raw = uploaded.getvalue()  # type: ignore[attr-defined]
        data = raw if isinstance(raw, bytes) else bytes(raw)
        if name.lower().endswith(".pdf"):
            return load_pdf(data, name, settings)
        return load_text(decode_bytes(data), name, settings)
    if pasted.strip():
        return load_text(pasted, "pasted text", settings)
    raise DocumentError("Upload a file or paste some text.", "empty")


def _show_stats_preview(words: int, pages: int) -> None:
    import streamlit as st

    minutes = words / 220 if words else 0
    st.write(f"{pages} page(s) · {words} words · about {minutes:.1f} min to read the source")


def _render_standard(result: StandardResult) -> None:
    import streamlit as st

    _metrics(
        result.stats.pages,
        result.stats.words,
        result.stats.reading_minutes,
        result.stats.summary_minutes,
    )
    _badges(result.all_warnings())
    tabs = st.tabs(["30-second summary", "3-minute explanation", "Detailed notes"])
    layers = (result.summary_30s, result.explain_3m, result.notes)
    for tab, layer in zip(tabs, layers, strict=True):
        with tab:
            if layer is None or not layer.text.strip():
                st.info("This layer was not generated.")
            else:
                _badges(layer.warnings)
                st.markdown(layer.text)
    _downloads()


def _render_goal(result: GoalResult) -> None:
    import streamlit as st

    _metrics(
        result.stats.pages,
        result.stats.words,
        result.stats.reading_minutes,
        result.stats.summary_minutes,
    )
    st.markdown(f"**Goal:** {result.goal}")
    if not result.covered:
        st.warning(result.message)
        for section in result.closest:
            pages = (
                f"p. {section.start_page}"
                if section.start_page == section.end_page
                else (f"pp. {section.start_page}-{section.end_page}")
            )
            st.markdown(f"**{pages}** (score {section.score})\n\n{section.excerpt}")
        _downloads()
        return
    _badges(result.all_warnings())
    sections = [
        ("Prerequisites", result.prerequisites),
        ("Practical steps", result.steps),
        ("Relevant parts", result.relevant_parts),
        ("Safe to skip", result.skip),
        ("Gotchas", result.gotchas),
    ]
    tabs = st.tabs([title for title, _layer in sections])
    for tab, (title, layer) in zip(tabs, sections, strict=True):
        del title
        with tab:
            _badges(layer.warnings)
            st.markdown(layer.text or "Not covered in the document.")
    _downloads()


def _metrics(pages: int, words: int, reading_minutes: float, summary_minutes: float) -> None:
    import streamlit as st

    left, middle, right = st.columns(3)
    left.metric("Source", f"{pages} pages / {words} words")
    middle.metric("Read the source", f"{reading_minutes:.1f} min")
    right.metric("Read the summary", f"{summary_minutes:.1f} min")


def _badges(warnings: Sequence[WarningFlag]) -> None:
    import streamlit as st

    for warning in warnings:
        if isinstance(warning, WarningFlag):
            st.warning(f"{warning.code}: {warning.message}")


def _downloads() -> None:
    import streamlit as st

    markdown = st.session_state.get("result_md", "")
    payload = st.session_state.get("result_json", "")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("Download Markdown", data=markdown, file_name="tldr.md", mime="text/markdown")
    with col2:
        st.download_button("Download JSON", data=payload, file_name="tldr.json", mime="application/json")


if __name__ == "__main__":
    main()
