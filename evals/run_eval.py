"""Optional quality check against a live local model.

Not used in CI. Run it only when Ollama is up and the model is pulled:

    python evals/run_eval.py

Prints word counts, warning flags, and latency for two short samples plus
the Docker-on-a-VPS example. Exits with the usual LLM errors if the server
or the model is missing.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tldr_everything.config import load_settings  # noqa: E402
from tldr_everything.extract import load_file  # noqa: E402
from tldr_everything.llm import build_client  # noqa: E402
from tldr_everything.pipeline import Pipeline  # noqa: E402
from tldr_everything.postprocess import word_count  # noqa: E402

SAMPLES = (
    ROOT / "examples" / "docker-vps.md",
    ROOT / "evals" / "samples" / "reading-habit.md",
    ROOT / "evals" / "samples" / "release-notes.md",
)


def main() -> list[dict[str, object]]:
    """Run Standard mode on the sample documents and print a plain report."""
    settings = load_settings()
    pipeline = Pipeline(build_client(settings), settings)
    rows: list[dict[str, object]] = []
    print(f"model={settings.llm_model} base={settings.llm_base_url}")
    for path in SAMPLES:
        document = load_file(path, settings)
        started = time.perf_counter()
        result = pipeline.summarize(document)
        elapsed = time.perf_counter() - started
        summary = result.summary_30s.text if result.summary_30s else ""
        explain = result.explain_3m.text if result.explain_3m else ""
        flags = len(result.all_warnings())
        row: dict[str, object] = {
            "file": path.name,
            "seconds": round(elapsed, 1),
            "summary_words": word_count(summary),
            "explain_words": word_count(explain),
            "warnings": flags,
            "single_pass": result.single_pass,
        }
        rows.append(row)
        print(
            f"{path.name}: {row['seconds']}s  "
            f"30s={row['summary_words']} words  "
            f"3m={row['explain_words']} words  "
            f"warnings={row['warnings']}  "
            f"single_pass={row['single_pass']}"
        )
    return rows


if __name__ == "__main__":
    main()
