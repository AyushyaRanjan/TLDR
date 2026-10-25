# Contributing

Thanks for looking at tldr-everything. The useful changes are the ones that
keep a small local model honest: shorter prompts, tighter checks, clearer
errors.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

On Windows, activate with `.venv\Scripts\activate`.

## Checks

```bash
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m mypy
```

`make test`, `make lint`, and `make typecheck` run the same commands when you
have Make. CI runs them on Python 3.10, 3.11, and 3.12.

Tests use a fake LLM client. Do not add a test that calls the network.
`pytest -m slow -o addopts=` runs the optional live eval and is skipped in CI.
It needs a local Ollama server.

## Design rules

- One model call, one job. Put the prompt in `prompts.py` with a docstring
  that states the output format and the token cap.
- Do not log document text.
- Prefer a plain-text format and a small parser over a JSON schema.
- If you add an environment variable, document it in `.env.example` and the
  README table.

## Pull requests

Use the PR template. A few focused commits are easier to review than one
mixed commit. Conventional prefixes (`feat`, `fix`, `docs`, `test`, `chore`)
match the history of this repo.
