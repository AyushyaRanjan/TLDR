.PHONY: install test lint format typecheck model ui eval

install:
	python -m pip install -e ".[dev]"

test:
	python -m pytest

lint:
	python -m ruff check .
	python -m ruff format --check .

format:
	python -m ruff check --fix .
	python -m ruff format .

typecheck:
	python -m mypy

model:
	ollama create tldr-qwen -f ollama/Modelfile

ui:
	tldr ui

eval:
	python evals/run_eval.py
