"""Error mapping, retries, streaming, and the Ollama context warning."""

from __future__ import annotations

import json

from tests.conftest import make_settings
from tldr_everything.llm import (
    LLMConnectionError,
    LLMContextError,
    LLMModelMissingError,
    OpenAICompatibleClient,
    collect_startup_warnings,
    endpoint_is_local,
    parse_num_ctx,
    probe_num_ctx,
)


class RateLimitError(Exception):
    pass


class NotFoundError(Exception):
    pass


class APIConnectionError(Exception):
    pass


class BadRequestError(Exception):
    pass


class _Message:
    def __init__(self, content: str) -> None:
        self.content = content


class _Choice:
    def __init__(self, content: str) -> None:
        self.message = _Message(content)


class _Response:
    def __init__(self, content: str) -> None:
        self.choices = [_Choice(content)]


class _Delta:
    def __init__(self, content: str) -> None:
        self.content = content


class _StreamChoice:
    def __init__(self, content: str) -> None:
        self.delta = _Delta(content)


class _StreamChunk:
    def __init__(self, content: str) -> None:
        self.choices = [_StreamChoice(content)]


class StubSDK:
    def __init__(self, effects: list[object]) -> None:
        self.effects = list(effects)
        self.calls = 0
        self.chat = self
        self.completions = self

    def create(self, **kwargs: object) -> object:
        del kwargs
        self.calls += 1
        effect = self.effects.pop(0)
        if isinstance(effect, Exception):
            raise effect
        return effect


def test_retry_then_success() -> None:
    delays: list[float] = []
    stub = StubSDK([RateLimitError("slow down"), RateLimitError("slow down"), _Response("hello")])
    client = OpenAICompatibleClient(make_settings(), sdk_client=stub, sleeper=delays.append)
    assert client.complete(system="s", user="u", max_tokens=16, call_type="goal_relevance") == "hello"
    assert delays == [0.5, 1.0]
    assert stub.calls == 3


def test_missing_model_is_not_retried() -> None:
    delays: list[float] = []
    stub = StubSDK([NotFoundError("model qwen is not found")])
    client = OpenAICompatibleClient(make_settings(), sdk_client=stub, sleeper=delays.append)
    try:
        client.complete(system="s", user="u", max_tokens=16, call_type="chunk_summary")
    except LLMModelMissingError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected a missing-model error")
    assert delays == []
    assert "ollama pull qwen2.5:3b" in message
    assert "make model" in message


def test_connection_and_context_errors() -> None:
    settings = make_settings()
    connection = OpenAICompatibleClient(
        settings,
        sdk_client=StubSDK([APIConnectionError("connection error")] * 3),
        sleeper=lambda _seconds: None,
    )
    try:
        connection.complete(system="s", user="u", max_tokens=8, call_type="chunk_summary")
    except LLMConnectionError as exc:
        assert "LLM_BASE_URL" in str(exc)
        assert "ollama serve" in str(exc)
    else:
        raise AssertionError("expected a connection error")

    context = OpenAICompatibleClient(
        settings,
        sdk_client=StubSDK([BadRequestError("maximum context length exceeded")]),
        sleeper=lambda _seconds: None,
    )
    try:
        context.complete(system="s", user="u", max_tokens=8, call_type="chunk_summary")
    except LLMContextError as exc:
        assert "make model" in str(exc)
    else:
        raise AssertionError("expected a context error")


def test_stream_joins_deltas() -> None:
    stub = StubSDK([[_StreamChunk("hel"), _StreamChunk("lo")]])
    client = OpenAICompatibleClient(make_settings(), sdk_client=stub, sleeper=lambda _seconds: None)
    assert "".join(client.stream(system="s", user="u", max_tokens=8, call_type="summary_30s")) == "hello"


def test_parse_num_ctx_and_warnings() -> None:
    assert parse_num_ctx({"parameters": "temperature 0.2\nnum_ctx 2048\n"}) == 2048
    assert parse_num_ctx({"model_info": {"qwen.context_length": 8192}}) == 8192
    assert parse_num_ctx({}) is None

    small = collect_startup_warnings(make_settings(), probe=lambda _settings: 2048)
    assert any("tldr-qwen" in warning for warning in small)
    assert any("num_ctx=2048" in warning for warning in small)

    quiet = collect_startup_warnings(
        make_settings(llm_model="tldr-qwen"),
        probe=lambda _settings: 8192,
    )
    assert quiet == []

    remote = collect_startup_warnings(
        make_settings(llm_base_url="https://openrouter.ai/api/v1", llm_model="qwen/qwen-2.5-7b-instruct"),
        probe=lambda _settings: None,
    )
    assert any("not local" in warning for warning in remote)
    assert endpoint_is_local("http://localhost:11434/v1")
    assert endpoint_is_local("http://ollama:11434/v1")
    assert not endpoint_is_local("https://openrouter.ai/api/v1")


def test_probe_num_ctx_reads_a_fake_response(monkeypatch: object) -> None:
    class _Response:
        def read(self, _limit: int) -> bytes:
            return json.dumps({"parameters": "num_ctx 4096"}).encode("utf-8")

        def __enter__(self) -> _Response:
            return self

        def __exit__(self, *_args: object) -> bool:
            return False

    def _open(*_args: object, **_kwargs: object) -> _Response:
        return _Response()

    monkeypatch.setattr("tldr_everything.llm.urllib.request.urlopen", _open)  # type: ignore[attr-defined]
    assert probe_num_ctx(make_settings()) == 4096
    assert probe_num_ctx(make_settings(llm_base_url="https://example.com/v1")) is None
