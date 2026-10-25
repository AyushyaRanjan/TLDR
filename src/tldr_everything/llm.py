"""OpenAI-compatible chat client with retries, streaming, and friendly errors."""

from __future__ import annotations

import json
import logging
import re
import time
import urllib.request
from collections.abc import Callable, Iterator
from typing import Any, Protocol
from urllib.parse import urlparse

from openai import OpenAI

from tldr_everything.config import Settings

logger = logging.getLogger(__name__)

_RETRYABLE = {"RateLimitError", "APITimeoutError", "InternalServerError", "APIConnectionError"}
_LOCAL_HOSTS = {
    "localhost",
    "127.0.0.1",
    "0.0.0.0",
    "ollama",
    "host.docker.internal",
}


class LLMError(Exception):
    """Base class for errors the CLI and UI can show directly."""


class LLMConnectionError(LLMError):
    """The configured server could not be reached."""


class LLMModelMissingError(LLMError):
    """The configured model is not pulled or not loaded."""


class LLMContextError(LLMError):
    """The prompt was larger than the server's context window."""


class LLMClient(Protocol):
    """The slice of the client the pipeline needs. Tests supply a fake."""

    def complete(self, *, system: str, user: str, max_tokens: int, call_type: str) -> str:
        """Return the full assistant message."""

    def stream(self, *, system: str, user: str, max_tokens: int, call_type: str) -> Iterator[str]:
        """Yield assistant text as it arrives."""


class OpenAICompatibleClient:
    """Chat client for Ollama, LM Studio, vLLM, or OpenRouter.

    Document text is sent to the model and is never written to the log.
    """

    def __init__(
        self,
        settings: Settings,
        sdk_client: object | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.settings = settings
        self._sleeper = sleeper
        self._sdk: Any
        if sdk_client is not None:
            self._sdk = sdk_client
        else:
            self._sdk = OpenAI(
                base_url=settings.llm_base_url,
                api_key=settings.llm_api_key,
                max_retries=0,
                timeout=120.0,
            )

    def complete(self, *, system: str, user: str, max_tokens: int, call_type: str) -> str:
        response = self._request(system=system, user=user, max_tokens=max_tokens, stream=False)
        choices = getattr(response, "choices", None) or []
        if not choices:
            return ""
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        return content if isinstance(content, str) else ""

    def stream(self, *, system: str, user: str, max_tokens: int, call_type: str) -> Iterator[str]:
        response = self._request(system=system, user=user, max_tokens=max_tokens, stream=True)
        for chunk in response:
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            content = getattr(delta, "content", None)
            if isinstance(content, str) and content:
                yield content

    def _request(self, *, system: str, user: str, max_tokens: int, stream: bool) -> Any:
        delay = 0.5
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                return self._once(system=system, user=user, max_tokens=max_tokens, stream=stream)
            except Exception as exc:
                mapped = translate_error(exc, self.settings)
                last_error = mapped
                if attempt == 2 or not is_retryable(exc):
                    raise mapped from exc
                logger.info(
                    "llm retry attempt=%s type=%s error=%s",
                    attempt + 1,
                    type(exc).__name__,
                    type(mapped).__name__,
                )
                self._sleeper(delay)
                delay *= 2
        raise last_error or LLMError("The model call failed.")

    def _once(self, *, system: str, user: str, max_tokens: int, stream: bool) -> Any:
        kwargs: dict[str, object] = {
            "model": self.settings.llm_model,
            "temperature": self.settings.llm_temperature,
            "top_p": self.settings.llm_top_p,
            "max_tokens": max_tokens,
            "stream": stream,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if _looks_like_ollama(self.settings.llm_base_url):
            # Hint only. Ollama often ignores this; the Modelfile is the real fix.
            kwargs["extra_body"] = {"options": {"num_ctx": self.settings.llm_max_context_tokens}}
        logger.info("llm request max_tokens=%s stream=%s", max_tokens, stream)
        return self._sdk.chat.completions.create(**kwargs)


def build_client(settings: Settings) -> OpenAICompatibleClient:
    """Construct the default client from settings."""
    return OpenAICompatibleClient(settings)


def is_retryable(exc: Exception) -> bool:
    """Retry rate limits, timeouts, and transient server or connection errors."""
    name = type(exc).__name__
    if name in {"NotFoundError", "BadRequestError", "AuthenticationError"}:
        return False
    if "context" in str(exc).lower():
        return False
    return name in _RETRYABLE


def translate_error(exc: Exception, settings: Settings) -> LLMError:
    """Map SDK exceptions to messages that say what to do next."""
    if isinstance(exc, LLMError):
        return exc
    name = type(exc).__name__
    text = str(exc).lower()
    if name in {"APIConnectionError", "ConnectError"} or "connection error" in text:
        return LLMConnectionError(
            "Cannot reach the LLM server. Start Ollama (`ollama serve`) or check "
            f"LLM_BASE_URL ({settings.llm_base_url})."
        )
    model_missing = name == "NotFoundError" or ("model" in text and "not found" in text)
    model_missing = model_missing or ("model" in text and "does not exist" in text)
    if model_missing:
        return LLMModelMissingError(
            f"Model `{settings.llm_model}` is not available. "
            "Run `ollama pull qwen2.5:3b` and `make model`, then set LLM_MODEL=tldr-qwen. "
            "If you skipped the Modelfile, set LLM_MODEL=qwen2.5:3b."
        )
    if "context" in text or "num_ctx" in text or "maximum context" in text:
        return LLMContextError(
            "The prompt exceeded the model's context window. "
            "Build the derived model with `make model` (num_ctx 8192) and set "
            "LLM_MODEL=tldr-qwen, or start Ollama with OLLAMA_CONTEXT_LENGTH=8192. "
            "You can also lower CHUNK_TOKENS."
        )
    return LLMError(f"The model call failed ({name}). Check the server log for details.")


def endpoint_is_local(url: str) -> bool:
    """Return whether the base URL points at this machine or a local compose service."""
    host = (urlparse(url).hostname or "").lower()
    return host in _LOCAL_HOSTS or host.endswith(".local")


def _looks_like_ollama(url: str) -> bool:
    return "11434" in url or (urlparse(url).hostname or "").lower() == "ollama"


def parse_num_ctx(payload: dict[str, object]) -> int | None:
    """Read a context length from an Ollama ``/api/show`` payload."""
    parameters = payload.get("parameters")
    if isinstance(parameters, str):
        match = re.search(r"num_ctx\s+(\d+)", parameters)
        if match:
            return int(match.group(1))
    info = payload.get("model_info")
    if isinstance(info, dict):
        for key, value in info.items():
            if str(key).endswith("context_length") and isinstance(value, int):
                return value
    return None


def probe_num_ctx(settings: Settings) -> int | None:
    """Ask a local Ollama server for the model's context length.

    Failures are ignored. The probe sends the model name only, never document text.
    """
    if not _looks_like_ollama(settings.llm_base_url):
        return None
    root = settings.llm_base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    request = urllib.request.Request(
        root + "/api/show",
        data=json.dumps({"name": settings.llm_model}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=0.4) as response:
            raw = response.read(100_000)
    except Exception:
        logger.info("ollama context probe did not answer")
        return None
    try:
        payload = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return parse_num_ctx(payload)


def collect_startup_warnings(
    settings: Settings,
    probe: Callable[[Settings], int | None] | None = None,
) -> list[str]:
    """Warnings shown before a run. They do not stop the run."""
    warnings: list[str] = []
    if not endpoint_is_local(settings.llm_base_url):
        warnings.append("LLM_BASE_URL is not local. Document text will be sent to that server.")
    if _looks_like_ollama(settings.llm_base_url) and settings.llm_model != "tldr-qwen":
        warnings.append(
            "The model is not tldr-qwen. Ollama's OpenAI endpoint often ignores a "
            "per-request context size and silently truncates prompts. Run `make model` "
            "and set LLM_MODEL=tldr-qwen, or start Ollama with OLLAMA_CONTEXT_LENGTH=8192."
        )
    reader = probe if probe is not None else probe_num_ctx
    num_ctx: int | None
    try:
        num_ctx = reader(settings)
    except Exception:
        num_ctx = None
    if num_ctx is not None and num_ctx < settings.llm_max_context_tokens:
        warnings.append(
            f"Ollama reports num_ctx={num_ctx}, below "
            f"LLM_MAX_CONTEXT_TOKENS={settings.llm_max_context_tokens}. "
            "Prompts may be truncated. Run `make model` or set OLLAMA_CONTEXT_LENGTH=8192."
        )
    return warnings
