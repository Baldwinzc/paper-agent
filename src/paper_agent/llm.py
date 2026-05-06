"""OpenAI-compatible LLM client with sync and async APIs."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

import httpx

from paper_agent.config import LLMConfig


class LLMError(RuntimeError):
    """Base class for LLM call failures."""


class LLMConfigurationError(LLMError):
    """Raised when the LLM client is not configured."""


class LLMTimeoutError(LLMError):
    """Raised when a request times out after retries."""


class LLMHTTPError(LLMError):
    """Raised for non-retryable HTTP errors or exhausted HTTP retries."""


class LLMResponseError(LLMError):
    """Raised when the response shape is not a valid chat completion."""


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


@dataclass(frozen=True)
class LLMResult:
    content: str
    model: str
    usage: dict[str, Any]
    raw: dict[str, Any]


class LLMClient:
    """Small OpenAI-compatible chat completions client.

    The client is provider-neutral and works with DashScope/Qwen through:
    https://dashscope.aliyuncs.com/compatible-mode/v1
    """

    RETRY_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.endpoint = self._build_endpoint(config.base_url)

    @property
    def available(self) -> bool:
        return self.config.configured

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> LLMResult:
        """Call chat completions synchronously."""

        self._ensure_configured()
        payload = self._payload(messages, temperature, max_tokens, response_format)
        timeout = self._timeout()
        last_error: Exception | None = None
        with httpx.Client(timeout=timeout) as client:
            for attempt in range(self.config.max_retries + 1):
                try:
                    response = client.post(self.endpoint, headers=self._headers(), json=payload)
                    if self._should_retry(response.status_code) and attempt < self.config.max_retries:
                        self._sleep(attempt)
                        continue
                    if response.status_code >= 400:
                        raise LLMHTTPError(self._http_error_message(response))
                    return self._parse(response.json())
                except httpx.TimeoutException as exc:
                    last_error = exc
                    if attempt >= self.config.max_retries:
                        raise LLMTimeoutError("LLM request timed out after retries.") from exc
                    self._sleep(attempt)
                except httpx.TransportError as exc:
                    last_error = exc
                    if attempt >= self.config.max_retries:
                        raise LLMHTTPError(f"LLM transport error after retries: {exc}") from exc
                    self._sleep(attempt)
        raise LLMHTTPError(f"LLM request failed: {last_error}")

    async def achat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> LLMResult:
        """Call chat completions asynchronously."""

        self._ensure_configured()
        payload = self._payload(messages, temperature, max_tokens, response_format)
        timeout = self._timeout()
        last_error: Exception | None = None
        async with httpx.AsyncClient(timeout=timeout) as client:
            for attempt in range(self.config.max_retries + 1):
                try:
                    response = await client.post(self.endpoint, headers=self._headers(), json=payload)
                    if self._should_retry(response.status_code) and attempt < self.config.max_retries:
                        await self._asleep(attempt)
                        continue
                    if response.status_code >= 400:
                        raise LLMHTTPError(self._http_error_message(response))
                    return self._parse(response.json())
                except httpx.TimeoutException as exc:
                    last_error = exc
                    if attempt >= self.config.max_retries:
                        raise LLMTimeoutError("LLM request timed out after retries.") from exc
                    await self._asleep(attempt)
                except httpx.TransportError as exc:
                    last_error = exc
                    if attempt >= self.config.max_retries:
                        raise LLMHTTPError(f"LLM transport error after retries: {exc}") from exc
                    await self._asleep(attempt)
        raise LLMHTTPError(f"LLM request failed: {last_error}")

    def _ensure_configured(self) -> None:
        if not self.available:
            raise LLMConfigurationError("OPENAI_API_KEY and TEXT_MODEL must be configured.")

    def _payload(
        self,
        messages: list[ChatMessage],
        temperature: float | None,
        max_tokens: int | None,
        response_format: dict[str, Any] | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [message.__dict__ for message in messages],
            "temperature": self.config.temperature if temperature is None else temperature,
            "max_tokens": self.config.max_tokens if max_tokens is None else max_tokens,
        }
        if response_format:
            payload["response_format"] = response_format
        return payload

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

    def _timeout(self) -> httpx.Timeout:
        return httpx.Timeout(
            timeout=self.config.timeout_seconds,
            connect=self.config.connect_timeout_seconds,
        )

    def _should_retry(self, status_code: int) -> bool:
        return status_code in self.RETRY_STATUS_CODES

    def _sleep(self, attempt: int) -> None:
        time.sleep(self.config.retry_base_seconds * (2**attempt))

    async def _asleep(self, attempt: int) -> None:
        await asyncio.sleep(self.config.retry_base_seconds * (2**attempt))

    def _parse(self, data: dict[str, Any]) -> LLMResult:
        try:
            choice = data["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMResponseError("LLM response did not contain choices[0].message.content.") from exc
        if not isinstance(content, str) or not content.strip():
            raise LLMResponseError("LLM response content is empty.")
        return LLMResult(
            content=content,
            model=str(data.get("model", self.config.model)),
            usage=data.get("usage", {}) if isinstance(data.get("usage", {}), dict) else {},
            raw=data,
        )

    def _http_error_message(self, response: httpx.Response) -> str:
        body = response.text[:500]
        return f"LLM HTTP {response.status_code}: {body}"

    def _build_endpoint(self, base_url: str) -> str:
        base = base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        return f"{base}/chat/completions"

