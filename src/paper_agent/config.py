"""Runtime configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    def load_dotenv(*args, **kwargs):
        return False


@dataclass(frozen=True)
class LLMConfig:
    api_key: str
    base_url: str
    model: str
    timeout_seconds: float = 120.0
    connect_timeout_seconds: float = 10.0
    max_retries: int = 3
    retry_base_seconds: float = 1.5
    max_tokens: int = 4096
    temperature: float = 0.3

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.model)


def load_llm_config() -> LLMConfig:
    """Load LLM settings from environment or local .env."""

    load_dotenv()
    if os.getenv("PAPER_AGENT_DISABLE_LLM", "").strip().lower() in {"1", "true", "yes", "on"}:
        return LLMConfig(
            api_key="",
            base_url=_default_base_url(),
            model=_default_text_model(),
        )

    api_key = (
        os.getenv("DEEPSEEK_API_KEY", "").strip()
        or os.getenv("OPENAI_API_KEY", "").strip()
        or os.getenv("ARK_API_KEY", "").strip()
    )
    return LLMConfig(
        api_key=api_key,
        base_url=_default_base_url(),
        model=_default_text_model(),
        timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", "120")),
        connect_timeout_seconds=float(os.getenv("LLM_CONNECT_TIMEOUT_SECONDS", "10")),
        max_retries=int(os.getenv("LLM_MAX_RETRIES", "3")),
        retry_base_seconds=float(os.getenv("LLM_RETRY_BASE_SECONDS", "1.5")),
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", "4096")),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.3")),
    )


def _default_base_url() -> str:
    return (
        os.getenv("DEEPSEEK_API_BASE", "").strip()
        or os.getenv("OPENAI_API_BASE", "").strip()
        or "https://api.deepseek.com"
    )


def _default_text_model() -> str:
    return os.getenv("TEXT_MODEL", "deepseek-v4-pro").strip()
