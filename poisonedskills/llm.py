from __future__ import annotations

import os
import time
from dataclasses import dataclass
import logging
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class OpenAICompatibleLLM:
    model: str = "gpt-4o-mini"
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.4
    max_tokens: int = 4096
    timeout_sec: float = 60
    max_retries: int = 2

    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = self.base_url or os.getenv("OPENAI_BASE_URL") or None
        self.model = self.model or os.getenv("LLM_MODEL", "gpt-4o-mini")
        self._client: Any = None

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        if not self.available:
            raise RuntimeError("LLM client is unavailable because OPENAI_API_KEY is not set")
        attempt = 0
        while True:
            try:
                response = self._get_client().chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature if temperature is None else temperature,
                    max_tokens=self.max_tokens if max_tokens is None else max_tokens,
                    timeout=self.timeout_sec,
                )
                content = response.choices[0].message.content or ""
                return content
            except Exception as exc:
                attempt += 1
                if attempt > self.max_retries or not _retryable(exc):
                    raise
                sleep_s = 1.5 * (2 ** (attempt - 1))
                logger.warning("LLM retry %s/%s after %s", attempt, self.max_retries, str(exc)[:160])
                time.sleep(sleep_s)

    def _get_client(self) -> Any:
        if self._client is None:
            import openai

            kwargs: dict[str, Any] = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = openai.OpenAI(**kwargs)
        return self._client


def build_llm(config: dict[str, Any] | None) -> OpenAICompatibleLLM:
    cfg = config or {}
    return OpenAICompatibleLLM(
        model=cfg.get("model") or os.getenv("LLM_MODEL", "gpt-4o-mini"),
        api_key=cfg.get("api_key"),
        base_url=cfg.get("base_url"),
        temperature=float(cfg.get("temperature", 0.4)),
        max_tokens=int(cfg.get("max_tokens", 4096)),
        timeout_sec=float(cfg.get("timeout_sec", 60)),
        max_retries=int(cfg.get("max_retries", 2)),
    )


def _retryable(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(x in text for x in ("timeout", "rate limit", "502", "503", "504", "temporarily"))
