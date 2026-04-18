"""Anthropic Claude direct API client."""

from __future__ import annotations

import base64
import logging
import os
from pathlib import Path
from typing import Any

from .base import LLMClient, LLMResult

logger = logging.getLogger(__name__)


class AnthropicClient(LLMClient):
    provider_name = "anthropic"

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        thinking_budget_tokens: int | None = None,
        max_attempts: int = 5,
    ) -> None:
        super().__init__(
            model=model,
            thinking_budget_tokens=thinking_budget_tokens,
            max_attempts=max_attempts,
        )
        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "the anthropic SDK is required: pip install 'anthropic>=0.40'"
            ) from e

        self._anthropic = anthropic
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        self._client = anthropic.Anthropic(api_key=key, max_retries=0)

    def text_block(self, text: str) -> dict[str, Any]:
        return {"type": "text", "text": text}

    def image_block(self, path: str | Path, fmt: str | None = None) -> dict[str, Any]:
        _canonical, mime = self._detect_image_mime(path, fmt)
        data = base64.standard_b64encode(Path(path).read_bytes()).decode("ascii")
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": mime,
                "data": data,
            },
        }

    def supports_thinking(self) -> bool:
        # All current Claude 3.7 / 4.x models support extended thinking.
        return True

    def default_max_output_tokens(self) -> int:
        m = self.model.lower()
        if "opus" in m:
            return 32000
        if "sonnet" in m:
            return 16000
        if "haiku" in m:
            return 8000
        return 8000

    def converse(
        self,
        user_blocks: list[dict[str, Any]],
        system: str | None = None,
        max_tokens: int | None = None,
        thinking: bool = True,
    ) -> LLMResult:
        max_tokens = max_tokens or self.default_max_output_tokens()
        use_thinking = thinking and self.supports_thinking()

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": user_blocks}],
            "temperature": 1.0 if use_thinking else 0.2,
        }
        if system:
            kwargs["system"] = system
        if use_thinking:
            # Thinking budget must be < max_tokens.
            budget = min(self.thinking_budget, max(1024, max_tokens - 1024))
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}

        retryable = (
            self._anthropic.RateLimitError,
            self._anthropic.APIConnectionError,
            self._anthropic.InternalServerError,
            self._anthropic.APITimeoutError,
        )
        response = self._retry(
            lambda: self._client.messages.create(**kwargs),
            retryable_exc=retryable,
        )

        text_parts: list[str] = []
        thinking_parts: list[str] = []
        for block in response.content or []:
            btype = getattr(block, "type", None)
            if btype == "text":
                text_parts.append(getattr(block, "text", "") or "")
            elif btype == "thinking":
                thinking_parts.append(getattr(block, "thinking", "") or "")
            elif btype == "redacted_thinking":
                thinking_parts.append("[redacted thinking block]")

        usage = getattr(response, "usage", None)
        return LLMResult(
            text="".join(text_parts).strip(),
            thinking="\n".join(thinking_parts).strip(),
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            total_tokens=int(
                (getattr(usage, "input_tokens", 0) or 0)
                + (getattr(usage, "output_tokens", 0) or 0)
            ),
            raw=response,
        )
