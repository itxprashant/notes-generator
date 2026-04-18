"""Google Gemini client (google-genai SDK)."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from .base import LLMClient, LLMResult

logger = logging.getLogger(__name__)


class GeminiClient(LLMClient):
    provider_name = "gemini"

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
            from google import genai
            from google.genai import types as genai_types
            from google.genai import errors as genai_errors
        except ImportError as e:
            raise ImportError(
                "the google-genai SDK is required: pip install 'google-genai>=0.5'"
            ) from e

        self._genai = genai
        self._types = genai_types
        self._errors = genai_errors
        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY (or GOOGLE_API_KEY) is not set")
        self._client = genai.Client(api_key=key)

    def text_block(self, text: str) -> Any:
        return self._types.Part.from_text(text=text)

    def image_block(self, path: str | Path, fmt: str | None = None) -> Any:
        _canonical, mime = self._detect_image_mime(path, fmt)
        return self._types.Part.from_bytes(
            data=Path(path).read_bytes(),
            mime_type=mime,
        )

    def supports_thinking(self) -> bool:
        m = self.model.lower()
        # 2.5-pro / 2.5-flash / 2.5-flash-lite all support thinking_config.
        return "2.5" in m or "3." in m or "thinking" in m

    def default_max_output_tokens(self) -> int:
        m = self.model.lower()
        if "pro" in m:
            return 65536
        if "flash" in m:
            return 65536
        return 8192

    def converse(
        self,
        user_blocks: list[Any],
        system: str | None = None,
        max_tokens: int | None = None,
        thinking: bool = True,
    ) -> LLMResult:
        max_tokens = max_tokens or self.default_max_output_tokens()
        use_thinking = thinking and self.supports_thinking()

        config_kwargs: dict[str, Any] = {
            "max_output_tokens": max_tokens,
            "temperature": 1.0 if use_thinking else 0.2,
        }
        if system:
            config_kwargs["system_instruction"] = system
        if use_thinking:
            config_kwargs["thinking_config"] = self._types.ThinkingConfig(
                thinking_budget=self.thinking_budget,
                include_thoughts=True,
            )

        contents = [self._types.Content(role="user", parts=list(user_blocks))]
        config = self._types.GenerateContentConfig(**config_kwargs)

        retryable = (
            self._errors.ServerError,
            self._errors.APIError,
            ConnectionError,
            TimeoutError,
        )
        response = self._retry(
            lambda: self._client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            ),
            retryable_exc=retryable,
        )

        text_parts: list[str] = []
        thinking_parts: list[str] = []
        for cand in getattr(response, "candidates", None) or []:
            content = getattr(cand, "content", None)
            for part in getattr(content, "parts", None) or []:
                ptext = getattr(part, "text", None)
                if not ptext:
                    continue
                if getattr(part, "thought", False):
                    thinking_parts.append(ptext)
                else:
                    text_parts.append(ptext)

        # Fallback: response.text aggregates non-thought text on newer SDKs.
        if not text_parts:
            text_attr = getattr(response, "text", None)
            if text_attr:
                text_parts.append(text_attr)

        usage = getattr(response, "usage_metadata", None)
        in_tok = int(getattr(usage, "prompt_token_count", 0) or 0)
        out_tok = int(getattr(usage, "candidates_token_count", 0) or 0)
        thought_tok = int(getattr(usage, "thoughts_token_count", 0) or 0)
        total_tok = int(getattr(usage, "total_token_count", in_tok + out_tok + thought_tok) or 0)
        return LLMResult(
            text="".join(text_parts).strip(),
            thinking="\n".join(thinking_parts).strip(),
            input_tokens=in_tok,
            output_tokens=out_tok + thought_tok,
            total_tokens=total_tok,
            raw=response,
        )
