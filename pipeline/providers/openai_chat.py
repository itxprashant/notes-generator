"""OpenAI Chat Completions client (and OpenRouter via base_url override)."""

from __future__ import annotations

import base64
import logging
import os
import re
from pathlib import Path
from typing import Any

from .base import LLMClient, LLMResult

logger = logging.getLogger(__name__)


class _OpenAICompatClient(LLMClient):
    """Base for any OpenAI-compatible Chat Completions endpoint."""

    provider_name = "openai_compatible"
    api_key_env: str = "OPENAI_API_KEY"
    base_url: str | None = None

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
            import openai
        except ImportError as e:
            raise ImportError(
                "the openai SDK is required: pip install 'openai>=1.50'"
            ) from e

        self._openai = openai
        key = api_key or os.environ.get(self.api_key_env)
        if not key:
            raise RuntimeError(f"{self.api_key_env} is not set")

        client_kwargs: dict[str, Any] = {"api_key": key, "max_retries": 0}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        self._client = openai.OpenAI(**client_kwargs)

    # ------------------------------------------------------------------ blocks

    def text_block(self, text: str) -> dict[str, Any]:
        return {"type": "text", "text": text}

    def image_block(self, path: str | Path, fmt: str | None = None) -> dict[str, Any]:
        _canonical, mime = self._detect_image_mime(path, fmt)
        data = base64.standard_b64encode(Path(path).read_bytes()).decode("ascii")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{data}"},
        }

    # ------------------------------------------------------------------ caps

    def supports_thinking(self) -> bool:
        m = self.model.lower()
        # OpenAI reasoning families: o1*, o3*, o4*, gpt-5*. OpenRouter passes
        # `reasoning` through to many backing models; assume true and let the
        # backend ignore it if unsupported.
        if self.provider_name == "openrouter":
            return True
        return bool(re.match(r"^(o\d|gpt-5|gpt-4\.5)", m))

    def default_max_output_tokens(self) -> int:
        m = self.model.lower()
        if "gpt-5" in m or m.startswith(("o3", "o4")):
            return 32000
        if m.startswith("o1"):
            return 32000
        if "gpt-4.1" in m:
            return 32000
        return 8000

    # ------------------------------------------------------------------ infer

    def _build_kwargs(
        self,
        user_blocks: list[dict[str, Any]],
        system: str | None,
        max_tokens: int,
        use_thinking: bool,
    ) -> dict[str, Any]:
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user_blocks})

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        # gpt-5 / o-series want max_completion_tokens; older models use max_tokens.
        if re.match(r"^(o\d|gpt-5)", self.model.lower()):
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = max_tokens
            kwargs["temperature"] = 1.0 if use_thinking else 0.2

        if use_thinking:
            self._apply_thinking(kwargs)
        return kwargs

    def _apply_thinking(self, kwargs: dict[str, Any]) -> None:
        # OpenAI native: reasoning_effort
        kwargs["reasoning_effort"] = "medium"

    def converse(
        self,
        user_blocks: list[dict[str, Any]],
        system: str | None = None,
        max_tokens: int | None = None,
        thinking: bool = True,
    ) -> LLMResult:
        max_tokens = max_tokens or self.default_max_output_tokens()
        use_thinking = thinking and self.supports_thinking()
        kwargs = self._build_kwargs(user_blocks, system, max_tokens, use_thinking)

        retryable = (
            self._openai.RateLimitError,
            self._openai.APIConnectionError,
            self._openai.APITimeoutError,
            self._openai.InternalServerError,
        )
        response = self._retry(
            lambda: self._client.chat.completions.create(**kwargs),
            retryable_exc=retryable,
        )

        text = ""
        thinking_text = ""
        if response.choices:
            msg = response.choices[0].message
            text = (msg.content or "").strip()
            # Some providers (e.g. DeepSeek via OpenRouter) expose reasoning here.
            for attr in ("reasoning_content", "reasoning"):
                val = getattr(msg, attr, None)
                if isinstance(val, str) and val:
                    thinking_text = val
                    break

        usage = getattr(response, "usage", None)
        in_tok = int(getattr(usage, "prompt_tokens", 0) or 0)
        out_tok = int(getattr(usage, "completion_tokens", 0) or 0)
        total = int(getattr(usage, "total_tokens", in_tok + out_tok) or 0)
        return LLMResult(
            text=text,
            thinking=thinking_text,
            input_tokens=in_tok,
            output_tokens=out_tok,
            total_tokens=total,
            raw=response,
        )


class OpenAIClient(_OpenAICompatClient):
    provider_name = "openai"
    api_key_env = "OPENAI_API_KEY"
    base_url = None


class OpenRouterClient(_OpenAICompatClient):
    provider_name = "openrouter"
    api_key_env = "OPENROUTER_API_KEY"
    base_url = "https://openrouter.ai/api/v1"

    def _apply_thinking(self, kwargs: dict[str, Any]) -> None:
        # OpenRouter accepts a unified `reasoning` block via extra_body.
        kwargs.setdefault("extra_body", {})["reasoning"] = {
            "max_tokens": self.thinking_budget,
        }
