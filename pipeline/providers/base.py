"""Provider-agnostic LLM client base.

Each concrete subclass owns its provider's SDK quirks (content block format,
extended-thinking / reasoning configuration, usage parsing). The orchestrator
treats `text_block(...)` and `image_block(...)` return values as opaque blobs
and only inspects the normalized `LLMResult` returned by `converse(...)`.
"""

from __future__ import annotations

import logging
import os
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class LLMResult:
    text: str
    thinking: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    raw: Any = field(default=None)


class LLMClient(ABC):
    """Abstract LLM client. Provider-specific subclasses implement the methods."""

    provider_name: str = "base"

    def __init__(
        self,
        model: str,
        thinking_budget_tokens: int | None = None,
        max_attempts: int = 5,
    ) -> None:
        self.model = model
        self.thinking_budget = int(
            thinking_budget_tokens
            if thinking_budget_tokens is not None
            else os.environ.get("THINKING_BUDGET_TOKENS", "12000")
        )
        self.max_attempts = max_attempts

    # ------------------------------------------------------------------ blocks

    @abstractmethod
    def text_block(self, text: str) -> Any: ...

    @abstractmethod
    def image_block(self, path: str | Path, fmt: str | None = None) -> Any: ...

    # ------------------------------------------------------------------ infer

    @abstractmethod
    def converse(
        self,
        user_blocks: list[Any],
        system: str | None = None,
        max_tokens: int | None = None,
        thinking: bool = True,
    ) -> LLMResult: ...

    # ------------------------------------------------------------------ caps

    def supports_thinking(self) -> bool:
        return True

    def default_max_output_tokens(self) -> int:
        return 8000

    # ------------------------------------------------------------------ utils

    def _retry(self, fn, *, retryable_exc: tuple[type[BaseException], ...] = ()):
        """Run `fn()` with exponential backoff for any retryable exception."""
        attempt = 0
        while True:
            attempt += 1
            try:
                return fn()
            except retryable_exc as e:
                if attempt >= self.max_attempts:
                    raise
                delay = min(60.0, (2 ** attempt) + random.uniform(0, 1))
                logger.warning(
                    "%s attempt %d/%d failed (%s); retrying in %.1fs",
                    self.provider_name, attempt, self.max_attempts,
                    type(e).__name__, delay,
                )
                time.sleep(delay)

    @staticmethod
    def _detect_image_mime(path: str | Path, fmt: str | None) -> tuple[str, str]:
        """Return (canonical_format, mime_type) for an image path."""
        p = Path(path)
        ext = (fmt or p.suffix.lower().lstrip(".")).lower()
        ext = {"jpg": "jpeg"}.get(ext, ext)
        if ext not in {"png", "jpeg", "gif", "webp"}:
            raise ValueError(f"unsupported image format: {ext}")
        return ext, f"image/{ext}"
