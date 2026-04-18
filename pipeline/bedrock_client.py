"""Thin wrapper around Bedrock Runtime `Converse` for Claude with extended thinking.

Handles:
- Building image content blocks from local PNG paths.
- Enabling extended thinking via `additionalModelRequestFields`.
- Retrying transient throttling / 5xx errors with exponential backoff.
- Returning concatenated text plus token usage.
"""

from __future__ import annotations

import logging
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


@dataclass
class ConverseResult:
    text: str
    thinking: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    raw: dict[str, Any] = field(default_factory=dict)


class BedrockClaudeClient:
    """Wrapper around `bedrock-runtime.converse` for a Claude model."""

    def __init__(
        self,
        model_id: str | None = None,
        region: str | None = None,
        thinking_budget_tokens: int | None = None,
        max_attempts: int = 5,
    ) -> None:
        self.model_id = model_id or os.environ.get(
            "BEDROCK_MODEL_ID",
            "us.meta.llama4-maverick-17b-instruct-v1:0",
        )
        self.region = region or os.environ.get("AWS_REGION", "us-east-1")
        self.thinking_budget = int(
            thinking_budget_tokens
            if thinking_budget_tokens is not None
            else os.environ.get("THINKING_BUDGET_TOKENS", "12000")
        )
        self.max_attempts = max_attempts

        self._client = boto3.client(
            "bedrock-runtime",
            region_name=self.region,
            config=Config(
                read_timeout=600,
                connect_timeout=30,
                retries={"max_attempts": 1, "mode": "standard"},
            ),
        )

    @staticmethod
    def image_block(path: str | Path, fmt: str | None = None) -> dict[str, Any]:
        """Build a Bedrock Converse image content block from a local file."""
        p = Path(path)
        if fmt is None:
            ext = p.suffix.lower().lstrip(".")
            fmt = {"jpg": "jpeg"}.get(ext, ext)
        if fmt not in {"png", "jpeg", "gif", "webp"}:
            raise ValueError(f"unsupported image format: {fmt}")
        return {
            "image": {
                "format": fmt,
                "source": {"bytes": p.read_bytes()},
            }
        }

    @staticmethod
    def text_block(text: str) -> dict[str, Any]:
        return {"text": text}

    def supports_thinking(self) -> bool:
        # Extended thinking via additionalModelRequestFields.thinking is
        # currently an Anthropic-Claude-specific feature on Bedrock Converse.
        mid = self.model_id.lower()
        return "anthropic" in mid and "claude" in mid

    def converse(
        self,
        user_blocks: list[dict[str, Any]],
        system: str | None = None,
        max_tokens: int = 16000,
        thinking: bool = True,
    ) -> ConverseResult:
        """Invoke `converse` with optional extended thinking."""
        # Silently downgrade thinking on models that don't support it.
        use_thinking = thinking and self.supports_thinking()
        kwargs: dict[str, Any] = {
            "modelId": self.model_id,
            "messages": [{"role": "user", "content": user_blocks}],
            "inferenceConfig": {
                "maxTokens": max_tokens,
                # Anthropic requires temperature=1 when thinking is enabled.
                "temperature": 1.0 if use_thinking else 0.2,
            },
        }
        if system:
            kwargs["system"] = [{"text": system}]
        if use_thinking:
            kwargs["additionalModelRequestFields"] = {
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": self.thinking_budget,
                }
            }

        response = self._invoke_with_retries(kwargs)

        text_parts: list[str] = []
        thinking_parts: list[str] = []
        for block in response.get("output", {}).get("message", {}).get("content", []):
            if "text" in block:
                text_parts.append(block["text"])
            elif "reasoningContent" in block:
                rc = block["reasoningContent"]
                rt = rc.get("reasoningText") or {}
                if "text" in rt:
                    thinking_parts.append(rt["text"])

        usage = response.get("usage", {}) or {}
        return ConverseResult(
            text="".join(text_parts).strip(),
            thinking="\n".join(thinking_parts).strip(),
            input_tokens=int(usage.get("inputTokens", 0)),
            output_tokens=int(usage.get("outputTokens", 0)),
            total_tokens=int(usage.get("totalTokens", 0)),
            raw=response,
        )

    def _invoke_with_retries(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        retryable = {
            "ThrottlingException",
            "ServiceUnavailableException",
            "ModelTimeoutException",
            "InternalServerException",
            "ModelStreamErrorException",
        }
        attempt = 0
        while True:
            attempt += 1
            try:
                return self._client.converse(**kwargs)
            except ClientError as e:
                code = e.response.get("Error", {}).get("Code", "")
                if code not in retryable or attempt >= self.max_attempts:
                    raise
                delay = min(60.0, (2 ** attempt) + random.uniform(0, 1))
                logger.warning(
                    "Bedrock %s on attempt %d/%d, retrying in %.1fs",
                    code,
                    attempt,
                    self.max_attempts,
                    delay,
                )
                time.sleep(delay)
