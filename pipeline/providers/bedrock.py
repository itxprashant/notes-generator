"""Amazon Bedrock client (Converse API)."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from .base import LLMClient, LLMResult

logger = logging.getLogger(__name__)


class BedrockClient(LLMClient):
    provider_name = "bedrock"

    def __init__(
        self,
        model: str,
        region: str | None = None,
        thinking_budget_tokens: int | None = None,
        max_attempts: int = 5,
    ) -> None:
        super().__init__(
            model=model,
            thinking_budget_tokens=thinking_budget_tokens,
            max_attempts=max_attempts,
        )
        self.region = region or os.environ.get("AWS_REGION", "us-east-1")
        self._client = boto3.client(
            "bedrock-runtime",
            region_name=self.region,
            config=Config(
                read_timeout=600,
                connect_timeout=30,
                retries={"max_attempts": 1, "mode": "standard"},
            ),
        )

    def text_block(self, text: str) -> dict[str, Any]:
        return {"text": text}

    def image_block(self, path: str | Path, fmt: str | None = None) -> dict[str, Any]:
        canonical, _mime = self._detect_image_mime(path, fmt)
        return {
            "image": {
                "format": canonical,
                "source": {"bytes": Path(path).read_bytes()},
            }
        }

    def supports_thinking(self) -> bool:
        # Extended thinking via additionalModelRequestFields.thinking is
        # currently an Anthropic-Claude-specific feature on Bedrock Converse.
        m = self.model.lower()
        return "anthropic" in m and "claude" in m

    def default_max_output_tokens(self) -> int:
        m = self.model.lower()
        if "anthropic" in m:
            return 16000
        if "amazon.nova" in m:
            return 9500
        if "meta.llama4" in m:
            return 8000
        if "mistral.pixtral" in m:
            return 8000
        return 4000

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
            "modelId": self.model,
            "messages": [{"role": "user", "content": user_blocks}],
            "inferenceConfig": {
                "maxTokens": max_tokens,
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
        return LLMResult(
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
        import random
        import time

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
                    code, attempt, self.max_attempts, delay,
                )
                time.sleep(delay)
