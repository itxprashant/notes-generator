"""Multi-provider LLM client factory.

Provider auto-detection priority:
1. Explicit `provider=` argument or `--provider` CLI flag
2. `LLM_PROVIDER` environment variable
3. Inferred from a slash-prefixed model id (`gemini/...`, `anthropic/...`,
   `openai/...`, `openrouter/...`, `bedrock/...`)
4. Inferred from Bedrock-style ids (containing `us.` or `global.` and a known
   provider segment) so existing `BEDROCK_MODEL_ID` values keep working
5. Default: `gemini`
"""

from __future__ import annotations

import logging
import os

from .base import LLMClient, LLMResult

logger = logging.getLogger(__name__)


KNOWN_PROVIDERS = ("gemini", "anthropic", "openai", "openrouter", "bedrock")

DEFAULT_MODELS: dict[str, str] = {
    "gemini": "gemini-2.5-pro",
    "anthropic": "claude-sonnet-4-5-20250929",
    "openai": "gpt-5",
    "openrouter": "google/gemini-2.5-pro",
    "bedrock": "us.meta.llama4-maverick-17b-instruct-v1:0",
}


def _strip_provider_prefix(model: str) -> tuple[str | None, str]:
    """If model starts with `provider/`, return (provider, rest). Otherwise (None, model)."""
    if "/" not in model:
        return None, model
    head, _, rest = model.partition("/")
    head_l = head.lower()
    if head_l in KNOWN_PROVIDERS:
        return head_l, rest
    return None, model


def _infer_bedrock_model(model: str) -> bool:
    """Heuristic: Bedrock cross-region inference profiles look like
    `us.<provider>.<model>` or `global.<provider>.<model>` or
    `<provider>.<model>-<date>-v<N>`."""
    m = model.lower()
    if m.startswith(("us.", "global.")):
        return True
    if any(seg in m for seg in (".claude-", ".llama", ".nova-", ".pixtral", ".mistral-")):
        return True
    return False


def resolve(provider: str | None, model: str | None) -> tuple[str, str]:
    """Resolve (provider, model) from explicit args, env, or defaults."""
    # Back-compat: BEDROCK_MODEL_ID still honored.
    env_provider = os.environ.get("LLM_PROVIDER")
    env_model = os.environ.get("LLM_MODEL") or os.environ.get("BEDROCK_MODEL_ID")

    p = (provider or env_provider or "").strip().lower() or None
    m = model or env_model

    if m and not p:
        head, rest = _strip_provider_prefix(m)
        if head:
            p, m = head, rest
        elif _infer_bedrock_model(m):
            p = "bedrock"

    if not p:
        p = "gemini"

    if p not in KNOWN_PROVIDERS:
        raise ValueError(
            f"unknown provider {p!r}; expected one of {', '.join(KNOWN_PROVIDERS)}"
        )

    if not m:
        m = DEFAULT_MODELS[p]

    return p, m


def make_client(
    provider: str | None = None,
    model: str | None = None,
    **kwargs,
) -> LLMClient:
    p, m = resolve(provider, model)
    logger.info("LLM provider=%s model=%s", p, m)

    if p == "bedrock":
        from .bedrock import BedrockClient
        return BedrockClient(model=m, **kwargs)
    if p == "anthropic":
        from .anthropic import AnthropicClient
        return AnthropicClient(model=m, **kwargs)
    if p == "openai":
        from .openai_chat import OpenAIClient
        return OpenAIClient(model=m, **kwargs)
    if p == "openrouter":
        from .openai_chat import OpenRouterClient
        return OpenRouterClient(model=m, **kwargs)
    if p == "gemini":
        from .gemini import GeminiClient
        return GeminiClient(model=m, **kwargs)
    raise ValueError(f"no client implementation for provider {p}")


__all__ = ["LLMClient", "LLMResult", "make_client", "resolve", "KNOWN_PROVIDERS"]
