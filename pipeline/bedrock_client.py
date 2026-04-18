"""Back-compat shim. Use `pipeline.providers` instead."""

from .providers import LLMResult as ConverseResult
from .providers.bedrock import BedrockClient as BedrockClaudeClient

__all__ = ["BedrockClaudeClient", "ConverseResult"]
