"""Abstract base class for all model providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator


@dataclass
class ChatResponse:
    """Unified chat completion response from any provider."""

    content: str
    model: str = ""
    finish_reason: str = "stop"
    usage: dict[str, int] = field(default_factory=dict)
    raw: Any = None

    @property
    def total_tokens(self) -> int:
        return self.usage.get("total_tokens", 0)


@dataclass
class StreamChunk:
    """A single streaming token from a provider.

    For models that support reasoning/thinking (DeepSeek R1, Claude
    extended thinking, Gemini 2.5 thinking), the `reasoning` field
    contains the model's internal thought process. Otherwise it's
    empty and `content` holds the normal output.
    """

    content: str = ""
    reasoning: str = ""


class BaseProvider(ABC):
    """Abstract base for all LLM providers.

    Every provider must implement:
    - chat()       → single-turn completion
    - chat_stream() → streaming completion (async generator of tokens)
    """

    provider_name: str = "base"
    model_name: str = ""
    api_key: str = ""
    base_url: str = ""

    def __init__(
        self,
        model_name: str,
        api_key: str = "",
        base_url: str = "",
        **kwargs: Any,
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.extra = kwargs

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: list[dict] | None = None,
    ) -> ChatResponse:
        """Send a chat completion request and return the full response."""

    @abstractmethod
    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AsyncIterator[StreamChunk]:
        """Stream chat completion tokens as an async iterator.

        Each item is a StreamChunk that may contain reasoning/thinking
        content in addition to the normal response text.
        """

    def count_tokens(self, text: str) -> int:
        """Estimate token count. Override for provider-specific counting."""
        # Rough estimate: ~4 chars per token for English
        return len(text) // 4

    def __repr__(self) -> str:
        return f"{self.provider_name}(model={self.model_name})"
