"""OpenAI-compatible provider — works with DeepSeek, Ollama, vLLM, etc."""

from __future__ import annotations

from typing import Any, AsyncIterator

from openai import AsyncOpenAI

from synapse.models.base import BaseProvider, ChatResponse, StreamChunk


class OpenAICompatProvider(BaseProvider):
    """Provider for any OpenAI-compatible API (DeepSeek, Ollama, vLLM, Groq, etc.)."""

    provider_name = "compat"

    def _get_client(self) -> AsyncOpenAI:
        return AsyncOpenAI(
            api_key=self.api_key or "not-needed",
            base_url=self.base_url,
        )

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: list[dict] | None = None,
    ) -> ChatResponse:
        client = self._get_client()

        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools

        response = await client.chat.completions.create(**kwargs)

        choice = response.choices[0]
        return ChatResponse(
            content=choice.message.content or "",
            model=response.model,
            finish_reason=choice.finish_reason or "stop",
            usage={
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            },
            raw=response,
        )

    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AsyncIterator[StreamChunk]:
        client = self._get_client()

        stream = await client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            content = delta.content or ""
            # DeepSeek R1 / reasoning models expose reasoning_content
            reasoning = getattr(delta, "reasoning_content", None) or ""
            if content or reasoning:
                yield StreamChunk(content=content, reasoning=reasoning)
