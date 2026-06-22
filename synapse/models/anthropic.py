"""Anthropic (Claude) provider."""

from __future__ import annotations

from typing import AsyncIterator

from anthropic import AsyncAnthropic

from synapse.models.base import BaseProvider, ChatResponse


class AnthropicProvider(BaseProvider):
    """Provider for Anthropic Claude models."""

    provider_name = "anthropic"

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: list[dict] | None = None,
    ) -> ChatResponse:
        client = AsyncAnthropic(api_key=self.api_key)

        # Extract system message if present
        system_msg = ""
        api_messages = []
        for m in messages:
            if m["role"] == "system":
                system_msg = m["content"]
            else:
                api_messages.append(m)

        kwargs = {
            "model": self.model_name,
            "messages": api_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if system_msg:
            kwargs["system"] = system_msg

        response = await client.messages.create(**kwargs)

        # Anthropic returns content as a list of blocks
        content = ""
        for block in response.content:
            if block.type == "text":
                content += block.text

        return ChatResponse(
            content=content,
            model=response.model,
            finish_reason=response.stop_reason or "stop",
            usage={
                "prompt_tokens": response.usage.input_tokens if response.usage else 0,
                "completion_tokens": response.usage.output_tokens if response.usage else 0,
                "total_tokens": (
                    response.usage.input_tokens + response.usage.output_tokens
                    if response.usage
                    else 0
                ),
            },
            raw=response,
        )

    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AsyncIterator[str]:
        client = AsyncAnthropic(api_key=self.api_key)

        system_msg = ""
        api_messages = []
        for m in messages:
            if m["role"] == "system":
                system_msg = m["content"]
            else:
                api_messages.append(m)

        kwargs = {
            "model": self.model_name,
            "messages": api_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if system_msg:
            kwargs["system"] = system_msg

        async with client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield text
