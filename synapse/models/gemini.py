"""Google Gemini provider."""

from __future__ import annotations

from typing import AsyncIterator

from synapse.models.base import BaseProvider, ChatResponse, StreamChunk

try:
    from google import genai
    HAS_GOOGLE = True
except ImportError:
    HAS_GOOGLE = False


class GeminiProvider(BaseProvider):
    """Provider for Google Gemini models."""

    provider_name = "gemini"

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: list[dict] | None = None,
    ) -> ChatResponse:
        if not HAS_GOOGLE:
            return ChatResponse(
                content="Error: google-genai not installed. Run: pip install google-genai",
                model=self.model_name,
            )

        client = genai.Client(api_key=self.api_key)

        # Convert messages to Gemini format
        contents = []
        system_instruction = None
        for m in messages:
            if m["role"] == "system":
                system_instruction = m["content"]
            elif m["role"] == "user":
                contents.append({"role": "user", "parts": [{"text": m["content"]}]})
            elif m["role"] == "assistant":
                contents.append({"role": "model", "parts": [{"text": m["content"]}]})

        kwargs = {
            "model": self.model_name,
            "contents": contents,
        }
        if system_instruction:
            # system_instruction must be passed at the top level
            pass  # Handle differently

        try:
            response = client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                },
            )
            text = response.text or ""
        except Exception as e:
            return ChatResponse(
                content=f"Gemini API error: {e}",
                model=self.model_name,
            )

        return ChatResponse(
            content=text,
            model=self.model_name,
            finish_reason="stop",
            raw=response,
        )

    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AsyncIterator[StreamChunk]:
        if not HAS_GOOGLE:
            yield StreamChunk(content="Error: google-genai not installed. Run: pip install google-genai")
            return

        client = genai.Client(api_key=self.api_key)

        contents = []
        for m in messages:
            if m["role"] == "system":
                pass
            elif m["role"] == "user":
                contents.append({"role": "user", "parts": [{"text": m["content"]}]})
            elif m["role"] == "assistant":
                contents.append({"role": "model", "parts": [{"text": m["content"]}]})

        try:
            response = client.models.generate_content_stream(
                model=self.model_name,
                contents=contents,
                config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                },
            )
            for chunk in response:
                # Gemini thinking models expose thought in candidates
                if hasattr(chunk, 'candidates') and chunk.candidates:
                    parts = getattr(chunk.candidates[0].content, 'parts', [])
                    for part in parts:
                        if hasattr(part, 'thought') and part.thought:
                            yield StreamChunk(reasoning=str(part.thought))
                        elif hasattr(part, 'text') and part.text:
                            yield StreamChunk(content=part.text)
                elif chunk.text:
                    yield StreamChunk(content=chunk.text)
        except Exception as e:
            yield StreamChunk(content=f"Gemini API error: {e}")
