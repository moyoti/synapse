"""Compactor — compresses session histories using an LLM."""

from __future__ import annotations


COMPACT_PROMPT = """You are a memory compactor. Summarize the following conversation into a structured summary.

Focus on:
1. Key decisions made
2. User preferences learned or changed
3. Important facts shared
4. Unresolved items or open questions
5. Actions taken and their outcomes

Keep the summary concise but complete. Use the same language as the conversation.

## Conversation

{messages}

## Summary"""


FACT_EXTRACTION_PROMPT = """You are a fact extractor. From the conversation below, extract persistent facts about the user and their context.

Output ONLY a JSON array of objects with these keys:
- "key": a short identifier (snake_case, e.g., "user_name", "preferred_language")
- "value": the fact value
- "namespace": "global" or "user_<user_id>"
- "confidence": number from 0.0 to 1.0 (how certain you are)

Rules:
- Only extract facts you are reasonably confident about (confidence >= 0.7)
- Skip trivial or obvious information
- Focus on: user preferences, environment details, project conventions, tool quirks, user identity
- Do NOT extract task progress, TODO items, or session-specific details
- Output ONLY the JSON array, nothing else

## Conversation

{messages}

## Facts"""


class Compactor:
    """Compresses conversations and extracts facts using an LLM."""

    def __init__(self, provider=None):
        """Provider should be an async function (messages, temp, max_tokens) -> ChatResponse."""
        self._provider_fn = provider

    async def summarize(self, messages: list[dict[str, str]]) -> str:
        """Generate a structured summary of a conversation."""
        if self._provider_fn is None:
            return ""

        text = _format_messages(messages)
        prompt = COMPACT_PROMPT.format(messages=text)

        try:
            response = await self._provider_fn(
                [{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1024,
            )
            return response.content if hasattr(response, 'content') else str(response)
        except Exception:
            return ""

    async def extract_facts(
        self,
        messages: list[dict[str, str]],
        user_id: str = "default",
    ) -> list[dict]:
        """Extract persistent facts from a conversation."""
        if self._provider_fn is None:
            return []

        text = _format_messages(messages)
        prompt = FACT_EXTRACTION_PROMPT.format(messages=text)

        try:
            import json

            response = await self._provider_fn(
                [{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1024,
            )

            content = response.content if hasattr(response, 'content') else str(response)

            # Extract JSON from response
            import re
            match = re.search(r"\[[\s\S]*\]", content)
            if match:
                return json.loads(match.group(0))
        except Exception:
            pass

        return []


def _format_messages(messages: list[dict[str, str]]) -> str:
    """Format message list into a readable text block."""
    lines = []
    for m in messages:
        role = m.get("role", "unknown")
        content = m.get("content", "")
        # Truncate very long messages
        if len(content) > 2000:
            content = content[:2000] + "...[truncated]"
        lines.append(f"[{role}]: {content}")
    return "\n\n".join(lines)
