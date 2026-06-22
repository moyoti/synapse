"""DeepSeek provider — uses OpenAI-compatible API."""

from __future__ import annotations

from synapse.models.compat import OpenAICompatProvider


class DeepSeekProvider(OpenAICompatProvider):
    """DeepSeek provider. Uses OpenAI-compatible API at api.deepseek.com."""

    provider_name = "deepseek"
    base_url = "https://api.deepseek.com/v1"
