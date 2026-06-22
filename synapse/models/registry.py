"""Provider registry — maps provider names to classes."""

from __future__ import annotations

from synapse.config.schema import ModelConfig
from synapse.models.anthropic import AnthropicProvider
from synapse.models.base import BaseProvider
from synapse.models.compat import OpenAICompatProvider
from synapse.models.deepseek import DeepSeekProvider
from synapse.models.gemini import GeminiProvider

PROVIDER_REGISTRY: dict[str, type[BaseProvider]] = {
    "deepseek": DeepSeekProvider,
    "anthropic": AnthropicProvider,
    "gemini": GeminiProvider,
    "compat": OpenAICompatProvider,
    "openai": OpenAICompatProvider,  # OpenAI uses the same API format
    "groq": OpenAICompatProvider,
    "together": OpenAICompatProvider,
    "ollama": OpenAICompatProvider,
    "vllm": OpenAICompatProvider,
    "fireworks": OpenAICompatProvider,
    "perplexity": OpenAICompatProvider,
}

# Well-known provider presets for the config wizard
PROVIDER_PRESETS: dict[str, dict] = {
    "deepseek": {
        "provider": "deepseek",
        "base_url": "https://api.deepseek.com/v1",
        "env_key": "DEEPSEEK_API_KEY",
        "models": ["deepseek-chat", "deepseek-reasoner"],
        "desc": "DeepSeek — 高性价比，中文优秀",
    },
    "openai": {
        "provider": "compat",
        "base_url": "https://api.openai.com/v1",
        "env_key": "OPENAI_API_KEY",
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-4.1"],
        "desc": "OpenAI — GPT-4o 系列",
    },
    "anthropic": {
        "provider": "anthropic",
        "base_url": "https://api.anthropic.com/v1",
        "env_key": "ANTHROPIC_API_KEY",
        "models": ["claude-sonnet-4-20250514", "claude-3-5-haiku-latest"],
        "desc": "Anthropic Claude — 深度推理，代码能力强",
    },
    "gemini": {
        "provider": "gemini",
        "base_url": "",
        "env_key": "GEMINI_API_KEY",
        "models": ["gemini-2.5-flash", "gemini-2.5-pro"],
        "desc": "Google Gemini — 多模态，免费额度大",
    },
    "groq": {
        "provider": "compat",
        "base_url": "https://api.groq.com/openai/v1",
        "env_key": "GROQ_API_KEY",
        "models": ["llama-4-scout-17b-16e-instruct", "deepseek-r1-distill-llama-70b"],
        "desc": "Groq — 极速推理（免费额度）",
    },
    "together": {
        "provider": "compat",
        "base_url": "https://api.together.xyz/v1",
        "env_key": "TOGETHER_API_KEY",
        "models": ["meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"],
        "desc": "Together AI — 开源模型托管",
    },
    "ollama": {
        "provider": "compat",
        "base_url": "http://localhost:11434/v1",
        "env_key": "",
        "models": ["qwen2.5:72b", "llama3.1:70b", "deepseek-r1:70b"],
        "desc": "Ollama — 本地运行，完全免费",
    },
}


def get_provider(provider_name: str) -> type[BaseProvider]:
    """Get a provider class by name."""
    if provider_name not in PROVIDER_REGISTRY:
        available = list(PROVIDER_REGISTRY.keys())
        raise ValueError(f"Unknown provider '{provider_name}'. Available: {available}")
    return PROVIDER_REGISTRY[provider_name]


def get_provider_for_model(model_config: ModelConfig) -> BaseProvider:
    """Instantiate a provider for the given model config."""
    provider_cls = get_provider(model_config.provider)
    return provider_cls(
        model_name=model_config.model,
        api_key=model_config.api_key,
        base_url=model_config.base_url,
    )
