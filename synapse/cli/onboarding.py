"""Onboarding wizard — guides users through first-time model setup."""

from __future__ import annotations

import os

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

from synapse.config.loader import (
    load_config, create_default_config, save_config, set_key_in_env, find_required_keys,
)
from synapse.config.schema import SynapseConfig, ModelConfig
from synapse.models.registry import PROVIDER_PRESETS
from synapse.cli.helpers import safe_ask

console = Console()


WELCOME = r"""
  ╔══════════════════════════════════════════════╗
  ║           🧠  Welcome to Synapse!             ║
  ║     Multi-Model Collaborative AI Agent        ║
  ╚══════════════════════════════════════════════╝
"""

NO_MODELS_MSG = """
Synapse needs at least one LLM provider to work.
Let's set one up — it takes 30 seconds.

You'll need an API key from one of these providers:
"""


async def run_onboarding(config: SynapseConfig | None = None) -> SynapseConfig:
    """Run the first-time setup wizard. Returns the configured SynapseConfig.

    Call this when:
    - No API keys are configured
    - User explicitly runs /setup in chat
    """
    if config is None:
        create_default_config()
        config = load_config()

    console.print(WELCOME)

    # Check if any keys are already set
    missing = find_required_keys(config)
    if not missing:
        console.print("[green]All API keys already configured![/green]")
        return config

    console.print("[bold]Let's connect your first AI model.[/bold]")
    console.print("[dim]Press Ctrl+C at any prompt to skip[/dim]")
    console.print()

    # Show provider options
    _show_provider_menu()

    # Let user pick
    choice = safe_ask(
        "Which provider would you like to use?",
        choices=list(PROVIDER_PRESETS.keys()),
        default="deepseek",
    )

    preset = PROVIDER_PRESETS[choice]

    # Guide through setup
    console.print()
    console.print(f"[bold]Setting up {choice}...[/bold]")
    console.print(f"  {preset['desc']}")

    # API key
    env_key = preset["env_key"]
    api_key = ""
    if env_key:
        console.print()
        if choice == "deepseek":
            console.print("  Get a key: [link=https://platform.deepseek.com]https://platform.deepseek.com[/link]")
            console.print("  (New users get free credits)")
        elif choice == "openai":
            console.print("  Get a key: [link=https://platform.openai.com/api-keys]https://platform.openai.com/api-keys[/link]")
        elif choice == "anthropic":
            console.print("  Get a key: [link=https://console.anthropic.com]https://console.anthropic.com[/link]")
        elif choice == "gemini":
            console.print("  Get a key: [link=https://aistudio.google.com]https://aistudio.google.com[/link]")
            console.print("  (Generous free tier available)")
        elif choice == "groq":
            console.print("  Get a key: [link=https://console.groq.com]https://console.groq.com[/link]")
            console.print("  (Free tier with generous limits)")

        console.print()
        api_key = Prompt.ask(f"Your {env_key}", password=True)

        if api_key:
            set_key_in_env(env_key, api_key)
            console.print(f"  [green]✓ Key saved[/green]")

    # Model selection
    console.print()
    console.print("[bold]Select model:[/bold]")
    model_name = safe_ask(
        "Model",
        choices=preset["models"],
        default=preset["models"][0],
    )

    # Config name
    config_name = safe_ask("Name for this model in config", default=choice)

    # Add to config
    api_key_ref = f"${{{env_key}}}" if env_key else ""
    config.models[config_name] = ModelConfig(
        provider=preset["provider"],
        model=model_name,
        api_key=api_key_ref,
        base_url=preset["base_url"],
        default_params={"temperature": 0.7, "max_tokens": 4096},
    )
    save_config(config)

    # Update default role to use this model
    if "default" in config.roles:
        config.roles["default"].model = config_name
        save_config(config)

    console.print()
    console.print(f"[green]✓ {choice} configured as '[bold]{config_name}[/bold]'[/green]")

    # Test connection
    console.print()
    if Confirm.ask("Test connection?", default=True):
        await _test_connection(config_name, config)

    console.print()
    console.print("[bold green]Ready![/bold green] You can now start chatting.")
    console.print(f"  Add more models anytime with: [bold]/setup[/bold] in chat, or [bold]synapse config add-model[/bold]")
    console.print()

    return config


async def _test_connection(name: str, config: SynapseConfig):
    """Quick test to verify the model works."""
    from synapse.models.registry import get_provider_for_model

    model_config = config.models[name]
    provider = get_provider_for_model(model_config)

    console.print(f"  Testing [cyan]{name}[/cyan]...", end=" ")
    try:
        response = await provider.chat(
            messages=[{"role": "user", "content": "Reply with just: OK"}],
            temperature=0.1,
            max_tokens=10,
        )
        console.print("[green]✓ Connected![/green]")
    except Exception as e:
        console.print(f"[yellow]⚠ {e}[/yellow]")
        console.print("  You can still use it — the key may just need a moment to activate.")


def _show_provider_menu():
    """Display the provider selection table."""
    table = Table(title="Available Providers")
    table.add_column("#", style="dim", width=3)
    table.add_column("Provider", style="cyan")
    table.add_column("Best For", style="white")
    table.add_column("Cost", style="green")

    providers_display = [
        ("deepseek", "Best value, great Chinese", "$0.14/M tokens"),
        ("openai", "GPT-4o, strongest overall", "~$2.50/M tokens"),
        ("anthropic", "Code & deep reasoning", "~$3/M tokens"),
        ("gemini", "Free tier, multimodal", "Free tier available"),
        ("groq", "Fastest inference, free tier", "Free tier available"),
        ("together", "Open-source models", "Pay per use"),
        ("ollama", "Run locally, fully free", "FREE"),
    ]

    for i, (pid, best, cost) in enumerate(providers_display, 1):
        table.add_row(str(i), pid, best, cost)

    console.print(table)


# ── In-chat /setup command ──

async def chat_setup(config: SynapseConfig):
    """Add a model from within a chat session. Press Ctrl+C at any step to cancel."""
    console.print()
    console.print("[bold]Add another model[/bold]")
    console.print("[dim]Press Ctrl+C at any prompt to cancel[/dim]")
    console.print()

    _show_provider_menu()

    choice = safe_ask(
        "Which provider?",
        choices=list(PROVIDER_PRESETS.keys()),
        default="deepseek",
    )
    if not choice:
        console.print("[dim]Setup cancelled.[/dim]")
        console.print()
        return

    # Simplified inline setup
    preset = PROVIDER_PRESETS[choice]
    console.print(f"\n[bold]{choice}[/bold] — {preset['desc']}")

    env_key = preset["env_key"]
    if env_key:
        if os.environ.get(env_key):
            console.print(f"  {env_key}: [green]already set[/green]")
        else:
            console.print(f"  Get a key: [link=https://platform.deepseek.com]https://platform.deepseek.com[/link]")
            key = Prompt.ask(f"  Your {env_key}", password=True)
            if key:
                set_key_in_env(env_key, key)

    model_name = safe_ask(
        "  Model",
        choices=preset["models"],
        default=preset["models"][0],
    )

    config_name = safe_ask("  Config name", default=choice)

    api_key_ref = f"${{{env_key}}}" if env_key else ""
    config.models[config_name] = ModelConfig(
        provider=preset["provider"],
        model=model_name,
        api_key=api_key_ref,
        base_url=preset["base_url"],
        default_params={"temperature": 0.7, "max_tokens": 4096},
    )
    save_config(config)

    console.print(f"\n  [green]✓[/green] Added [bold]{config_name}[/bold] — switch to it with [bold]/model {config_name}[/bold]")
    console.print()

    # Test
    console.print("  Testing connection...", end=" ")
    try:
        from synapse.models.registry import get_provider_for_model
        provider = get_provider_for_model(config.models[config_name])
        resp = await provider.chat(
            messages=[{"role": "user", "content": "Reply with: OK"}],
            temperature=0.1,
            max_tokens=5,
        )
        console.print("[green]Connected![/green]")
    except Exception as e:
        console.print(f"[yellow]Note: {e}[/yellow]")
