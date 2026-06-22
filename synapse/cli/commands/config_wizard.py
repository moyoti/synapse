"""Interactive configuration commands — add-model, add-role."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.table import Table

from synapse.config.loader import load_config, create_default_config, save_config, set_key_in_env
from synapse.config.schema import ModelConfig, RoleConfig
from synapse.models.registry import PROVIDER_PRESETS

console = Console()


def cmd_add_model():
    """Interactively add a new model to the config."""
    config = load_config()

    console.print("[bold blue]Add a new model[/bold blue]")
    console.print()

    # Show presets
    console.print("[bold]Available provider presets:[/bold]")
    table = Table()
    table.add_column("ID", style="cyan")
    table.add_column("Provider", style="green")
    table.add_column("Description", style="white")

    for pid, preset in PROVIDER_PRESETS.items():
        table.add_row(pid, preset["provider"], preset["desc"])
    table.add_row("custom", "any", "Custom OpenAI-compatible endpoint")
    console.print(table)
    console.print()

    # Choose preset or custom
    choice = Prompt.ask(
        "Choose provider preset (or 'custom')",
        choices=list(PROVIDER_PRESETS.keys()) + ["custom"],
        default="deepseek",
    )

    if choice == "custom":
        _add_custom_model(config)
    else:
        _add_preset_model(config, choice)


def _add_preset_model(config, preset_id: str):
    """Add a model from a known preset."""
    preset = PROVIDER_PRESETS[preset_id]

    name = Prompt.ask("Config name (e.g. 'my-deepseek', 'my-claude')", default=preset_id)
    if name in config.models:
        if not Confirm.ask(f"Model '{name}' already exists. Overwrite?"):
            return

    model_name = Prompt.ask(
        "Model ID",
        choices=preset["models"],
        default=preset["models"][0],
    )

    env_key = preset["env_key"]
    api_key = ""
    if env_key:
        api_key = f"${{{env_key}}}"
        # Check if key is set
        import os
        if not os.environ.get(env_key):
            console.print(f"\n[yellow]⚠ {env_key} is not set.[/yellow]")
            if Confirm.ask(f"Enter your {env_key} now?"):
                key_value = Prompt.ask(f"{env_key}", password=True)
                set_key_in_env(env_key, key_value)
                console.print(f"[green]✓[/green] Key saved to ~/.synapse/.env")

    temp = float(Prompt.ask("Temperature", default="0.7"))
    max_tokens = int(Prompt.ask("Max tokens", default="4096"))

    config.models[name] = ModelConfig(
        provider=preset["provider"],
        model=model_name,
        api_key=api_key,
        base_url=preset["base_url"],
        default_params={"temperature": temp, "max_tokens": max_tokens},
    )

    save_config(config)
    console.print(f"[green]✓[/green] Model '[bold]{name}[/bold]' added.")


def _add_custom_model(config):
    """Add a custom OpenAI-compatible model."""
    name = Prompt.ask("Config name (e.g., 'local-qwen', 'my-groq')")
    if name in config.models:
        if not Confirm.ask(f"Model '{name}' already exists. Overwrite?"):
            return

    provider = Prompt.ask(
        "Provider type",
        choices=["compat", "deepseek", "anthropic", "gemini"],
        default="compat",
    )

    model_name = Prompt.ask("Model ID (e.g., 'qwen2.5:72b')")
    api_key = Prompt.ask("API key (or ${ENV_VAR} format)", default="")
    base_url = Prompt.ask("Base URL", default="https://api.openai.com/v1")

    temp = float(Prompt.ask("Temperature", default="0.7"))
    max_tokens = int(Prompt.ask("Max tokens", default="4096"))

    config.models[name] = ModelConfig(
        provider=provider,
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        default_params={"temperature": temp, "max_tokens": max_tokens},
    )

    save_config(config)
    console.print(f"[green]✓[/green] Model '[bold]{name}[/bold]' added.")


def cmd_add_role():
    """Interactively add a new role to the config."""
    config = load_config()

    console.print("[bold blue]Add a new role[/bold blue]")
    console.print()

    name = Prompt.ask("Role name (e.g., 'coder', 'reviewer')")
    if name in config.roles:
        if not Confirm.ask(f"Role '{name}' already exists. Overwrite?"):
            return

    console.print("\nAvailable models:")
    for m_name in config.models:
        console.print(f"  • {m_name}")
    model = Prompt.ask("Model to bind", default="deepseek")

    if model not in config.models:
        console.print(f"[red]Model '{model}' not found. Add it first with: synapse config add-model[/red]")
        return

    desc = Prompt.ask("Description", default="")
    console.print("\nSystem prompt (enter on a new line, Ctrl+D to finish):")
    lines = []
    try:
        while True:
            line = input()
            lines.append(line)
    except EOFError:
        pass
    system_prompt = "\n".join(lines)

    config.roles[name] = RoleConfig(
        description=desc,
        model=model,
        system_prompt=system_prompt,
    )

    save_config(config)
    console.print(f"[green]✓[/green] Role '[bold]{name}[/bold]' added.")
