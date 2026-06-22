"""Synapse CLI — Typer-based command-line interface."""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel

from synapse import __version__
from synapse.config.loader import create_default_config
from synapse.cli.helpers import load_synapse_config

app = typer.Typer(
    name="synapse",
    help="Multi-model collaborative AI Agent CLI",
    add_completion=False,
)

console = Console()

# Sub-command groups
config_app = typer.Typer(help="Manage Synapse configuration")
app.add_typer(config_app, name="config")

memory_app = typer.Typer(help="Search and manage memories")
app.add_typer(memory_app, name="memory")

# Register memory subcommands
from synapse.cli.commands.memory_cmd import (
    memory_search, memory_list, memory_forget, memory_stats, memory_compact,
)
memory_app.command("search")(memory_search)
memory_app.command("list")(memory_list)
memory_app.command("forget")(memory_forget)
memory_app.command("stats")(memory_stats)
memory_app.command("compact")(memory_compact)

from synapse.cli.commands.models_cmd import models_app
app.add_typer(models_app, name="models")

# Session management
session_app = typer.Typer(help="Manage conversation sessions")
app.add_typer(session_app, name="session")

from synapse.cli.commands.session_cmd import session_list, session_show, session_delete
session_app.command("list")(session_list)
session_app.command("show")(session_show)
session_app.command("delete")(session_delete)


@app.command()
def version():
    """Show Synapse version."""
    console.print(f"Synapse v{__version__}", style="bold green")


@config_app.command("init")
def config_init():
    """Initialize Synapse configuration with defaults."""
    path = create_default_config()
    console.print(f"[green]✓[/green] Configuration created at [bold]{path}[/bold]")
    console.print("  Edit this file to add models and roles.")


@config_app.command("show")
def config_show():
    """Show current configuration."""
    config = load_synapse_config()

    console.print(Panel("Synapse Configuration", style="bold blue"))
    console.print()

    console.print("[bold]Models:[/bold]")
    for name, model in config.models.items():
        console.print(f"  • {name} ([dim]{model.provider}[/dim] / {model.model})")

    console.print()
    console.print("[bold]Roles:[/bold]")
    for name, role in config.roles.items():
        console.print(f"  • {name} → [dim]{role.model}[/dim]")

    console.print()
    console.print("[bold]Embedding:[/bold]")
    console.print(f"  Provider: {config.embedding.provider}, Model: {config.embedding.model}")

    console.print()
    console.print("[bold]Memory:[/bold]")
    console.print(f"  Store: {config.memory.store_dir}")


@config_app.command("add-model")
def config_add_model():
    """Interactively add a new model."""
    from synapse.cli.commands.config_wizard import cmd_add_model
    cmd_add_model()


@config_app.command("add-role")
def config_add_role():
    """Interactively add a new role."""
    from synapse.cli.commands.config_wizard import cmd_add_role
    cmd_add_role()


@config_app.command("set-key")
def config_set_key(
    key: str = typer.Argument(..., help="Environment variable name, e.g. DEEPSEEK_API_KEY"),
    value: str = typer.Argument(..., help="Your API key"),
):
    """Set an API key. Saves to ~/.synapse/.env (loaded automatically)."""
    from synapse.config.loader import set_key_in_env
    set_key_in_env(key, value)
    console.print(f"[green]✓[/green] API key [bold]{key}[/bold] saved to [dim]~/.synapse/.env[/dim]")


@config_app.command("check")
def config_check():
    """Check if API keys are configured correctly."""
    config = load_synapse_config()
    from synapse.config.loader import find_required_keys

    missing = find_required_keys(config)
    if missing:
        console.print("[bold yellow]⚠ Missing API keys:[/bold yellow]")
        for var_name in missing:
            console.print()
            console.print(f"  [cyan]{var_name}[/cyan] — not set")
            console.print(f"    Set it with: [bold]synapse config set-key {var_name} YOUR_KEY[/bold]")
            console.print(f"    Or get a key from the provider's website")
    else:
        console.print("[green]✓[/green] All API keys are configured!")

    # Also show configured models
    console.print()
    console.print(f"[bold]Configured models:[/bold] {len(config.models)}")
    for name, model in config.models.items():
        has_key = bool(model.api_key and model.api_key != "${%s}" % model.api_key.replace("${", "").replace("}", ""))
        status = "[green]✓[/green]" if has_key else "[yellow]?[/yellow]"
        console.print(f"  {status} {name} → {model.provider} / {model.model}")


@app.command()
def chat(
    mode: str = typer.Option("auto", help="Collaboration mode: auto, single, orchestrate, debate, pipeline"),
    role: Optional[str] = typer.Option(None, help="Role to use (default: 'default')"),
    model: Optional[str] = typer.Option(None, help="Override model name"),
    classic: bool = typer.Option(False, "--classic", help="Use legacy rich-based TUI"),
):
    """Start an interactive chat session."""
    import asyncio
    from synapse.cli.chat import run_chat
    asyncio.run(run_chat(mode=mode, role=role, model=model, classic=classic))


@app.command()
def run(
    prompt: str = typer.Argument(..., help="Task prompt to execute"),
    mode: str = typer.Option("auto", help="Collaboration mode: auto, single, orchestrate, debate, pipeline"),
    role: Optional[str] = typer.Option(None, help="Role to use (default: 'default')"),
    model: Optional[str] = typer.Option(None, help="Override model name"),
    output: Optional[str] = typer.Option(None, help="Save output to file"),
):
    """Execute a single task."""
    import asyncio
    from synapse.cli.run import execute_task
    asyncio.run(execute_task(prompt=prompt, mode=mode, role=role, model=model, output=output))


@app.command()
def pipe(
    mode: str = typer.Option("auto", help="Collaboration mode: auto, single, orchestrate, debate, pipeline"),
    role: Optional[str] = typer.Option(None, help="Role to use (default: 'default')"),
    model: Optional[str] = typer.Option(None, help="Override model name"),
    output: Optional[str] = typer.Option(None, help="Save output to file"),
):
    """Read prompt from stdin (pipe mode). Example: echo 'hello' | synapse pipe"""
    import asyncio
    from synapse.cli.pipe import run_pipe
    asyncio.run(run_pipe(mode=mode, role=role, model=model, output=output))


if __name__ == "__main__":
    app()
