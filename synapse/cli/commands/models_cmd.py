"""CLI commands for model management."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from synapse.config.loader import load_config, create_default_config

console = Console()
models_app = typer.Typer(help="List and test models")


def _get_config():
    create_default_config()
    return load_config()


@models_app.command("list")
def models_list():
    """List all configured models."""
    config = _get_config()

    table = Table(title="Configured Models")
    table.add_column("Name", style="cyan")
    table.add_column("Provider", style="green")
    table.add_column("Model", style="yellow")
    table.add_column("Base URL", style="dim")

    for name, model in config.models.items():
        table.add_row(name, model.provider, model.model, model.base_url)

    console.print(table)


@models_app.command("test")
def models_test(
    name: str = typer.Argument(..., help="Model name to test"),
):
    """Test connectivity to a specific model."""
    import asyncio

    config = _get_config()

    if name not in config.models:
        console.print(f"[red]Model '{name}' not found. Available: {list(config.models.keys())}[/red]")
        raise typer.Exit(1)

    model_config = config.models[name]

    async def _test():
        from synapse.models.registry import get_provider_for_model

        provider = get_provider_for_model(model_config)
        console.print(f"Testing [cyan]{name}[/cyan] ({model_config.provider} / {model_config.model})...")

        try:
            response = await provider.chat(
                messages=[{"role": "user", "content": "Reply with just: OK"}],
                temperature=0.1,
                max_tokens=10,
            )
            console.print(f"[green]✓[/green] Connected! Response: {response.content.strip()}")
            console.print(f"  Tokens: {response.total_tokens} | Model: {response.model}")
        except Exception as e:
            console.print(f"[red]✗[/red] Connection failed: {e}")

    asyncio.run(_test())
