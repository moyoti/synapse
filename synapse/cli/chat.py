"""CLI interactive chat mode — now powered by the full TUI renderer."""

from __future__ import annotations

from rich.console import Console

from synapse.cli.helpers import load_synapse_config
from synapse.cli.tui import ChatTUI
from synapse.config.loader import find_required_keys

console = Console()


async def run_chat(
    mode: str = "auto",
    role: str | None = None,
    model: str | None = None,
):
    """Start interactive chat with full TUI.

    Validates config, runs first-time onboarding if needed, then launches
    the ChatTUI main loop.
    """
    config = load_synapse_config()

    role_name = role or "default"
    if role_name not in config.roles:
        console.print(
            f"[red]Role '{role_name}' not found. Available: {list(config.roles.keys())}[/red]"
        )
        return

    model_name = model or config.roles[role_name].model
    if model_name not in config.models:
        console.print(
            f"[red]Model '{model_name}' not found. Available: {list(config.models.keys())}[/red]"
        )
        return

    tui = ChatTUI(config, role=role_name, model=model_name, mode=mode)
    try:
        await tui.run()
    finally:
        tui.close()
