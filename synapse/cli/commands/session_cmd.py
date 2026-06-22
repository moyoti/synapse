"""CLI commands for session management."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from synapse.memory import MemoryStore
from synapse.cli.helpers import load_synapse_config
from pathlib import Path

console = Console()


def _get_store() -> MemoryStore:
    config = load_synapse_config()
    store_dir = Path(config.memory.store_dir).expanduser()
    return MemoryStore(store_dir / "synapse.db")


def session_list(
    limit: int = typer.Option(20, help="Max sessions to show"),
):
    """List recent sessions."""
    store = _get_store()
    sessions = store.list_sessions(limit=limit)

    if not sessions:
        console.print("[dim]No sessions recorded yet.[/dim]")
        return

    table = Table(title="Sessions")
    table.add_column("ID", style="dim")
    table.add_column("Title", style="cyan")
    table.add_column("Messages", style="yellow")
    table.add_column("Model", style="green")
    table.add_column("Created", style="dim")

    for s in sessions:
        table.add_row(
            s.id[:8],
            s.title or s.summary[:50] if s.summary else "(untitled)",
            str(s.message_count),
            s.model_used or "-",
            s.created_at[:16],
        )

    console.print(table)


def session_show(
    session_id: str = typer.Argument(..., help="Session ID to view"),
):
    """View session details."""
    store = _get_store()
    session = store.get_session(session_id)

    if not session:
        console.print(f"[red]Session '{session_id}' not found.[/red]")
        return

    console.print(Panel(f"Session: {session.id}", style="bold blue"))
    console.print(f"  Title: {session.title or '(untitled)'}")
    console.print(f"  Created: {session.created_at}")
    console.print(f"  Messages: {session.message_count}")
    console.print(f"  Model: {session.model_used or '-'}")

    if session.tags:
        console.print(f"  Tags: {', '.join(session.tags)}")

    if session.summary:
        console.print()
        console.print(Panel(session.summary, title="Summary"))


def session_delete(
    session_id: str = typer.Argument(..., help="Session ID to delete"),
):
    """Delete a session and its memories."""
    store = _get_store()
    store.delete_session(session_id)
    console.print(f"[green]✓[/green] Deleted session [bold]{session_id}[/bold]")
