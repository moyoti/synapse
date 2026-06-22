"""CLI commands for memory management."""

from __future__ import annotations

import asyncio

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from synapse.cli.helpers import load_synapse_config
from synapse.memory import MemoryAgent

console = Console()


def _get_memory_agent() -> MemoryAgent:
    config = load_synapse_config()
    return MemoryAgent(config)


def memory_search(
    query: str = typer.Argument(..., help="Search query for semantic retrieval"),
    category: str = typer.Option(None, help="Filter by category: fact, pref, decision, knowledge, event, relation"),
    top_k: int = typer.Option(5, help="Number of results"),
):
    """Search memories by semantic similarity."""
    from synapse.memory.schemas import MemoryCategory

    agent = _get_memory_agent()

    cat = None
    if category:
        try:
            cat = MemoryCategory(category)
        except ValueError:
            console.print(f"[red]Invalid category: {category}. Valid: {[c.value for c in MemoryCategory]}[/red]")
            raise typer.Exit(1)

    async def _search():
        return await agent.recall(query=query, top_k=top_k, category=cat)

    results = asyncio.run(_search())

    if not results:
        console.print("[dim]No memories found.[/dim]")
        return

    console.print(f"\n[bold]Results for:[/bold] {query}\n")
    for i, mem in enumerate(results, 1):
        console.print(
            Panel(
                mem.content,
                title=f"[bold cyan]{i}.[/bold cyan] [{mem.category.value}] [dim]{mem.id}[/dim]",
                subtitle=f"Importance: {mem.importance:.1f} | Last accessed: {mem.last_accessed[:10]}",
            )
        )


def memory_list(
    category: str = typer.Option(None, help="Filter by category"),
    limit: int = typer.Option(20, help="Max results"),
):
    """List recent memories."""
    from synapse.memory.schemas import MemoryCategory

    agent = _get_memory_agent()

    cat = None
    if category:
        try:
            cat = MemoryCategory(category)
        except ValueError:
            console.print(f"[red]Invalid category: {category}[/red]")
            raise typer.Exit(1)

    memories = agent.store.list_memories(category=cat, limit=limit)

    if not memories:
        console.print("[dim]No memories stored yet.[/dim]")
        return

    table = Table(title="Memories")
    table.add_column("Category", style="cyan")
    table.add_column("Content", style="white")
    table.add_column("Importance", style="yellow")
    table.add_column("Created", style="dim")

    for mem in memories:
        table.add_row(
            mem.category.value,
            mem.content[:80] + ("..." if len(mem.content) > 80 else ""),
            f"{mem.importance:.1f}",
            mem.created_at[:10],
        )

    console.print(table)


def memory_forget(
    memory_id: str = typer.Option(None, help="Memory ID to delete"),
    category: str = typer.Option(None, help="Delete all memories of this category"),
):
    """Delete memories."""
    from synapse.memory.schemas import MemoryCategory

    agent = _get_memory_agent()

    if memory_id:
        async def _del():
            return await agent.forget(memory_id=memory_id)
        count = asyncio.run(_del())
        console.print(f"[green]✓[/green] Deleted memory [bold]{memory_id}[/bold]")
    elif category:
        try:
            cat = MemoryCategory(category)
        except ValueError:
            console.print(f"[red]Invalid category: {category}[/red]")
            raise typer.Exit(1)

        async def _del_cat():
            return await agent.forget(category=cat)
        count = asyncio.run(_del_cat())
        console.print(f"[green]✓[/green] Deleted [bold]{count}[/bold] memories in category '{category}'")
    else:
        console.print("[red]Specify --id or --category[/red]")


def memory_stats():
    """Show memory system statistics."""
    agent = _get_memory_agent()
    stats = agent.stats()

    console.print(Panel("Memory Statistics", style="bold blue"))
    console.print()

    console.print(f"[bold]Total memories:[/bold] {stats['total_memories']}")
    console.print(f"[bold]Vector memories:[/bold] {stats['vector_memories']}")
    console.print(f"[bold]Total sessions:[/bold] {stats['total_sessions']}")
    console.print(f"[bold]Total facts:[/bold] {stats['total_facts']}")

    if stats["by_category"]:
        console.print()
        console.print("[bold]By category:[/bold]")
        for cat, count in stats["by_category"].items():
            console.print(f"  • {cat}: {count}")


def memory_compact():
    """Run memory maintenance — cleanup expired entries."""
    agent = _get_memory_agent()

    async def _run():
        return await agent.consolidate()

    deleted = asyncio.run(_run())
    console.print(f"[green]✓[/green] Consolidated: [bold]{deleted}[/bold] expired memories removed.")
