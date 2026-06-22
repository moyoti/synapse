"""Debate CLI UI — renders multi-agent debate with rich panels."""

from __future__ import annotations

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from synapse.config.schema import SynapseConfig
from synapse.core.debate import DebateEngine


class DebateUI:
    """Renders a multi-agent debate with live-updating panels."""

    def __init__(self, config: SynapseConfig, console: Console | None = None):
        self.config = config
        self.console = console or Console()
        self.engine = DebateEngine(config)

    async def run(self, question: str, num_perspectives: int = 3) -> str:
        """Run a debate and render results."""
        self.console.print()
        self.console.print(f"[bold magenta]Debate Mode[/bold magenta] — {num_perspectives} perspectives")
        self.console.print(f"[dim]Question: {question[:100]}...[/dim]")
        self.console.print()

        # Show which perspectives
        agents = self.engine._auto_assign_perspectives(question, num_perspectives)
        for i, agent in enumerate(agents, 1):
            self.console.print(f"  [cyan]Perspective {i}:[/cyan] {agent.name} — {agent.perspective}")

        self.console.print()
        self.console.print("[dim]Gathering perspectives in parallel...[/dim]")

        # Run debate (this blocks — we show a simple spinner)
        with self.console.status("[bold yellow]Debating...[/bold yellow]") as status:
            result = await self.engine.debate(
                question=question,
                agents=agents,
            )

        # Show individual perspectives
        self.console.print()
        for i, p in enumerate(result.perspectives, 1):
            self.console.print(
                Panel(
                    Markdown(p["answer"]),
                    title=f"[bold cyan]Perspective {i}: {p['name']} — {p['perspective']}[/bold cyan]",
                    border_style="cyan",
                )
            )

        # Show synthesis
        self.console.print()
        self.console.print(
            Panel(
                Markdown(result.synthesis),
                title="[bold yellow]Synthesis[/bold yellow]",
                border_style="yellow",
            )
        )

        return result.synthesis
