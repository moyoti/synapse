"""Orchestration CLI — ties together plan → execute → aggregate with rich UI."""

from __future__ import annotations

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

from synapse.config.schema import SynapseConfig
from synapse.core.aggregator import Aggregator
from synapse.core.orchestrator import Orchestrator
from synapse.core.scheduler import Scheduler
from synapse.core.task import TaskStatus, TaskTree


class OrchestrationUI:
    """Renders the orchestration process with Rich panels."""

    def __init__(self, config: SynapseConfig, console: Console | None = None):
        self.config = config
        self.console = console or Console()
        self.orchestrator = Orchestrator(config)
        self.scheduler = Scheduler(config)
        self.aggregator = Aggregator(config)

    async def run(self, user_input: str) -> str:
        """Execute the full orchestration flow with live UI updates.

        Returns the final aggregated response.
        """
        # Step 1: Plan
        self.console.print()
        self.console.print("[bold yellow]Orchestrator[/bold yellow] analyzing task...")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            plan_task = progress.add_task("Decomposing into subtasks...", total=None)
            tree = await self.orchestrator.plan(user_input)
            progress.remove_task(plan_task)

        # Show plan
        if len(tree.tasks) == 1:
            task = next(iter(tree.tasks.values()))
            self.console.print(f"  → [cyan]{task.role}[/cyan]: {task.prompt[:100]}...")
            self.console.print()
        else:
            self._render_plan(tree)
            self.console.print()

        # Step 2: Execute
        self.console.print("[bold green]Executing[/bold green] tasks...")
        self.console.print()

        # Live-updating panels for each task
        task_panels: dict[str, str] = {tid: "" for tid in tree.tasks}

        def _make_layout() -> Table:
            table = Table.grid()
            row: list[Panel] = []
            for tid, content in task_panels.items():
                task = tree.tasks[tid]
                status_icon = {
                    TaskStatus.PENDING: "○",
                    TaskStatus.RUNNING: "◎",
                    TaskStatus.COMPLETED: "✓",
                    TaskStatus.FAILED: "✗",
                    TaskStatus.CANCELLED: "⊘",
                }.get(task.status, "?")

                title = f"[{status_icon}] {task.id} ({task.role})"
                if task.status == TaskStatus.RUNNING:
                    title = f"[bold blue]{title}[/bold blue]"
                elif task.status == TaskStatus.COMPLETED:
                    title = f"[bold green]{title}[/bold green]"
                elif task.status == TaskStatus.FAILED:
                    title = f"[bold red]{title}[/bold red]"

                panel = Panel(
                    content[:500] + ("..." if len(content) > 500 else "") or "[dim]waiting...[/dim]",
                    title=title,
                    border_style="blue" if task.status == TaskStatus.RUNNING else "dim",
                )
                row.append(panel)
            table.add_row(*row)
            return table

        # Execute with live updates
        executed_tree: TaskTree | None = None
        with Live(_make_layout(), console=self.console, refresh_per_second=4, transient=False) as live:
            # Run the scheduler in parallel with UI updates
            import asyncio

            # Start execution
            exec_future = asyncio.ensure_future(self.scheduler.run(tree))

            # Poll for updates
            while not exec_future.done():
                await asyncio.sleep(0.3)
                # Update panels with current results
                for tid, task in tree.tasks.items():
                    if task.result:
                        task_panels[tid] = task.result
                    elif task.error:
                        task_panels[tid] = f"[red]Error: {task.error}[/red]"
                live.update(_make_layout())

            executed_tree = await exec_future

        self.console.print()

        # Step 3: Show individual results
        for tid, task in executed_tree.tasks.items():
            if task.status == TaskStatus.COMPLETED and task.result:
                self.console.print(
                    Panel(
                        task.result,
                        title=f"[bold green]✓ {task.id} ({task.role})[/bold green]",
                        border_style="green",
                    )
                )

        # Step 4: Aggregate
        if len(executed_tree.tasks) > 1 and executed_tree.all_success():
            self.console.print()
            self.console.print("[bold yellow]Aggregator[/bold yellow] synthesizing results...")

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
            ) as progress:
                agg_task = progress.add_task("Generating final response...", total=None)
                final = await self.aggregator.aggregate(user_input, executed_tree)
                progress.remove_task(agg_task)

            self.console.print()
            self.console.print(Panel(final, title="[bold]Final Result[/bold]", border_style="yellow"))
            return final

        # Single task or failure
        results_map = executed_tree.get_results_map()
        if results_map:
            return next(iter(results_map.values()))
        return "No results produced."

    def _render_plan(self, tree: TaskTree) -> None:
        """Render the task plan as a tree."""
        self.console.print()
        self.console.print("[bold]Plan:[/bold]")

        # Build a simple dependency tree display
        for task in tree.tasks.values():
            indent = "  "
            deps = f" → depends on [{', '.join(task.depends_on)}]" if task.depends_on else ""
            self.console.print(
                f"{indent}[cyan]{task.id}[/cyan] [dim]({task.role})[/dim]: "
                f"{task.prompt[:120]}{'...' if len(task.prompt) > 120 else ''}{deps}"
            )
