"""Pipeline CLI UI — renders sequential processing stages."""

from __future__ import annotations

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from synapse.config.schema import SynapseConfig
from synapse.core.pipeline import PipelineEngine


class PipelineUI:
    """Renders a processing pipeline with stage-by-stage output."""

    def __init__(self, config: SynapseConfig, console: Console | None = None):
        self.config = config
        self.console = console or Console()
        self.engine = PipelineEngine(config)

    async def run(self, input_data: str) -> str:
        """Execute a pipeline and render each stage."""
        self.console.print()
        self.console.print("[bold green]Pipeline Mode[/bold green] — 3 stages: Analyze → Write → Polish")
        self.console.print()

        # Execute pipeline stage by stage with progress
        result = await self.engine.run(input_data=input_data)

        # Show each stage's output
        for i, stage in enumerate(result.stages, 1):
            self.console.print(
                Panel(
                    Markdown(stage["output"]),
                    title=f"[bold green]Stage {i}: {stage['name']} ({stage['role']})[/bold green]",
                    border_style="green",
                )
            )

        # Show final output
        if result.final_output:
            self.console.print()
            self.console.print(
                Panel(
                    Markdown(result.final_output),
                    title="[bold yellow]Final Output[/bold yellow]",
                    border_style="yellow",
                )
            )

        return result.final_output
