"""CLI pipe mode — read prompt from stdin."""

from __future__ import annotations

import sys

from rich.console import Console

from synapse.cli.helpers import load_synapse_config
from synapse.core.router import detect_mode
from synapse.models.registry import get_provider_for_model
from synapse.utils.streaming import stream_to_console

console = Console()


async def run_pipe(
    mode: str = "auto",
    role: str | None = None,
    model: str | None = None,
    output: str | None = None,
):
    """Read input from stdin and execute."""
    # Read all from stdin
    if sys.stdin.isatty():
        console.print("[red]No input from pipe. Use: echo 'prompt' | synapse pipe[/red]")
        return

    prompt = sys.stdin.read().strip()
    if not prompt:
        console.print("[red]Empty input.[/red]")
        return

    config = load_synapse_config()

    # Determine effective mode
    effective_mode = mode
    if effective_mode == "auto":
        detected = detect_mode(prompt)
        effective_mode = detected.value
        console.print(f"[dim]Auto-detected: [bold]{effective_mode}[/bold][/dim]")

    result = ""
    try:
        if effective_mode == "debate":
            from synapse.cli.debate_ui import DebateUI
            ui = DebateUI(config, console=console)
            result = await ui.run(prompt)

        elif effective_mode == "pipeline":
            from synapse.cli.pipeline_ui import PipelineUI
            ui = PipelineUI(config, console=console)
            result = await ui.run(prompt)

        elif effective_mode == "orchestrate":
            from synapse.cli.orchestrate_ui import OrchestrationUI
            ui = OrchestrationUI(config, console=console)
            result = await ui.run(prompt)

        else:
            result = await _pipe_single(config, prompt, role, model)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")

    if output and result:
        from pathlib import Path
        Path(output).write_text(result)
        console.print(f"\n[green]✓[/green] Output saved to [bold]{output}[/bold]")


async def _pipe_single(config, prompt, role=None, model=None):
    role_name = role or "default"
    if role_name not in config.roles:
        console.print(f"[red]Role '{role_name}' not found.[/red]")
        return ""

    role_config = config.roles[role_name]
    model_name = model or role_config.model
    if model_name not in config.models:
        console.print(f"[red]Model '{model_name}' not found.[/red]")
        return ""

    model_config = config.models[model_name]
    provider = get_provider_for_model(model_config)
    temperature = model_config.default_params.temperature
    max_tokens = model_config.default_params.max_tokens

    messages = []
    if role_config.system_prompt:
        messages.append({"role": "system", "content": role_config.system_prompt})
    messages.append({"role": "user", "content": prompt})

    stream = provider.chat_stream(
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return await stream_to_console(stream, title=role_name, console=console)
