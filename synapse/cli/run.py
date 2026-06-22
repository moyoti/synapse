"""CLI single-task execution mode — supports all collaboration modes."""

from __future__ import annotations

from rich.console import Console

from synapse.cli.helpers import load_synapse_config
from synapse.core.router import detect_mode
from synapse.models.registry import get_provider_for_model
from synapse.utils.streaming import stream_to_console

console = Console()


async def execute_task(
    prompt: str,
    mode: str = "auto",
    role: str | None = None,
    model: str | None = None,
    output: str | None = None,
):
    """Execute a single task with the appropriate mode."""
    config = load_synapse_config()

    # Determine effective mode
    effective_mode = mode
    if effective_mode == "auto":
        detected = detect_mode(prompt)
        effective_mode = detected.value
        console.print(f"[dim]Auto-detected mode: [bold]{effective_mode}[/bold][/dim]")

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

        else:  # single
            result = await _run_single(config, prompt, role, model)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        # Fallback to single mode
        try:
            console.print("[dim]Falling back to single mode...[/dim]")
            result = await _run_single(config, prompt, role, model)
        except Exception as e2:
            console.print(f"[red]Fallback also failed: {e2}[/red]")

    # Save output
    if output and result:
        from pathlib import Path
        Path(output).write_text(result)
        console.print(f"\n[green]✓[/green] Output saved to [bold]{output}[/bold]")

    return result


async def _run_single(config, prompt, role=None, model=None):
    """Run a single-model task."""
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

    console.print(f"[bold blue]{role_name} ({model_name}):[/bold blue]")

    stream = provider.chat_stream(
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return await stream_to_console(stream, title=role_name, console=console)
