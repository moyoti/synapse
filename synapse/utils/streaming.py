"""Streaming output utilities for terminal display."""

from __future__ import annotations

import sys
from typing import AsyncIterator

from synapse.models.base import StreamChunk

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel


async def stream_to_console(
    stream: AsyncIterator[StreamChunk],
    title: str | None = None,
    console: Console | None = None,
) -> str:
    """Stream tokens to the console with live-updating Markdown rendering.

    Handles StreamChunk with optional reasoning/thinking content.
    """
    if console is None:
        console = Console()

    full_text = ""
    reasoning_text = ""

    with Live(console=console, refresh_per_second=10, transient=False) as live:
        async for chunk in stream:
            if chunk.reasoning:
                reasoning_text += chunk.reasoning
            if chunk.content:
                full_text += chunk.content

            display = full_text
            if reasoning_text:
                display = f"[dim italic]🤔 Thinking...\n{reasoning_text}[/]\n\n{display}"

            rendered = Markdown(display or "...")
            if title:
                rendered = Panel(rendered, title=title, border_style="blue")
            live.update(rendered)

    if reasoning_text:
        full_text = f"<details><summary>🤔 Thinking</summary>\n\n{reasoning_text}\n\n</details>\n\n{full_text}"
    return full_text


async def stream_simple(
    stream: AsyncIterator[StreamChunk],
    console: Console | None = None,
) -> str:
    """Stream tokens to stdout directly, one at a time. No rich rendering."""
    if console is None:
        console = Console()

    full_text = ""
    reasoning_text = ""
    async for chunk in stream:
        if chunk.reasoning:
            reasoning_text += chunk.reasoning
            console.print(chunk.reasoning, end="", style="dim italic")
        if chunk.content:
            full_text += chunk.content
            console.print(chunk.content, end="")
    console.print()

    if reasoning_text:
        full_text = f"<details><summary>🤔 Thinking</summary>\n\n{reasoning_text}\n\n</details>\n\n{full_text}"
    return full_text
