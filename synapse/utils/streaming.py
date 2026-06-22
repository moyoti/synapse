"""Streaming output utilities for terminal display."""

from __future__ import annotations

import sys
from typing import AsyncIterator

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel


async def stream_to_console(
    stream: AsyncIterator[str],
    title: str | None = None,
    console: Console | None = None,
) -> str:
    """Stream tokens to the console with live-updating Markdown rendering.

    Args:
        stream: Async iterator of text tokens.
        title: Optional panel title (e.g., role name).
        console: Rich Console instance. Creates one if None.

    Returns:
        The full accumulated text.
    """
    if console is None:
        console = Console()

    full_text = ""

    with Live(console=console, refresh_per_second=10, transient=False) as live:
        async for token in stream:
            full_text += token
            rendered = Markdown(full_text)
            if title:
                rendered = Panel(rendered, title=title, border_style="blue")
            live.update(rendered)

    return full_text


async def stream_simple(
    stream: AsyncIterator[str],
    console: Console | None = None,
) -> str:
    """Stream tokens to stdout directly, one at a time. No rich rendering."""
    if console is None:
        console = Console()

    full_text = ""
    async for token in stream:
        full_text += token
        console.print(token, end="")
    console.print()  # final newline
    return full_text
