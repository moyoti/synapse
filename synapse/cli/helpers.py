"""Shared CLI helpers to avoid circular imports."""

from __future__ import annotations

from rich.prompt import Prompt

from synapse.config.loader import load_config, create_default_config
from synapse.config.schema import SynapseConfig


def load_synapse_config() -> SynapseConfig:
    """Load config, creating defaults if missing. Safe to import from any CLI module."""
    create_default_config()
    return load_config()


def safe_ask(
    prompt: str,
    choices: list[str] | None = None,
    default: str = "",
    password: bool = False,
) -> str:
    """Ask for input with the default shown in prompt text, NOT as editable pre-filled text.

    This prevents users from accidentally deleting the prompt prefix.
    Pressing Enter without typing uses the default value.
    """
    suffix = f" [{default}]" if default else ""
    full_prompt = f"{prompt}{suffix}"

    if password:
        value = Prompt.ask(full_prompt, password=True)
        return value if value else default

    if choices:
        choices_str = "/".join(choices)
        full_prompt = f"{prompt} [{choices_str}]"

    value = Prompt.ask(full_prompt, choices=choices or None, default="", show_default=False)
    return value if value else default
