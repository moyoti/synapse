"""Shared CLI helpers to avoid circular imports."""

from synapse.config.loader import load_config, create_default_config
from synapse.config.schema import SynapseConfig


def load_synapse_config() -> SynapseConfig:
    """Load config, creating defaults if missing. Safe to import from any CLI module."""
    create_default_config()
    return load_config()
