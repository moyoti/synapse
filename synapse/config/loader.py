"""Configuration loader with environment variable substitution."""

from __future__ import annotations

import os
import re
from pathlib import Path

import yaml

from synapse.config.defaults import DEFAULT_CONFIG_YAML
from synapse.config.schema import SynapseConfig

# Pattern: ${VAR_NAME} or ${VAR_NAME:-default}
_ENV_VAR_PATTERN = re.compile(r"\$\{(\w+)(?::-([^}]*))?\}")
_SYNAPSE_DIR = Path.home() / ".synapse"


def _load_dotenv():
    """Load ~/.synapse/.env into os.environ if it exists."""
    env_file = _SYNAPSE_DIR / ".env"
    if not env_file.exists():
        return
    try:
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except Exception:
        pass


def _expand_env_vars(value: str) -> str:
    """Replace ${VAR} and ${VAR:-default} in a string."""

    def _replacer(match: re.Match) -> str:
        var_name = match.group(1)
        default = match.group(2)
        env_val = os.environ.get(var_name)
        if env_val:
            return env_val
        if default is not None:
            return default
        return match.group(0)  # keep original if no env and no default

    return _ENV_VAR_PATTERN.sub(_replacer, value)


def _expand_dict(obj: object) -> object:
    """Recursively expand ${VAR} in all string values of a dict/list."""
    if isinstance(obj, dict):
        return {k: _expand_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_dict(item) for item in obj]
    if isinstance(obj, str):
        return _expand_env_vars(obj)
    return obj


def load_config(config_path: str | Path | None = None) -> SynapseConfig:
    """Load configuration from a YAML file with env var expansion.

    Search order:
    1. Load ~/.synapse/.env into environment (if exists)
    2. Explicit config_path
    3. SYNAPSE_CONFIG environment variable
    4. ~/.synapse/config.yaml
    """
    # Load .env file first so its values are available for ${VAR} expansion
    _load_dotenv()

    if config_path is None:
        config_path = os.environ.get("SYNAPSE_CONFIG")
    if config_path is None:
        config_path = _SYNAPSE_DIR / "config.yaml"

    config_path = Path(config_path).expanduser()

    if config_path.exists():
        raw = yaml.safe_load(config_path.read_text()) or {}
    else:
        raw = yaml.safe_load(DEFAULT_CONFIG_YAML) or {}

    expanded = _expand_dict(raw)
    return SynapseConfig(**expanded)


def create_default_config(dir_path: str | Path | None = None) -> Path:
    """Write the default config to ~/.synapse/config.yaml (or custom dir)."""
    if dir_path is None:
        dir_path = _SYNAPSE_DIR
    dir_path = Path(dir_path).expanduser()
    dir_path.mkdir(parents=True, exist_ok=True)

    config_file = dir_path / "config.yaml"
    if not config_file.exists():
        config_file.write_text(DEFAULT_CONFIG_YAML)

    # Also create .env template if not exists
    env_file = dir_path / ".env"
    if not env_file.exists():
        env_file.write_text(
            "# Synapse API Keys\n"
            "# Set your keys here. They are loaded automatically.\n"
            "# Get a DeepSeek key: https://platform.deepseek.com\n"
            "# DEEPSEEK_API_KEY=sk-your-key-here\n"
            "# Get a Claude key: https://console.anthropic.com\n"
            "# ANTHROPIC_API_KEY=sk-ant-your-key-here\n"
        )

    return config_file


def save_config(config: SynapseConfig, path: str | Path | None = None) -> Path:
    """Save configuration to a YAML file."""
    if path is None:
        path = _SYNAPSE_DIR / "config.yaml"
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)

    # Use model_dump to serialize
    data = config.model_dump(exclude_defaults=False)
    yaml_text = yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False)
    path.write_text(yaml_text)
    return path


def find_required_keys(config: SynapseConfig) -> dict[str, str]:
    """Find all ${VAR} references in the config that are not set.

    Returns {var_name: placeholder} for missing keys.
    """
    missing = {}

    def _scan(obj: object):
        if isinstance(obj, dict):
            for v in obj.values():
                _scan(v)
        elif isinstance(obj, list):
            for item in obj:
                _scan(item)
        elif isinstance(obj, str):
            for match in _ENV_VAR_PATTERN.finditer(obj):
                var_name = match.group(1)
                if var_name not in os.environ:
                    missing[var_name] = match.group(0)

    raw = config.model_dump(exclude_defaults=False)
    _scan(raw)
    return missing


def set_key_in_env(key: str, value: str):
    """Write an API key to ~/.synapse/.env."""
    _SYNAPSE_DIR.mkdir(parents=True, exist_ok=True)
    env_file = _SYNAPSE_DIR / ".env"

    lines = []
    found = False
    if env_file.exists():
        lines = env_file.read_text().splitlines()
        for i, line in enumerate(lines):
            if line.startswith(f"{key}=") or line.startswith(f"export {key}="):
                lines[i] = f'{key}={value}'
                found = True
                break

    if not found:
        lines.append(f'{key}={value}')

    env_file.write_text("\n".join(lines) + "\n")
    os.environ[key] = value
