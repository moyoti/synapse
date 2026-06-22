# Synapse

**Multi-model collaborative AI Agent CLI framework.**

Synapse lets you configure multiple LLM providers, define roles with custom system prompts, and collaborate with multiple models from your terminal.

## Features (Phase 1)

- **Multi-model support**: DeepSeek, Anthropic Claude, and any OpenAI-compatible API (Ollama, vLLM, Groq, etc.)
- **Role system**: Bind models to roles with custom system prompts
- **Interactive chat**: `synapse chat` with live streaming Markdown
- **Single tasks**: `synapse run "prompt"` for quick one-shot queries
- **Config wizard**: `synapse config add-model` and `add-role` for easy setup

## Quick Start

```bash
# Install
cd ~/projects/synapse
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .

# Initialize config
synapse config init

# Set your API key
export DEEPSEEK_API_KEY="sk-..."

# Add a model interactively
synapse config add-model

# List models
synapse models list

# Start chatting
synapse chat

# Single task
synapse run "Explain quantum computing in one paragraph"
```

## Configuration

Config lives at `~/.synapse/config.yaml`:

```yaml
models:
  deepseek:
    provider: deepseek
    model: deepseek-chat
    api_key: ${DEEPSEEK_API_KEY}

  claude:
    provider: anthropic
    model: claude-sonnet-4-20250514
    api_key: ${ANTHROPIC_API_KEY}

  local:
    provider: compat
    model: qwen2.5:72b
    api_key: ollama
    base_url: http://localhost:11434/v1

roles:
  default:
    model: deepseek
    system_prompt: "You are a helpful assistant."

  coder:
    model: deepseek
    system_prompt: "You are a senior software engineer..."

  reviewer:
    model: claude
    system_prompt: "You are a code reviewer..."
```

## Commands

```
synapse chat                     Interactive chat (default role)
synapse chat --role coder        Chat with specific role
synapse chat --model claude      Override model

synapse run "write a quicksort"  Single task
synapse run --output result.md   Save output to file

synapse config init              Create default config
synapse config show              View configuration
synapse config add-model         Add model interactively
synapse config add-role          Add role interactively

synapse models list              List models
synapse models test deepseek     Test model connectivity

synapse version                  Show version
```

## Roadmap

- **Phase 2**: Multi-model orchestration (task decomposition, parallel execution)
- **Phase 3**: Memory Agent (vector storage, semantic retrieval, context injection)
- **Phase 4**: Debate and pipeline collaboration modes
- **Phase 5**: PyPI release, docs, CI/CD

## Design

See [design.md](https://github.com/reinyo/synapse) for the full architecture document.
