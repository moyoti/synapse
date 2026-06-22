"""Configuration defaults."""

DEFAULT_CONFIG_YAML = """# Synapse Configuration
# See design.md for full documentation

models:
  deepseek:
    provider: deepseek
    model: deepseek-chat
    api_key: ${DEEPSEEK_API_KEY}
    base_url: https://api.deepseek.com/v1
    default_params:
      temperature: 0.7
      max_tokens: 4096

  # Uncomment to add more providers:
  # claude:
  #   provider: anthropic
  #   model: claude-sonnet-4-20250514
  #   api_key: ${ANTHROPIC_API_KEY}
  #   default_params:
  #     temperature: 0.5
  #     max_tokens: 8192
  #
  # openai:
  #   provider: openai
  #   model: gpt-4o
  #   api_key: ${OPENAI_API_KEY}
  #
  # gemini:
  #   provider: gemini
  #   model: gemini-2.5-flash
  #   api_key: ${GEMINI_API_KEY}
  #
  # ollama:
  #   provider: ollama
  #   model: qwen2.5:72b
  #   base_url: http://localhost:11434/v1

roles:
  default:
    description: "Default assistant role"
    model: deepseek
    system_prompt: "You are a helpful AI assistant."

  orchestrator:
    description: "Task decomposition and planning"
    model: deepseek
    system_prompt: |
      You are Synapse's task orchestrator. Analyze user requests and decompose complex tasks into subtasks.
      See the orchestration protocol for details.

  coder:
    description: "Code generation and modification"
    model: deepseek
    system_prompt: |
      You are a senior software engineer. Write high-quality, maintainable code with:
      - Type hints and docstrings
      - Modern best practices
      - Proper error handling
      Output only the code unless asked to explain.

  reviewer:
    description: "Code review and quality assurance"
    model: deepseek
    system_prompt: |
      You are a code reviewer. Review code for quality, security, and performance.
      Output structured feedback:
      - Critical: blocking issues
      - Warning: potential problems
      - Suggestion: improvements

embedding:
  provider: local
  model: all-MiniLM-L6-v2
  dims: 384

memory:
  store_dir: ~/.synapse/memory
  vector_top_k: 15
  keyword_top_k: 5
  final_top_k: 5
  recency_halflife_days: 30
  importance_threshold: 0.3
  rerank_enabled: false
  rerank_candidate_n: 10
  auto_compact: true
  auto_inject: true

execution:
  max_parallel_tasks: 3
  task_timeout: 300
  max_retries: 2
  stream_output: true

terminal:
  theme: dark
  stream_panel: true
  history_file: ~/.synapse/history
  max_history: 1000
"""

DEFAULT_ORCHESTRATOR_PROMPT = """You are Synapse's task orchestrator. Your job is to analyze user requests and decide how to handle them.

For simple queries (single-step, no decomposition needed), respond directly.

For complex tasks, break them down into subtasks. Output your plan as a JSON object:

{
  "mode": "orchestrate",
  "tasks": [
    {
      "id": "task_1",
      "role": "coder",
      "prompt": "specific task description",
      "depends_on": []
    }
  ]
}

Rules:
- Use depends_on to express task dependencies (list of task IDs)
- Tasks with no dependencies can run in parallel
- Choose the best role for each task
- Keep prompts specific and actionable
"""

DEFAULT_CODER_PROMPT = """You are a senior software engineer. Write high-quality, maintainable code with:
- Type hints
- Docstrings where appropriate
- Modern best practices
- Proper error handling
"""

DEFAULT_REVIEWER_PROMPT = """You are a code reviewer. Review code for quality, security, and performance.
Output structured feedback:
- **Critical**: blocking issues
- **Warning**: potential problems
- **Suggestion**: improvements
"""

DEFAULT_MEMORY_PROMPT = """You are a memory management agent. Your tasks:
1. Store memories with appropriate categories and importance
2. Retrieve relevant memories for queries
3. Extract persistent facts from conversations
4. Summarize and compress session histories

Memory categories: fact, preference, decision, knowledge, event, relation
"""
