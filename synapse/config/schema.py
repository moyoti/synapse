"""Pydantic configuration models for Synapse."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ModelDefaultParams(BaseModel):
    temperature: float = 0.7
    max_tokens: int = 4096


class ModelConfig(BaseModel):
    provider: str = "compat"
    model: str
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    default_params: ModelDefaultParams = Field(default_factory=ModelDefaultParams)


class RoleConfig(BaseModel):
    description: str = ""
    model: str
    system_prompt: str = ""


class EmbeddingConfig(BaseModel):
    provider: str = "local"
    model: str = "all-MiniLM-L6-v2"
    dims: int = 384
    api_key: str = ""
    base_url: str = ""


class MemoryConfig(BaseModel):
    store_dir: str = "~/.synapse/memory"
    vector_top_k: int = 15
    keyword_top_k: int = 5
    final_top_k: int = 5
    recency_halflife_days: int = 30
    importance_threshold: float = 0.3
    rerank_enabled: bool = False
    rerank_candidate_n: int = 10
    auto_compact: bool = True
    auto_inject: bool = True


class ExecutionConfig(BaseModel):
    max_parallel_tasks: int = 3
    task_timeout: int = 300
    max_retries: int = 2
    stream_output: bool = True


class TerminalConfig(BaseModel):
    theme: str = "dark"
    stream_panel: bool = True
    history_file: str = "~/.synapse/history"
    max_history: int = 1000


class SynapseConfig(BaseModel):
    models: dict[str, ModelConfig] = Field(default_factory=dict)
    roles: dict[str, RoleConfig] = Field(default_factory=dict)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    terminal: TerminalConfig = Field(default_factory=TerminalConfig)

    def get_model(self, name: str) -> ModelConfig:
        if name not in self.models:
            raise KeyError(f"Model '{name}' not found in config. Available: {list(self.models.keys())}")
        return self.models[name]

    def get_role(self, name: str) -> RoleConfig:
        if name not in self.roles:
            raise KeyError(f"Role '{name}' not found in config. Available: {list(self.roles.keys())}")
        return self.roles[name]

    def resolve_role_model(self, role_name: str) -> tuple[RoleConfig, ModelConfig]:
        """Resolve a role to its (RoleConfig, ModelConfig) pair."""
        role = self.get_role(role_name)
        model = self.get_model(role.model)
        return role, model
