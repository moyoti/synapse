"""Smoke tests for Synapse Phase 1."""

import pytest
from synapse.config.schema import SynapseConfig, ModelConfig, RoleConfig
from synapse.config.loader import _expand_env_vars, load_config
from synapse.models.registry import PROVIDER_REGISTRY, get_provider


class TestConfigSchema:
    def test_minimal_config(self):
        config = SynapseConfig(
            models={
                "test": ModelConfig(provider="compat", model="test-model"),
            },
            roles={
                "default": RoleConfig(model="test", system_prompt="You are helpful."),
            },
        )
        assert config.get_model("test").model == "test-model"
        role, model = config.resolve_role_model("default")
        assert role.system_prompt == "You are helpful."
        assert model.model == "test-model"

    def test_missing_model_raises(self):
        config = SynapseConfig()
        with pytest.raises(KeyError, match="nonexistent"):
            config.get_model("nonexistent")

    def test_missing_role_raises(self):
        config = SynapseConfig()
        with pytest.raises(KeyError, match="nonexistent"):
            config.get_role("nonexistent")

    def test_config_defaults(self):
        config = SynapseConfig()
        assert config.embedding.provider == "local"
        assert config.embedding.model == "all-MiniLM-L6-v2"
        assert config.memory.vector_top_k == 15
        assert config.execution.max_parallel_tasks == 3


class TestEnvVarExpansion:
    def test_simple_var(self):
        import os
        os.environ["TEST_SYNAPSE_VAR"] = "hello"
        result = _expand_env_vars("prefix_${TEST_SYNAPSE_VAR}_suffix")
        assert result == "prefix_hello_suffix"

    def test_var_with_default_used(self):
        result = _expand_env_vars("${NONEXISTENT_VAR:-default_value}")
        assert result == "default_value"

    def test_var_no_default_kept(self):
        result = _expand_env_vars("${NONEXISTENT_VAR}")
        assert result == "${NONEXISTENT_VAR}"


class TestProviderRegistry:
    def test_all_providers_registered(self):
        assert "deepseek" in PROVIDER_REGISTRY
        assert "anthropic" in PROVIDER_REGISTRY
        assert "compat" in PROVIDER_REGISTRY

    def test_get_provider(self):
        assert get_provider("compat").provider_name == "compat"

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            get_provider("nonexistent")
