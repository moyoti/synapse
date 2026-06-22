"""Tests for Phase 4 — Debate, Pipeline, Router."""

import pytest
from synapse.core.router import detect_mode, CollabMode
from synapse.core.debate import DebateEngine, DebateAgent, DebateResult
from synapse.core.pipeline import PipelineEngine, PipelineStage, PipelineResult
from synapse.config.schema import SynapseConfig, ModelConfig, RoleConfig


class TestRouter:
    def test_short_input_single(self):
        assert detect_mode("hello") == CollabMode.SINGLE

    def test_debate_detection(self):
        assert detect_mode("What do you think about this architecture? Is it a good idea?") == CollabMode.DEBATE

    def test_debate_question(self):
        assert detect_mode("Should I use PostgreSQL or MongoDB for this project?") == CollabMode.DEBATE

    def test_pipeline_detection(self):
        assert detect_mode("Analyze this CSV data and generate a report") == CollabMode.PIPELINE

    def test_orchestrate_detection(self):
        assert detect_mode(
            "Build a complete REST API with authentication, write tests, and review the code"
        ) == CollabMode.ORCHESTRATE

    def test_complex_question_orchestrate(self):
        # Long input should trigger orchestrate
        long_input = "I need to " + "implement and test " * 10
        assert detect_mode(long_input) == CollabMode.ORCHESTRATE

    def test_all_modes_unique(self):
        """Verify that each detection branch returns different modes."""
        inputs = {
            "hi": CollabMode.SINGLE,
            "what do you think about this idea and is it good?": CollabMode.DEBATE,
            "analyze this data and generate a report from the CSV": CollabMode.PIPELINE,
            "build a web app with backend and frontend and deploy it": CollabMode.ORCHESTRATE,
        }
        for text, expected in inputs.items():
            assert detect_mode(text) == expected, f"Failed for: {text}"


class TestDebateEngine:
    def test_engine_creation(self):
        config = SynapseConfig(
            models={"m": ModelConfig(provider="compat", model="test")},
            roles={
                "coder": RoleConfig(model="m"),
                "reviewer": RoleConfig(model="m"),
                "default": RoleConfig(model="m"),
            },
        )
        engine = DebateEngine(config)
        assert engine.config is config

    def test_auto_assign_perspectives(self):
        config = SynapseConfig(
            models={"m": ModelConfig(provider="compat", model="test")},
            roles={
                "coder": RoleConfig(model="m"),
                "default": RoleConfig(model="m"),
            },
        )
        engine = DebateEngine(config)
        agents = engine._auto_assign_perspectives("test question", num=3)
        assert len(agents) == 3
        assert all(isinstance(a, DebateAgent) for a in agents)
        # Each should have a unique perspective
        perspectives = [a.perspective for a in agents]
        assert len(perspectives) == len(set(perspectives))

    def test_create_debate_task(self):
        config = SynapseConfig(
            models={"m": ModelConfig(provider="compat", model="test")},
            roles={"coder": RoleConfig(model="m")},
        )
        engine = DebateEngine(config)
        agent = DebateAgent("Test", "coder", "security")
        task = engine._create_debate_task("Is this secure?", agent)
        assert task.role == "coder"
        assert "security" in task.prompt
        assert "Is this secure?" in task.prompt


class TestPipelineEngine:
    def test_engine_creation(self):
        config = SynapseConfig(
            models={"m": ModelConfig(provider="compat", model="test")},
            roles={
                "coder": RoleConfig(model="m"),
                "default": RoleConfig(model="m"),
                "reviewer": RoleConfig(model="m"),
            },
        )
        engine = PipelineEngine(config)
        assert engine.config is config

    def test_default_pipeline(self):
        config = SynapseConfig(
            models={"m": ModelConfig(provider="compat", model="test")},
            roles={"coder": RoleConfig(model="m"), "default": RoleConfig(model="m"), "reviewer": RoleConfig(model="m")},
        )
        engine = PipelineEngine(config)
        stages = engine._default_pipeline()
        assert len(stages) == 3
        assert stages[0].name == "Analyze"
        assert stages[1].name == "Write"
        assert stages[2].name == "Polish"

    def test_build_stage_prompt(self):
        config = SynapseConfig(
            models={"m": ModelConfig(provider="compat", model="test")},
            roles={"coder": RoleConfig(model="m")},
        )
        engine = PipelineEngine(config)
        stage = PipelineStage("Test", "coder", "Summarize this")
        prompt = engine._build_stage_prompt(stage, "INPUT DATA")
        assert "Summarize this" in prompt
        assert "INPUT DATA" in prompt
