"""Pipeline Engine — sequential multi-stage processing chain."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from synapse.config.schema import SynapseConfig
from synapse.core.task import Task, TaskStatus
from synapse.models.registry import get_provider_for_model


@dataclass
class PipelineStage:
    """A single stage in the processing pipeline."""

    name: str
    role: str         # Role to use for this stage
    instruction: str   # What this stage should do with the input
    model: str | None = None  # Optional model override


@dataclass
class PipelineResult:
    stages: list[dict] = field(default_factory=list)  # [{name, role, output}]
    final_output: str = ""


class PipelineEngine:
    """Executes a sequence of processing stages where each stage's output
    feeds into the next stage."""

    def __init__(self, config: SynapseConfig):
        self.config = config

    async def run(
        self,
        input_data: str,
        stages: list[PipelineStage] | None = None,
    ) -> PipelineResult:
        """Execute a pipeline of processing stages.

        Args:
            input_data: The initial input to process.
            stages: Pre-defined stages. If None, creates a default 3-stage pipeline.

        Returns:
            PipelineResult with each stage's output and the final result.
        """
        if stages is None:
            stages = self._default_pipeline()

        result = PipelineResult()
        current_input = input_data

        for stage in stages:
            # Build prompt for this stage
            prompt = self._build_stage_prompt(stage, current_input)

            # Execute
            stage_output = await self._execute_stage(stage, prompt)

            result.stages.append({
                "name": stage.name,
                "role": stage.role,
                "output": stage_output,
            })

            # Feed output as input to next stage
            current_input = stage_output

        result.final_output = current_input
        return result

    def _build_stage_prompt(self, stage: PipelineStage, input_data: str) -> str:
        """Build the prompt for a pipeline stage."""
        return f"""## Your Task: {stage.instruction}

## Input Data
{input_data}

## Instructions
Process the input data according to your task. Output ONLY the processed result.
Do not add explanations, justifications, or meta-commentary.
"""

    async def _execute_stage(self, stage: PipelineStage, prompt: str) -> str:
        """Execute a single pipeline stage."""
        # Resolve role and model
        role_name = stage.role
        if role_name not in self.config.roles:
            role_name = next(iter(self.config.roles.keys()))

        role_config = self.config.roles[role_name]
        model_name = stage.model or role_config.model

        if model_name not in self.config.models:
            model_name = next(iter(self.config.models.keys()))

        model_config = self.config.models[model_name]
        provider = get_provider_for_model(model_config)

        temperature = model_config.default_params.temperature
        max_tokens = model_config.default_params.max_tokens

        messages = [
            {"role": "system", "content": role_config.system_prompt},
            {"role": "user", "content": prompt},
        ]

        try:
            response = await provider.chat(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.content
        except Exception as e:
            return f"(Pipeline stage '{stage.name}' failed: {e})"

    def _default_pipeline(self) -> list[PipelineStage]:
        """Create a default 3-stage pipeline: analyze → write → polish."""
        return [
            PipelineStage(
                name="Analyze",
                role="coder",
                instruction="Analyze the input data thoroughly. Extract key findings, patterns, and insights. Output a structured analysis.",
            ),
            PipelineStage(
                name="Write",
                role="default",
                instruction="Based on the analysis, write a comprehensive report or solution. Make it professional and well-structured.",
            ),
            PipelineStage(
                name="Polish",
                role="reviewer",
                instruction="Review and polish the output. Fix any issues, improve clarity, and ensure it's ready for final delivery.",
            ),
        ]
