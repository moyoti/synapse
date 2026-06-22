"""Aggregator — combines results from multiple tasks into a final output."""

from __future__ import annotations

from synapse.config.schema import SynapseConfig
from synapse.core.task import TaskStatus, TaskTree
from synapse.models.registry import get_provider_for_model


AGGREGATOR_PROMPT = """You are Synapse's result aggregator. You have received the outputs from multiple subtasks that were executed to fulfill a user's request.

Your job is to synthesize these results into a single, coherent final response.

## Original user request
{user_input}

## Subtask results
{task_results}

## Instructions
- Combine the results into a natural, flowing response
- If there are code outputs, integrate them appropriately
- If the reviewer found issues, mention them and provide the revised approach
- Use the same language as the original user request
- Be concise but complete
"""


class Aggregator:
    """Combines multiple task results into a final user-facing response."""

    def __init__(self, config: SynapseConfig):
        self.config = config

    async def aggregate(
        self,
        user_input: str,
        tree: TaskTree,
    ) -> str:
        """Aggregate task results into a final response.

        If there's only one task, returns its result directly.
        """
        results_map = tree.get_results_map()

        if len(results_map) == 0:
            failures = [t for t in tree.tasks.values() if t.status == TaskStatus.FAILED]
            if failures:
                errors = "\n".join(f"- {t.id} ({t.role}): {t.error}" for t in failures)
                return f"All tasks failed:\n{errors}"
            return "No results to aggregate."

        if len(results_map) == 1:
            return next(iter(results_map.values()))

        # Multiple results — use LLM to aggregate
        # Use the orchestrator model for aggregation
        orchestrator_role = self.config.roles.get("orchestrator")
        if orchestrator_role:
            model_config = self.config.models[orchestrator_role.model]
        else:
            model_config = next(iter(self.config.models.values()))

        provider = get_provider_for_model(model_config)

        task_results_text = ""
        for tid, result in results_map.items():
            task = tree.tasks[tid]
            task_results_text += f"\n### Task: {tid} (Role: {task.role})\n{result}\n"

        prompt = AGGREGATOR_PROMPT.format(
            user_input=user_input,
            task_results=task_results_text,
        )

        response = await provider.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=4096,
        )

        return response.content
