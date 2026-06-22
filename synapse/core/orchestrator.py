"""Orchestrator — uses an LLM to decompose complex tasks into subtasks."""

from __future__ import annotations

import json
import re

from synapse.config.schema import SynapseConfig
from synapse.core.task import Task, TaskTree
from synapse.models.registry import get_provider_for_model


ORCHESTRATOR_SYSTEM_PROMPT = """You are Synapse's task orchestrator. Your job is to analyze user requests and decompose complex tasks into subtasks.

When a task is simple (single-step, no decomposition needed), respond with:
{"mode": "single", "role": "coder", "prompt": "the task description"}

When a task requires multiple steps, respond with:
{
  "mode": "orchestrate",
  "tasks": [
    {
      "id": "task_1",
      "role": "coder",
      "prompt": "specific task description with clear requirements",
      "depends_on": []
    },
    {
      "id": "task_2",
      "role": "reviewer",
      "prompt": "review the code from task_1 for correctness",
      "depends_on": ["task_1"]
    }
  ]
}

Rules:
- Each task gets a unique ID (task_1, task_2, ...)
- Use depends_on to express dependencies (list of task IDs from earlier tasks)
- Tasks with no dependencies can run in parallel
- Choose the best role for each task from the available roles
- Keep prompts specific and actionable — include the exact requirements
- Maximum 5 tasks per plan
- If the user's request is simple, prefer mode: "single"

Available roles and their purposes:
{croles}

Respond with ONLY the JSON object, no other text."""


def _extract_json(text: str) -> dict:
    """Extract a JSON object from text that may have markdown fences or extra content."""
    # Try to find JSON in code fences
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence_match:
        text = fence_match.group(1)

    # Try to find the outermost { ... }
    brace_match = re.search(r"\{[\s\S]*\}", text)
    if brace_match:
        text = brace_match.group(0)

    return json.loads(text)


class Orchestrator:
    """Decomposes complex user requests into TaskTrees using an LLM."""

    def __init__(self, config: SynapseConfig, orchestrator_role: str = "orchestrator"):
        self.config = config

        # Resolve the orchestrator's own role
        if orchestrator_role in config.roles:
            self.role_config = config.roles[orchestrator_role]
            self.model_config = config.models[self.role_config.model]
        else:
            # Fallback: use first available model
            self.model_config = next(iter(config.models.values()))
            self.role_config = None

        self.provider = get_provider_for_model(self.model_config)

    async def plan(self, user_input: str) -> TaskTree:
        """Analyze user input and produce a TaskTree of subtasks.

        Returns a TaskTree with at least one task.
        """
        # Build the roles description for the prompt
        roles_desc = ""
        for name, role in self.config.roles.items():
            if name == "orchestrator":
                continue
            roles_desc += f"- {name}: {role.description or 'No description'}\n"

        system_prompt = ORCHESTRATOR_SYSTEM_PROMPT.format(croles=roles_desc)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]

        response = await self.provider.chat(
            messages=messages,
            temperature=0.3,
            max_tokens=2000,
        )

        plan_data = _extract_json(response.content)
        return self._build_tree(plan_data)

    def _build_tree(self, plan_data: dict) -> TaskTree:
        """Convert a plan JSON into a TaskTree."""
        tree = TaskTree()

        mode = plan_data.get("mode", "single")

        if mode == "single":
            task = Task(
                role=plan_data.get("role", "default"),
                prompt=plan_data.get("prompt", plan_data.get("task", "")),
            )
            tree.add_task(task)
            return tree

        # Orchestration mode
        tasks_data = plan_data.get("tasks", [])
        dependents: dict[str, list[str]] = {}  # task_id → list of tasks that depend on it

        for td in tasks_data:
            task = Task(
                id=td.get("id", ""),
                role=td.get("role", "default"),
                prompt=td.get("prompt", ""),
                depends_on=td.get("depends_on", []),
            )
            tree.add_task(task)

            for dep_id in task.depends_on:
                if dep_id not in dependents:
                    dependents[dep_id] = []
                dependents[dep_id].append(task.id)

        return tree
