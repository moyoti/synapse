"""Executor — runs a single task against a model."""

from __future__ import annotations

from synapse.config.schema import SynapseConfig
from synapse.core.task import Task, TaskStatus
from synapse.models.base import BaseProvider
from synapse.models.registry import get_provider_for_model


class Executor:
    """Executes a single Task by calling the bound model."""

    def __init__(self, config: SynapseConfig):
        self.config = config

    async def execute(self, task: Task) -> Task:
        """Execute a single task. Returns the task with result/error set.

        Raises no exceptions — failures are recorded in task.error.
        """
        # Resolve role and model
        try:
            role_config = self.config.get_role(task.role)
        except KeyError:
            task.status = TaskStatus.FAILED
            task.error = f"Role '{task.role}' not found in config"
            return task

        try:
            model_config = self.config.get_model(role_config.model)
        except KeyError:
            task.status = TaskStatus.FAILED
            task.error = f"Model '{role_config.model}' not found in config"
            return task

        provider = get_provider_for_model(model_config)
        temperature = model_config.default_params.temperature
        max_tokens = model_config.default_params.max_tokens

        messages = task.to_messages(system_prompt=role_config.system_prompt)

        task.status = TaskStatus.RUNNING
        task.retry_count += 1

        try:
            response = await provider.chat(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            task.result = response.content
            task.status = TaskStatus.COMPLETED
            task.metadata["model"] = response.model
            task.metadata["tokens"] = response.total_tokens
        except Exception as e:
            task.error = str(e)
            if task.can_retry:
                task.status = TaskStatus.RETRYING
            else:
                task.status = TaskStatus.FAILED

        return task

    async def execute_with_retry(self, task: Task) -> Task:
        """Execute with automatic retry on failure."""
        while True:
            task = await self.execute(task)
            if task.status == TaskStatus.COMPLETED:
                return task
            if task.status == TaskStatus.RETRYING:
                continue
            # FAILED or CANCELLED
            return task
