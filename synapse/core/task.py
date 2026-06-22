"""Task data models — the atomic unit of work in Synapse."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TaskStatus(str, Enum):
    PENDING = "pending"        # Dependencies not yet satisfied
    READY = "ready"             # Dependencies met, waiting to be scheduled
    RUNNING = "running"         # Currently executing
    COMPLETED = "completed"     # Successfully finished
    FAILED = "failed"           # Execution failed
    RETRYING = "retrying"       # Being retried
    CANCELLED = "cancelled"     # Cancelled by user or due to dependency failure


@dataclass
class Task:
    """A single unit of work assigned to a specific role/model."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    parent_id: str | None = None       # Parent task ID (for tree structure)
    role: str = "default"              # Which role to use
    prompt: str = ""                   # The task description / prompt
    context: str | None = None         # Additional context (e.g., output of dependency tasks)
    depends_on: list[str] = field(default_factory=list)  # Task IDs this depends on
    status: TaskStatus = TaskStatus.PENDING
    result: str | None = None          # Output text from the model
    error: str | None = None           # Error message if failed
    retry_count: int = 0
    max_retries: int = 2
    timeout: int = 300
    metadata: dict[str, Any] = field(default_factory=dict)  # Extra info (model used, tokens, etc.)

    @property
    def is_terminal(self) -> bool:
        """Task is done (success, failure, or cancelled)."""
        return self.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED)

    @property
    def can_retry(self) -> bool:
        return self.status == TaskStatus.FAILED and self.retry_count < self.max_retries

    def to_messages(self, system_prompt: str = "") -> list[dict[str, str]]:
        """Build the messages list for the model call."""
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        user_content = self.prompt
        if self.context:
            user_content = f"## Context from previous tasks\n\n{self.context}\n\n---\n\n## Your task\n\n{self.prompt}"

        messages.append({"role": "user", "content": user_content})
        return messages


@dataclass
class TaskTree:
    """A tree of tasks with dependencies, representing a decomposed plan."""

    tasks: dict[str, Task] = field(default_factory=dict)
    root_id: str | None = None

    def add_task(self, task: Task) -> None:
        self.tasks[task.id] = task
        if self.root_id is None:
            self.root_id = task.id

    def get_ready_tasks(self) -> list[Task]:
        """Return tasks whose dependencies are all COMPLETED and are PENDING."""
        ready = []
        for task in self.tasks.values():
            if task.status != TaskStatus.PENDING:
                continue
            if all(
                self.tasks[dep_id].status == TaskStatus.COMPLETED
                for dep_id in task.depends_on
            ):
                ready.append(task)
        return ready

    def all_done(self) -> bool:
        """True if all tasks are terminal."""
        return all(t.is_terminal for t in self.tasks.values())

    def all_success(self) -> bool:
        """True if all tasks completed successfully."""
        return all(t.status == TaskStatus.COMPLETED for t in self.tasks.values())

    def has_failures(self) -> bool:
        """True if any task failed."""
        return any(t.status == TaskStatus.FAILED for t in self.tasks.values())

    def get_results_map(self) -> dict[str, str]:
        """Get a map of task ID → result for all completed tasks."""
        return {
            tid: t.result
            for tid, t in self.tasks.items()
            if t.status == TaskStatus.COMPLETED and t.result
        }
