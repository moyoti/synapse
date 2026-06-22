"""Scheduler — executes tasks in dependency order with parallelism."""

from __future__ import annotations

import asyncio
from collections import deque

from synapse.config.schema import SynapseConfig
from synapse.core.executor import Executor
from synapse.core.task import Task, TaskStatus, TaskTree


class Scheduler:
    """Executes a TaskTree respecting dependencies with parallel execution."""

    def __init__(self, config: SynapseConfig):
        self.config = config
        self.executor = Executor(config)
        self.max_parallel = config.execution.max_parallel_tasks

    async def run(self, tree: TaskTree) -> TaskTree:
        """Execute all tasks in the tree in dependency order.

        Returns the tree with results filled in.
        """
        pending = list(tree.tasks.values())
        running: set[str] = set()
        completed: set[str] = set()

        while pending or running:
            # Find tasks ready to execute
            ready = []
            for task in pending:
                if all(dep in completed for dep in task.depends_on):
                    ready.append(task)

            # Move ready tasks from pending to running (up to max_parallel)
            for task in ready[: self.max_parallel - len(running)]:
                pending.remove(task)
                running.add(task.id)

                # Build context from completed dependencies
                if task.depends_on:
                    context_parts = []
                    for dep_id in task.depends_on:
                        dep_task = tree.tasks[dep_id]
                        if dep_task.result:
                            context_parts.append(
                                f"### Output from task '{dep_id}' ({dep_task.role})\n{dep_task.result}"
                            )
                    task.context = "\n\n".join(context_parts)

            # Execute all running tasks in parallel
            if running:
                tasks_to_run = [tree.tasks[tid] for tid in running]
                results = await asyncio.gather(
                    *(self.executor.execute_with_retry(t) for t in tasks_to_run),
                    return_exceptions=True,
                )

                for task_or_exc in results:
                    if isinstance(task_or_exc, BaseException):
                        # This shouldn't happen since execute_with_retry catches internally
                        continue
                    completed.add(task_or_exc.id)
                    running.discard(task_or_exc.id)

                    # Cascade failures: cancel tasks that depend on a failed task
                    if task_or_exc.status == TaskStatus.FAILED:
                        for t in pending:
                            if task_or_exc.id in t.depends_on:
                                t.status = TaskStatus.CANCELLED
                                t.error = f"Cancelled: dependency '{task_or_exc.id}' failed"
                                pending.remove(t)

            # If nothing is running and nothing is ready, we're stuck or done
            if not running and not ready:
                break

        return tree
