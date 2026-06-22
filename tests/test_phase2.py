"""Tests for Phase 2 — Core modules (Task, Executor, Orchestrator, Scheduler, Aggregator)."""

import pytest
from synapse.core.task import Task, TaskTree, TaskStatus
from synapse.config.schema import SynapseConfig, ModelConfig, RoleConfig


class TestTask:
    def test_task_creation(self):
        task = Task(role="coder", prompt="Write a function")
        assert task.id
        assert task.status == TaskStatus.PENDING
        assert task.depends_on == []
        assert not task.is_terminal

    def test_task_terminal_states(self):
        task = Task()
        task.status = TaskStatus.COMPLETED
        assert task.is_terminal
        task.status = TaskStatus.FAILED
        assert task.is_terminal
        task.status = TaskStatus.CANCELLED
        assert task.is_terminal

    def test_task_can_retry(self):
        task = Task()
        task.status = TaskStatus.FAILED
        assert task.can_retry is True
        task.retry_count = 2  # max_retries default is 2
        assert task.can_retry is False

    def test_task_to_messages(self):
        task = Task(prompt="Hello", context="Previous: 42")
        msgs = task.to_messages(system_prompt="You are helpful.")
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert "Previous: 42" in msgs[1]["content"]
        assert "Hello" in msgs[1]["content"]

    def test_task_to_messages_no_context(self):
        task = Task(prompt="Hello")
        msgs = task.to_messages(system_prompt="")
        assert len(msgs) == 1
        assert msgs[0]["content"] == "Hello"


class TestTaskTree:
    def test_empty_tree(self):
        tree = TaskTree()
        assert tree.all_done() is True
        assert tree.all_success() is True

    def test_add_tasks(self):
        tree = TaskTree()
        t1 = Task(id="t1", prompt="Task 1")
        t2 = Task(id="t2", prompt="Task 2", depends_on=["t1"])
        tree.add_task(t1)
        tree.add_task(t2)

        assert len(tree.tasks) == 2
        assert tree.root_id == "t1"

    def test_get_ready_tasks(self):
        tree = TaskTree()
        t1 = Task(id="t1", prompt="Task 1")
        t2 = Task(id="t2", prompt="Task 2", depends_on=["t1"])
        t3 = Task(id="t3", prompt="Task 3")
        tree.add_task(t1)
        tree.add_task(t2)
        tree.add_task(t3)

        ready = tree.get_ready_tasks()
        assert len(ready) == 2  # t1 and t3 have no deps
        ready_ids = {t.id for t in ready}
        assert ready_ids == {"t1", "t3"}

    def test_get_ready_after_completion(self):
        tree = TaskTree()
        t1 = Task(id="t1", prompt="Task 1", status=TaskStatus.COMPLETED)
        t2 = Task(id="t2", prompt="Task 2", depends_on=["t1"])
        tree.add_task(t1)
        tree.add_task(t2)

        ready = tree.get_ready_tasks()
        assert len(ready) == 1
        assert ready[0].id == "t2"

    def test_all_done(self):
        tree = TaskTree()
        t1 = Task(id="t1", status=TaskStatus.COMPLETED)
        t2 = Task(id="t2", status=TaskStatus.FAILED)
        tree.add_task(t1)
        tree.add_task(t2)

        assert tree.all_done() is True
        assert tree.all_success() is False
        assert tree.has_failures() is True

    def test_get_results_map(self):
        tree = TaskTree()
        t1 = Task(id="t1", status=TaskStatus.COMPLETED, result="result1")
        t2 = Task(id="t2", status=TaskStatus.FAILED, result="result2")
        tree.add_task(t1)
        tree.add_task(t2)

        results = tree.get_results_map()
        assert len(results) == 1
        assert results["t1"] == "result1"


class TestExecutorMinimal:
    """Minimal executor test (no actual model calls)."""

    def test_executor_resolves_config(self):
        config = SynapseConfig(
            models={
                "test-model": ModelConfig(provider="compat", model="gpt-test"),
            },
            roles={
                "coder": RoleConfig(model="test-model", system_prompt="You are a coder."),
            },
        )
        from synapse.core.executor import Executor
        executor = Executor(config)
        # Just verify it was created without error
        assert executor.config is config


class TestOrchestratorJsonExtraction:
    """Test the JSON extraction logic."""

    def test_extract_plain_json(self):
        from synapse.core.orchestrator import _extract_json
        result = _extract_json('{"mode": "single", "role": "coder", "prompt": "test"}')
        assert result["mode"] == "single"
        assert result["role"] == "coder"

    def test_extract_fenced_json(self):
        from synapse.core.orchestrator import _extract_json
        text = '''Here's the plan:
```json
{"mode": "orchestrate", "tasks": [{"id": "t1", "role": "coder", "prompt": "do it", "depends_on": []}]}
```
That's it.'''
        result = _extract_json(text)
        assert result["mode"] == "orchestrate"
        assert len(result["tasks"]) == 1

    def test_build_tree_single(self):
        from synapse.core.orchestrator import Orchestrator
        from synapse.core.task import TaskTree

        config = SynapseConfig(
            models={"m": ModelConfig(provider="compat", model="gpt-test")},
            roles={"coder": RoleConfig(model="m")},
        )
        orch = Orchestrator(config)
        tree = orch._build_tree({
            "mode": "single",
            "role": "coder",
            "prompt": "Write hello world",
        })
        assert isinstance(tree, TaskTree)
        assert len(tree.tasks) == 1

    def test_build_tree_orchestrate(self):
        from synapse.core.orchestrator import Orchestrator

        config = SynapseConfig(
            models={"m": ModelConfig(provider="compat", model="gpt-test")},
            roles={"coder": RoleConfig(model="m"), "reviewer": RoleConfig(model="m")},
        )
        orch = Orchestrator(config)
        tree = orch._build_tree({
            "mode": "orchestrate",
            "tasks": [
                {"id": "t1", "role": "coder", "prompt": "Write code", "depends_on": []},
                {"id": "t2", "role": "reviewer", "prompt": "Review code", "depends_on": ["t1"]},
            ],
        })
        assert len(tree.tasks) == 2
        assert tree.tasks["t2"].depends_on == ["t1"]


class TestScheduler:
    def test_scheduler_creation(self):
        from synapse.core.scheduler import Scheduler

        config = SynapseConfig(
            models={"m": ModelConfig(provider="compat", model="gpt-test")},
            roles={"coder": RoleConfig(model="m")},
        )
        scheduler = Scheduler(config)
        assert scheduler.max_parallel == 3


class TestAggregator:
    def test_aggregator_creation(self):
        from synapse.core.aggregator import Aggregator

        config = SynapseConfig(
            models={"m": ModelConfig(provider="compat", model="gpt-test")},
            roles={"orchestrator": RoleConfig(model="m")},
        )
        agg = Aggregator(config)
        assert agg.config is config
