"""Debate Engine — parallel multi-agent debate with perspective assignment and synthesis."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from synapse.config.schema import SynapseConfig
from synapse.core.executor import Executor
from synapse.core.task import Task, TaskStatus


DEBATE_SYNTHESIS_PROMPT = """You are a debate synthesizer. You have received independent analyses from {n} colleagues, each examining the same question from a different perspective.

## Original Question
{question}

## Colleague Analyses
{analyses}

## Your Task
Synthesize these perspectives into a comprehensive report with these sections:

### 1. Consensus (共识)
Points where ≥2 colleagues agree. These are high-confidence findings.

### 2. Unique Insights (独到见解)
Valuable points raised by only one colleague that deserve attention.

### 3. Conflicts (冲突)
If any colleagues disagree on important points, surface these productively.

### 4. Recommendation (综合建议)
Your synthesis — the most important takeaway, combining multiple perspectives.

Format as a clear Markdown report. Be concise but thorough."""


@dataclass
class DebateAgent:
    """A single perspective in the debate."""

    name: str
    role: str
    perspective: str  # The specific angle/expertise (e.g., "security", "performance")
    model: str | None = None  # Override model if desired


@dataclass
class DebateResult:
    question: str
    perspectives: list[dict] = field(default_factory=list)  # [{name, role, answer}]
    synthesis: str = ""
    consensus: list[str] = field(default_factory=list)
    unique_insights: list[str] = field(default_factory=list)


class DebateEngine:
    """Orchestrates a multi-agent debate: assign perspectives, run in parallel, synthesize."""

    def __init__(self, config: SynapseConfig):
        self.config = config

    async def debate(
        self,
        question: str,
        agents: list[DebateAgent] | None = None,
        num_perspectives: int = 3,
    ) -> DebateResult:
        """Run a debate on the given question.

        Args:
            question: The question to debate.
            agents: Pre-configured debate agents. If None, auto-assigns perspectives.
            num_perspectives: Number of perspectives to auto-assign (if agents not provided).

        Returns:
            DebateResult with all perspective answers and synthesis.
        """
        if agents is None:
            agents = self._auto_assign_perspectives(question, num_perspectives)

        # Step 1: Run all agents in parallel
        perspective_tasks = []
        for agent in agents:
            task = self._create_debate_task(question, agent)
            perspective_tasks.append(task)

        executor = Executor(self.config)
        results = await asyncio.gather(
            *(executor.execute_with_retry(t) for t in perspective_tasks),
            return_exceptions=True,
        )

        # Collect answers
        perspectives = []
        for i, result in enumerate(results):
            agent = agents[i]
            if isinstance(result, Task) and result.status == TaskStatus.COMPLETED:
                perspectives.append({
                    "name": agent.name,
                    "role": agent.role,
                    "perspective": agent.perspective,
                    "answer": result.result or "(no response)",
                })
            else:
                err = getattr(result, 'error', str(result)) if not isinstance(result, Task) else result.error
                perspectives.append({
                    "name": agent.name,
                    "role": agent.role,
                    "perspective": agent.perspective,
                    "answer": f"(Error: {err})",
                })

        # Step 2: Synthesize
        synthesis = await self._synthesize(question, perspectives)

        return DebateResult(
            question=question,
            perspectives=perspectives,
            synthesis=synthesis,
        )

    def _create_debate_task(self, question: str, agent: DebateAgent) -> Task:
        """Create a task for a debate agent with a perspective prompt."""

        prompt = f"""You are analyzing the following question from the perspective of: **{agent.perspective}**

## Your Role
{agent.role}

## Your Perspective
Focus specifically on {agent.perspective} aspects. You don't need to cover everything — go deep on what matters from your angle.

## Question
{question}

## Instructions
1. Analyze the question from your specific perspective
2. Identify key issues, risks, opportunities, or insights
3. Be opinionated — don't try to be neutral
4. Structure your response with clear points
5. Keep it concise (2-4 paragraphs max)
"""

        role_name = agent.role
        if role_name not in self.config.roles:
            role_name = self._find_best_role()

        model_override = agent.model
        if model_override:
            # We'll apply the model override via Task metadata
            pass

        task = Task(
            role=role_name,
            prompt=prompt,
            metadata={"model_override": model_override} if model_override else {},
        )

        # If model override specified, swap the role's model
        if model_override and model_override in self.config.models:
            # HACK: temporarily swap model in role config (this is safe because roles are copied)
            pass  # Executor will handle model override

        return task

    async def _synthesize(self, question: str, perspectives: list[dict]) -> str:
        """Synthesize debate perspectives into a final report."""
        # Format analyses
        analyses = ""
        for i, p in enumerate(perspectives, 1):
            analyses += f"\n### Colleague {i}: {p['name']} ({p['perspective']})\n"
            analyses += f"{p['answer']}\n"

        prompt = DEBATE_SYNTHESIS_PROMPT.format(
            n=len(perspectives),
            question=question,
            analyses=analyses,
        )

        # Use orchestrator model for synthesis
        synthesizer_role = self.config.roles.get("orchestrator")
        if synthesizer_role:
            model_name = synthesizer_role.model
        else:
            model_name = next(iter(self.config.models.keys()))

        model_config = self.config.models[model_name]
        from synapse.models.registry import get_provider_for_model
        provider = get_provider_for_model(model_config)

        response = await provider.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=4096,
        )

        return response.content

    def _auto_assign_perspectives(self, question: str, num: int = 3) -> list[DebateAgent]:
        """Auto-assign debate perspectives based on the question type."""
        # Default perspective palette
        default_perspectives = [
            DebateAgent("Critic-A", "coder", "technical correctness and implementation feasibility"),
            DebateAgent("Critic-B", "reviewer", "code quality, security, and maintainability"),
            DebateAgent("Strategist", "default", "business value, user experience, and strategic fit"),
            DebateAgent("Optimizer", "coder", "performance, scalability, and efficiency"),
            DebateAgent("Skeptic", "default", "risks, edge cases, and potential failure modes"),
        ]

        # Pick diverse perspectives
        selected = default_perspectives[:num]
        return selected

    def _find_best_role(self) -> str:
        """Find the best available role for debate tasks."""
        preferred = ["coder", "reviewer", "default"]
        for r in preferred:
            if r in self.config.roles:
                return r
        return next(iter(self.config.roles.keys()))
