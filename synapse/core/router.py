"""Router — automatically determines the best collaboration mode for a given input."""

from __future__ import annotations

from enum import Enum


class CollabMode(str, Enum):
    SINGLE = "single"
    ORCHESTRATE = "orchestrate"
    DEBATE = "debate"
    PIPELINE = "pipeline"


# Keywords and patterns for auto-detection
_ORCHESTRATE_PATTERNS = [
    "write and test", "write and review", "implement and document",
    "build a", "create a", "design and implement",
    "analyze and", "refactor and",
    "first", "then", "after that", "next step",
    "decompose", "break down",
]

_DEBATE_PATTERNS = [
    "what do you think", "is this a good idea",
    "pros and cons", "should i", "which is better",
    "evaluate", "compare", "trade-off", "tradeoff",
    "review this", "what are the risks", "is this safe",
    "critique", "analyze this decision", "opinion on",
    "thoughts on", "what would you recommend",
    "best approach", "better way",
]

_PIPELINE_PATTERNS = [
    "analyze this data", "analyze this file",
    "process this", "transform this",
    "summarize this document", "summarize and",
    "extract from", "convert to",
    "refine this", "polish this",
    "csv", "json", "data analysis", "report",
    "write a report", "generate a report",
]


def detect_mode(user_input: str) -> CollabMode:
    """Heuristically detect the best collaboration mode for the given input.

    Priority:
    1. Short/simple → SINGLE
    2. Contains debate keywords → DEBATE
    3. Contains pipeline keywords → PIPELINE
    4. Contains orchestration keywords or is complex → ORCHESTRATE
    5. Fallback → SINGLE
    """
    lower = user_input.lower()
    length = len(user_input)

    # Very short queries → single
    if length < 30:
        return CollabMode.SINGLE

    # Check debate patterns
    debate_score = sum(1 for p in _DEBATE_PATTERNS if p in lower)
    if debate_score >= 2 or (debate_score >= 1 and "?" in user_input):
        return CollabMode.DEBATE

    # Check pipeline patterns
    pipeline_score = sum(1 for p in _PIPELINE_PATTERNS if p in lower)
    if pipeline_score >= 2:
        return CollabMode.PIPELINE

    # Check orchestration patterns
    orch_score = sum(1 for p in _ORCHESTRATE_PATTERNS if p in lower)
    if orch_score >= 1 or length > 150:
        return CollabMode.ORCHESTRATE

    # Default to single
    return CollabMode.SINGLE
