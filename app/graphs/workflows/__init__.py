"""
Workflow patterns for LangGraph Orchestrator v2.

Provides specialized workflow graphs:
- parallel_researcher: Parallel data gathering with Send API
- evaluator_optimizer: Quality gates with retry cycles
- mentor_hub_workflow: Live API data access
"""

from __future__ import annotations

from app.graphs.workflows.parallel_researcher import (
    create_parallel_researcher,
    spawn_researchers,
)
from app.graphs.workflows.evaluator_optimizer import (
    create_evaluator_optimizer,
)

__all__ = [
    "create_parallel_researcher",
    "spawn_researchers",
    "create_evaluator_optimizer",
]
