"""LangGraph definitions for the orchestrator."""

from app.graphs.orchestrator import (
    create_orchestrator,
    get_orchestrator,
    clear_orchestrator_cache,
    ORCHESTRATOR_INSTRUCTION,
)
from app.graphs.state import (
    OrchestratorState,
    ResearcherState,
    DeepReasoningState,
    MentorMatcherState,
    # V2 State schemas
    ParallelResearchState,
    WorkerState,
    EvaluatorOptimizerState,
    MentorHubWorkflowState,
    SupervisorState,
    ResearchTask,
    ResearchResult,
    Evaluation,
    EvaluationCriteria,
)
from app.graphs.workflows import (
    create_parallel_researcher,
    create_evaluator_optimizer,
)
from app.graphs.orchestrator_v2 import (
    create_orchestrator_v2,
    get_orchestrator_v2,
    clear_orchestrator_v2_cache,
    run_with_optimal_workflow,
    ORCHESTRATOR_V2_PROMPT,
)
from app.graphs.routing import (
    classify_query,
    analyze_query,
    get_workflow_for_query_type,
    QueryType,
)

__all__ = [
    # Orchestrator
    "create_orchestrator",
    "get_orchestrator",
    "clear_orchestrator_cache",
    "ORCHESTRATOR_INSTRUCTION",
    # Legacy state schemas
    "OrchestratorState",
    "ResearcherState",
    "DeepReasoningState",
    "MentorMatcherState",
    # V2 State schemas
    "ParallelResearchState",
    "WorkerState",
    "EvaluatorOptimizerState",
    "MentorHubWorkflowState",
    "SupervisorState",
    "ResearchTask",
    "ResearchResult",
    "Evaluation",
    "EvaluationCriteria",
    # V2 Workflows
    "create_parallel_researcher",
    "create_evaluator_optimizer",
    # V2 Orchestrator
    "create_orchestrator_v2",
    "get_orchestrator_v2",
    "clear_orchestrator_v2_cache",
    "run_with_optimal_workflow",
    "ORCHESTRATOR_V2_PROMPT",
    # Routing
    "classify_query",
    "analyze_query",
    "get_workflow_for_query_type",
    "QueryType",
]
