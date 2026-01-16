"""
Evaluator-Optimizer Workflow with Quality Gates.

Implements a cycle pattern where research results are evaluated
against quality criteria. If thresholds aren't met, the workflow
retries with refined queries.

Flow:
1. Research → 2. Synthesize → 3. Evaluate → (pass?) → Finalize
                                    ↓ (fail)
                              Retry with refined queries (max 2 retries)
"""

from __future__ import annotations

import logging
from typing import Any, Literal, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from app.config import get_settings
from app.graphs.state import (
    EvaluatorOptimizerState,
    ResearchTask,
    ResearchResult,
    Evaluation,
    EvaluationCriteria,
)

logger = logging.getLogger(__name__)


# =============================================================================
# EVALUATION PROMPTS
# =============================================================================

EVALUATOR_SYSTEM_PROMPT = """You are a quality evaluator for research results.

Evaluate the research results against these criteria:
1. **Sufficient Data**: Are there at least 3 meaningful results?
2. **Confidence**: Is the overall confidence >= 0.6?
3. **Query Coverage**: Does the answer address the original question?
4. **Citations**: Are there sources for the claims?

Respond in this exact JSON format:
{
    "has_sufficient_data": true/false,
    "confidence_threshold_met": true/false,
    "answers_query": true/false,
    "has_citations": true/false,
    "passed": true/false,
    "feedback": "Brief explanation",
    "suggested_refinements": ["refinement 1", "refinement 2"] or []
}
"""

REFINEMENT_SYSTEM_PROMPT = """You are a search query optimizer.

Based on the evaluation feedback, generate refined search queries
that will better answer the original question.

Return 3-5 refined queries that:
1. Address the gaps identified in the feedback
2. Use different phrasings or synonyms
3. Target specific aspects that were missing

Respond with just the queries, one per line.
"""


# =============================================================================
# NODES
# =============================================================================

def initial_research(state: EvaluatorOptimizerState) -> dict[str, Any]:
    """
    Create initial research tasks.

    This is similar to the planner in parallel_researcher but
    simplified for the evaluator-optimizer flow.
    """
    query = state["query"]

    # Create diverse research tasks
    tasks = [
        ResearchTask(
            id="task_1",
            type="text",
            query=query,
            focus="direct text search",
            priority="high",
            config={"top_k": 15},
        ),
        ResearchTask(
            id="task_2",
            type="entity",
            query=query,
            focus="entity lookup",
            priority="high",
            config=None,
        ),
        ResearchTask(
            id="task_3",
            type="summary",
            query=query,
            focus="summary search",
            priority="medium",
            config=None,
        ),
    ]

    return {
        "research_tasks": tasks,
        "retry_count": state.get("retry_count", 0),
        "max_retries": state.get("max_retries", 2),
    }


def spawn_research_workers(state: EvaluatorOptimizerState) -> list[Send]:
    """Spawn parallel workers for research tasks."""
    return [
        Send("research_worker", {"task": task})
        for task in state["research_tasks"]
    ]


async def research_worker(state: dict) -> dict[str, Any]:
    """Execute a single research task."""
    from app.tools.graph_tools import find_entity, search_text
    from app.tools.cognee_tools import search_summaries, search_rag

    task = state["task"]

    try:
        if task["type"] == "entity":
            result = await find_entity.ainvoke({"name": task["query"]})
            success = result.get("found", False)
            confidence = 0.8 if success else 0.3
        elif task["type"] == "text":
            result = await search_text.ainvoke({
                "query": task["query"],
                "top_k": task.get("config", {}).get("top_k", 10) if task.get("config") else 10
            })
            success = bool(result.get("results"))
            confidence = 0.7 if success else 0.3
        elif task["type"] == "summary":
            result = await search_summaries.ainvoke({"query": task["query"]})
            success = bool(result.get("results"))
            confidence = 0.6 if success else 0.3
        else:
            result = await search_rag.ainvoke({"query": task["query"]})
            success = True
            confidence = 0.5

        return {
            "research_results": [ResearchResult(
                task_id=task["id"],
                task_type=task["type"],
                success=success,
                data=result,
                confidence=confidence,
                sources=[f"cognee:{task['type']}"],
                error=None,
                execution_time_ms=0,
            )]
        }

    except Exception as e:
        logger.error(f"Research worker failed: {e}")
        return {
            "research_results": [ResearchResult(
                task_id=task["id"],
                task_type=task["type"],
                success=False,
                data=None,
                confidence=0.0,
                sources=[],
                error=str(e),
                execution_time_ms=0,
            )]
        }


async def synthesizer(state: EvaluatorOptimizerState) -> dict[str, Any]:
    """Synthesize research results into an answer."""
    results = state["research_results"]
    query = state["query"]

    successful = [r for r in results if r["success"]]

    if not successful:
        return {
            "final_answer": "I couldn't find sufficient information to answer this question.",
            "confidence": 0.0,
        }

    # Calculate confidence
    avg_confidence = sum(r["confidence"] for r in successful) / len(successful)

    # Use LLM to synthesize
    settings = get_settings()
    llm = ChatOpenAI(
        model=settings.default_research_model,
        api_key=settings.openai_api_key,
    )

    # Format results for synthesis
    results_text = ""
    for r in successful:
        if r["data"]:
            results_text += f"\n[{r['task_type']}]: {str(r['data'])[:500]}\n"

    synthesis_prompt = f"""Based on these research results, provide a comprehensive answer to: {query}

Research Results:
{results_text}

Provide a well-structured answer with citations where possible."""

    response = await llm.ainvoke([
        SystemMessage(content="You synthesize research results into clear, comprehensive answers."),
        HumanMessage(content=synthesis_prompt),
    ])

    return {
        "final_answer": response.content,
        "confidence": avg_confidence,
    }


async def evaluator(state: EvaluatorOptimizerState) -> dict[str, Any]:
    """Evaluate the quality of the synthesized answer."""
    results = state["research_results"]
    answer = state.get("final_answer", "")
    query = state["query"]
    confidence = state.get("confidence", 0.0)

    successful_results = [r for r in results if r["success"]]

    # Build evaluation criteria
    criteria = EvaluationCriteria(
        has_sufficient_data=len(successful_results) >= 3,
        confidence_threshold_met=confidence >= 0.6,
        answers_query=len(answer) > 100,  # Simple heuristic
        has_citations=any(r.get("sources") for r in successful_results),
    )

    passed = all([
        criteria["has_sufficient_data"],
        criteria["confidence_threshold_met"],
        criteria["answers_query"],
    ])

    # Generate feedback
    feedback_parts = []
    refinements = []

    if not criteria["has_sufficient_data"]:
        feedback_parts.append("Not enough data sources")
        refinements.append("Try broader search terms")
        refinements.append("Search for related concepts")

    if not criteria["confidence_threshold_met"]:
        feedback_parts.append(f"Low confidence ({confidence:.2f})")
        refinements.append("Search for more specific information")

    if not criteria["answers_query"]:
        feedback_parts.append("Answer may not fully address query")
        refinements.append("Focus search on specific aspects of the question")

    evaluation = Evaluation(
        criteria=criteria,
        passed=passed,
        feedback="; ".join(feedback_parts) if feedback_parts else "Quality criteria met",
        suggested_refinements=refinements,
    )

    logger.info(f"Evaluation: passed={passed}, confidence={confidence:.2f}")

    return {"evaluation": evaluation}


def should_retry(state: EvaluatorOptimizerState) -> Literal["retry", "finalize"]:
    """Decide whether to retry research or finalize."""
    evaluation = state.get("evaluation")
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)

    if evaluation and evaluation["passed"]:
        logger.info("Evaluation passed, finalizing")
        return "finalize"

    if retry_count >= max_retries:
        logger.info(f"Max retries ({max_retries}) reached, finalizing")
        return "finalize"

    logger.info(f"Evaluation failed, retrying (attempt {retry_count + 1}/{max_retries})")
    return "retry"


async def retry_planner(state: EvaluatorOptimizerState) -> dict[str, Any]:
    """Generate refined research tasks based on evaluation feedback."""
    evaluation = state.get("evaluation")
    query = state["query"]
    retry_count = state.get("retry_count", 0)

    refinements = evaluation.get("suggested_refinements", []) if evaluation else []

    # Generate refined tasks
    refined_tasks = []

    # Add tasks based on refinements
    for i, refinement in enumerate(refinements[:3]):
        refined_tasks.append(ResearchTask(
            id=f"retry_{retry_count + 1}_task_{i + 1}",
            type="text",
            query=f"{query} {refinement}",
            focus=f"refined search: {refinement[:30]}",
            priority="high",
            config={"top_k": 15},
        ))

    # Add a RAG task for comprehensive coverage
    refined_tasks.append(ResearchTask(
        id=f"retry_{retry_count + 1}_rag",
        type="summary",
        query=query,
        focus="RAG comprehensive search",
        priority="high",
        config={"use_rag": True},
    ))

    logger.info(f"Retry planner created {len(refined_tasks)} refined tasks")

    return {
        "research_tasks": refined_tasks,
        "retry_count": retry_count + 1,
    }


def finalize(state: EvaluatorOptimizerState) -> dict[str, Any]:
    """Finalize the answer (approved or best-effort)."""
    answer = state.get("final_answer", "")
    evaluation = state.get("evaluation")
    confidence = state.get("confidence", 0.0)

    if evaluation and evaluation["passed"]:
        return {"approved_answer": answer}
    else:
        # Return best-effort answer with disclaimer if confidence is low
        if confidence < 0.5:
            answer = f"*Note: This answer may be incomplete.*\n\n{answer}"
        return {"approved_answer": answer}


# =============================================================================
# GRAPH BUILDER
# =============================================================================

def create_evaluator_optimizer(
    checkpointer: Optional[Any] = None,
) -> CompiledStateGraph:
    """
    Create the evaluator-optimizer workflow graph.

    Flow:
    1. Initial research creates tasks
    2. Workers execute in parallel
    3. Synthesizer combines results
    4. Evaluator checks quality
    5. If failed and retries left → Retry planner → Workers
    6. If passed or max retries → Finalize

    Args:
        checkpointer: Optional checkpointer for state persistence

    Returns:
        Compiled evaluator-optimizer graph
    """
    builder = StateGraph(EvaluatorOptimizerState)

    # Add nodes
    builder.add_node("initial_research", initial_research)
    builder.add_node("research_worker", research_worker)
    builder.add_node("synthesizer", synthesizer)
    builder.add_node("evaluator", evaluator)
    builder.add_node("retry_planner", retry_planner)
    builder.add_node("finalize", finalize)

    # Add edges
    builder.add_edge(START, "initial_research")
    builder.add_conditional_edges(
        "initial_research",
        spawn_research_workers,
        ["research_worker"]
    )
    builder.add_edge("research_worker", "synthesizer")
    builder.add_edge("synthesizer", "evaluator")

    # Conditional edge for retry decision
    builder.add_conditional_edges(
        "evaluator",
        should_retry,
        {
            "retry": "retry_planner",
            "finalize": "finalize",
        }
    )

    # Retry loop
    builder.add_conditional_edges(
        "retry_planner",
        spawn_research_workers,
        ["research_worker"]
    )

    builder.add_edge("finalize", END)

    # Compile
    graph = builder.compile(checkpointer=checkpointer)

    logger.info("Created evaluator_optimizer workflow graph")

    return graph


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def run_with_quality_gate(
    query: str,
    max_retries: int = 2,
) -> dict[str, Any]:
    """
    Run research with quality gates.

    Convenience function for standalone usage.

    Args:
        query: The research query
        max_retries: Maximum retry attempts (default 2)

    Returns:
        Approved answer with confidence and evaluation info
    """
    graph = create_evaluator_optimizer()

    initial_state = EvaluatorOptimizerState(
        query=query,
        messages=[],
        research_tasks=[],
        research_results=[],
        final_answer=None,
        confidence=0.0,
        evaluation=None,
        retry_count=0,
        max_retries=max_retries,
        approved_answer=None,
    )

    result = await graph.ainvoke(initial_state)

    return {
        "answer": result.get("approved_answer"),
        "confidence": result.get("confidence", 0.0),
        "evaluation": result.get("evaluation"),
        "retries_used": result.get("retry_count", 0),
    }
