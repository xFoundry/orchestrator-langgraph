"""
Parallel Researcher Workflow using LangGraph Send API.

Spawns multiple research workers in parallel for faster data gathering.
Each worker executes independently and writes results to shared state.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, Optional

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from app.config import get_settings
from app.graphs.state import (
    ParallelResearchState,
    WorkerState,
    ResearchTask,
    ResearchResult,
)
# All tools go through Cognee API
from app.tools.graph_tools import query_graph, search_text, find_entity  # Cognee Cypher interface
from app.tools.cognee_tools import search_chunks, search_summaries, search_graph, search_rag  # Cognee search

logger = logging.getLogger(__name__)


# =============================================================================
# QUERY ANALYSIS
# =============================================================================

def analyze_query(query: str) -> dict[str, Any]:
    """
    Analyze a query to determine research strategy.

    Returns task types and priorities based on query characteristics.
    """
    query_lower = query.lower()

    analysis = {
        "needs_entity_search": False,
        "needs_text_search": True,  # Almost always needed
        "needs_summary": False,
        "needs_graph_query": False,
        "needs_mentor_hub": False,
        "entities": [],
        "query_variations": [],
    }

    # Check for entity queries
    entity_indicators = ["who is", "what is", "tell me about", "find", "show me"]
    if any(ind in query_lower for ind in entity_indicators):
        analysis["needs_entity_search"] = True

    # Check for summary/overview queries
    summary_indicators = ["overview", "summary", "summarize", "high-level", "big picture"]
    if any(ind in query_lower for ind in summary_indicators):
        analysis["needs_summary"] = True

    # Check for relationship/graph queries
    graph_indicators = ["related to", "connected to", "relationship", "members", "mentors of"]
    if any(ind in query_lower for ind in graph_indicators):
        analysis["needs_graph_query"] = True

    # Check for live data queries
    live_data_indicators = ["upcoming", "scheduled", "my sessions", "my tasks", "this week"]
    if any(ind in query_lower for ind in live_data_indicators):
        analysis["needs_mentor_hub"] = True

    # Generate query variations for broader search
    analysis["query_variations"] = generate_query_variations(query)

    return analysis


def generate_query_variations(query: str) -> list[str]:
    """Generate search query variations for broader coverage."""
    variations = [query]  # Original query

    # Add variations based on common synonyms
    synonym_map = {
        "challenges": ["problems", "blockers", "difficulties", "obstacles", "issues"],
        "feedback": ["recommendations", "suggestions", "comments", "input"],
        "progress": ["status", "updates", "advancement", "development"],
        "team": ["group", "squad", "project"],
    }

    query_lower = query.lower()
    for word, synonyms in synonym_map.items():
        if word in query_lower:
            for syn in synonyms[:2]:  # Limit to 2 synonyms per word
                variations.append(query_lower.replace(word, syn))

    return variations[:5]  # Limit total variations


# =============================================================================
# PLANNING NODE
# =============================================================================

def planner(state: ParallelResearchState) -> dict[str, Any]:
    """
    Analyze query and create parallel research tasks.

    This node decomposes the query into multiple tasks that can
    be executed in parallel by worker nodes.
    """
    query = state["query"]
    analysis = analyze_query(query)

    tasks: list[ResearchTask] = []
    task_id_counter = 0

    def make_task_id() -> str:
        nonlocal task_id_counter
        task_id_counter += 1
        return f"task_{task_id_counter}_{uuid.uuid4().hex[:6]}"

    # Entity search tasks
    if analysis["needs_entity_search"]:
        tasks.append(ResearchTask(
            id=make_task_id(),
            type="entity",
            query=query,
            focus="entities and relationships",
            priority="high",
            config=None,
        ))

    # Text search tasks (with variations)
    for i, variation in enumerate(analysis["query_variations"][:3]):
        tasks.append(ResearchTask(
            id=make_task_id(),
            type="text",
            query=variation,
            focus=f"text passages (variation {i+1})",
            priority="high" if i == 0 else "medium",
            config={"top_k": 10},
        ))

    # Summary search
    if analysis["needs_summary"]:
        tasks.append(ResearchTask(
            id=make_task_id(),
            type="summary",
            query=query,
            focus="document summaries",
            priority="medium",
            config=None,
        ))

    # Graph query
    if analysis["needs_graph_query"]:
        tasks.append(ResearchTask(
            id=make_task_id(),
            type="graph",
            query=query,
            focus="graph relationships",
            priority="high",
            config=None,
        ))

    # Always include RAG for comprehensive coverage
    tasks.append(ResearchTask(
        id=make_task_id(),
        type="summary",
        query=query,
        focus="RAG synthesis",
        priority="medium",
        config={"use_rag": True},
    ))

    logger.info(f"Planner created {len(tasks)} research tasks for query: {query[:50]}...")

    return {"research_tasks": tasks}


# =============================================================================
# WORKER NODE
# =============================================================================

async def researcher_worker(state: WorkerState) -> dict[str, Any]:
    """
    Execute a single research task.

    This node is spawned multiple times in parallel via Send API.
    Each instance processes one task and writes to shared state.
    """
    task = state["task"]
    start_time = time.time()

    logger.info(f"Worker executing task {task['id']}: {task['type']} - {task['focus']}")

    try:
        result_data = None
        sources = []
        confidence = 0.5

        if task["type"] == "entity":
            # Entity search via Cognee Cypher
            entity_result = await find_entity.ainvoke({"name": task["query"]})
            result_data = entity_result
            sources = ["cognee:cypher:entity"]
            confidence = 0.8 if entity_result.get("found") else 0.3

        elif task["type"] == "text":
            # Text search via Cognee
            top_k = task.get("config", {}).get("top_k", 10) if task.get("config") else 10
            text_result = await search_text.ainvoke({"query": task["query"], "top_k": top_k})
            result_data = text_result
            sources = ["cognee:chunks"]
            confidence = 0.7 if text_result else 0.3

        elif task["type"] == "summary":
            # Summary or RAG search
            if task.get("config", {}).get("use_rag"):
                rag_result = await search_rag.ainvoke({"query": task["query"]})
                result_data = rag_result
                sources = ["cognee:rag"]
            else:
                summary_result = await search_summaries.ainvoke({"query": task["query"]})
                result_data = summary_result
                sources = ["cognee:summary"]
            confidence = 0.6

        elif task["type"] == "graph":
            # Graph search
            graph_result = await search_graph.ainvoke({"query": task["query"]})
            result_data = graph_result
            sources = ["cognee:graph"]
            confidence = 0.7 if graph_result else 0.3

        execution_time = int((time.time() - start_time) * 1000)

        result = ResearchResult(
            task_id=task["id"],
            task_type=task["type"],
            success=True,
            data=result_data,
            confidence=confidence,
            sources=sources,
            error=None,
            execution_time_ms=execution_time,
        )

        logger.info(f"Worker completed task {task['id']} in {execution_time}ms")

        return {"research_results": [result]}

    except Exception as e:
        execution_time = int((time.time() - start_time) * 1000)
        logger.error(f"Worker failed task {task['id']}: {e}")

        result = ResearchResult(
            task_id=task["id"],
            task_type=task["type"],
            success=False,
            data=None,
            confidence=0.0,
            sources=[],
            error=str(e),
            execution_time_ms=execution_time,
        )

        return {"research_results": [result]}


# =============================================================================
# SPAWN FUNCTION (CONDITIONAL EDGE)
# =============================================================================

def spawn_researchers(state: ParallelResearchState) -> list[Send]:
    """
    Spawn parallel research workers using Send API.

    This conditional edge function creates a Send object for each
    research task, enabling parallel execution.
    """
    tasks = state["research_tasks"]

    logger.info(f"Spawning {len(tasks)} parallel research workers")

    return [
        Send("researcher_worker", {"task": task})
        for task in tasks
    ]


# =============================================================================
# SYNTHESIZER NODE
# =============================================================================

def synthesizer(state: ParallelResearchState) -> dict[str, Any]:
    """
    Synthesize all research results into unified response.

    Aggregates data from parallel workers and calculates
    overall confidence score.
    """
    results = state["research_results"]

    # Separate successful and failed results
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    logger.info(f"Synthesizer received {len(successful)} successful, {len(failed)} failed results")

    # Calculate overall confidence
    if successful:
        avg_confidence = sum(r["confidence"] for r in successful) / len(successful)
        # Boost confidence if we have diverse sources
        source_types = set(r["task_type"] for r in successful)
        diversity_bonus = min(len(source_types) * 0.05, 0.15)
        overall_confidence = min(avg_confidence + diversity_bonus, 1.0)
    else:
        overall_confidence = 0.0

    # Collect citations
    citations = []
    for r in successful:
        for source in r["sources"]:
            citations.append({
                "source": source,
                "task_id": r["task_id"],
                "confidence": r["confidence"],
            })

    # Combine all data for synthesis
    combined_data = {
        "entity_data": [r["data"] for r in successful if r["task_type"] == "entity" and r["data"]],
        "text_data": [r["data"] for r in successful if r["task_type"] == "text" and r["data"]],
        "summary_data": [r["data"] for r in successful if r["task_type"] == "summary" and r["data"]],
        "graph_data": [r["data"] for r in successful if r["task_type"] == "graph" and r["data"]],
    }

    # Note: Actual synthesis would be done by LLM in the main orchestrator
    # Here we just prepare the data structure

    return {
        "synthesized_answer": None,  # Will be filled by orchestrator LLM
        "confidence": overall_confidence,
        "citations": citations,
        # Store combined data for access by orchestrator
        "research_results": results,  # Keep results for reference
    }


# =============================================================================
# GRAPH BUILDER
# =============================================================================

def create_parallel_researcher(
    checkpointer: Optional[Any] = None,
) -> CompiledStateGraph:
    """
    Create the parallel researcher workflow graph.

    Flow:
    1. Planner analyzes query and creates tasks
    2. Workers execute tasks in parallel (via Send API)
    3. Synthesizer aggregates results

    Args:
        checkpointer: Optional checkpointer for state persistence

    Returns:
        Compiled parallel researcher graph
    """
    builder = StateGraph(ParallelResearchState)

    # Add nodes
    builder.add_node("planner", planner)
    builder.add_node("researcher_worker", researcher_worker)
    builder.add_node("synthesizer", synthesizer)

    # Add edges
    builder.add_edge(START, "planner")

    # Conditional edge spawns parallel workers
    builder.add_conditional_edges(
        "planner",
        spawn_researchers,
        ["researcher_worker"]
    )

    # All workers converge to synthesizer
    builder.add_edge("researcher_worker", "synthesizer")
    builder.add_edge("synthesizer", END)

    # Compile
    graph = builder.compile(checkpointer=checkpointer)

    logger.info("Created parallel_researcher workflow graph")

    return graph


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def run_parallel_research(
    query: str,
    user_context: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Run parallel research for a query.

    Convenience function for standalone usage.

    Args:
        query: The research query
        user_context: Optional user context

    Returns:
        Research results with confidence and citations
    """
    graph = create_parallel_researcher()

    initial_state = ParallelResearchState(
        query=query,
        messages=[],
        user_context=user_context,
        research_tasks=[],
        research_results=[],
        synthesized_answer=None,
        confidence=0.0,
        citations=[],
    )

    result = await graph.ainvoke(initial_state)

    return {
        "results": result["research_results"],
        "confidence": result["confidence"],
        "citations": result["citations"],
    }
