"""
Orchestrator v2 - Enhanced LangGraph orchestrator with advanced patterns.

Integrates:
- Parallel research with Send API
- Evaluator-optimizer cycles for quality gates
- Mentor Hub API tools for live data
- Cognee memory for cross-session persistence
- Query routing based on classification
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Optional

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.base import BaseCheckpointSaver

from app.config import get_settings
from app.graphs.routing import analyze_query, QueryType
from app.tools.graph_tools import query_graph, search_text, find_entity, get_graph_schema
from app.tools.cognee_tools import (
    search_chunks,
    search_summaries,
    search_graph,
    search_rag,
)
from app.tools.mentor_hub_tools import get_mentor_hub_tools

logger = logging.getLogger(__name__)


# =============================================================================
# V2 ORCHESTRATOR PROMPT
# =============================================================================

ORCHESTRATOR_V2_PROMPT = """You are the Master Orchestrator for the MentorHub platform, powered by Cognee knowledge graph.

## YOUR CAPABILITIES

You have access to multiple tool categories:

### 1. Cognee RAG Tools (Knowledge from documents)
- `search_text(query)` - Semantic text search
- `search_chunks(query)` - Raw passage retrieval
- `search_summaries(query)` - Document overviews
- `search_graph(query)` - Graph-enhanced synthesis
- `search_rag(query)` - Full RAG completion

### 2. Graph Query Tools (Cypher via Cognee)
- `query_graph(cypher)` - Run Cypher queries
- `find_entity(name)` - Find entity + relationships
- `get_graph_schema()` - Get graph structure

### 3. Mentor Hub Live Data Tools
- `get_mentor_hub_sessions(...)` - Upcoming/past sessions
- `get_mentor_hub_team(team_id)` - Team details
- `search_mentor_hub_mentors(...)` - Find mentors
- `get_mentor_hub_tasks(...)` - Action items

### 4. Memory Tools (Cross-session via Cognee)
- `cognee_add` - Remember information
- `cognee_search` - Recall memories

## EXECUTION STRATEGY

### For Simple Queries (Who is X? What is Y?)
Call 3-5 tools in parallel:
- find_entity + search_text + search_chunks

### For Complex Queries (Explain X, List all Y)
Call 5-8 tools in parallel with query variations:
- Multiple search_text calls with different phrasings
- find_entity for key entities
- search_summaries for overviews

### For Analytical Queries (Compare, Recommend, Why)
1. First gather data with parallel searches
2. Then reason about the data
3. Provide well-structured analysis

### For Live Data Queries (My sessions, Upcoming, This week)
Prioritize Mentor Hub tools for real-time data:
- get_mentor_hub_sessions for scheduling
- get_mentor_hub_tasks for action items
- Supplement with Cognee for context

## PARALLEL EXECUTION - CRITICAL

You MUST call multiple tools simultaneously when possible.
NEVER call one tool, wait, then call another.

Example - For "Tell me about DefenX team":
Call ALL of these together:
1. find_entity("defenx")
2. search_text("defenx team overview")
3. search_text("defenx project description")
4. search_chunks("defenx members")
5. query_graph("MATCH (t:Entity)-[:has_member]->(m) WHERE t.name = 'defenx' RETURN m.name")

## RESPONSE FORMAT

Structure your responses clearly:
- Use headers (##) for sections
- Use bullet points for lists
- Use bold for key information
- Cite sources when possible
- Be comprehensive but focused

{context_section}
"""


def _build_context_section(user_context: Optional[dict[str, Any]] = None) -> str:
    """Build the dynamic context section for the prompt."""
    now = datetime.now()
    parts = [
        "",
        "## CURRENT CONTEXT",
        "",
        f"**Date:** {now.strftime('%A, %B %d, %Y')}",
        f"**Time:** {now.strftime('%I:%M %p')}",
    ]

    if user_context:
        parts.append("")
        parts.append("**Current User:**")
        if user_context.get("name"):
            parts.append(f"- Name: {user_context['name']}")
        if user_context.get("email"):
            parts.append(f"- Email: {user_context['email']}")
        if user_context.get("role"):
            parts.append(f"- Role: {user_context['role']}")
        if user_context.get("teams"):
            teams = user_context["teams"]
            if isinstance(teams, list):
                teams = ", ".join(teams)
            parts.append(f"- Teams: {teams}")
        if user_context.get("cohort"):
            parts.append(f"- Cohort: {user_context['cohort']}")
        parts.append("")
        parts.append("Use this to personalize responses.")

    return "\n".join(parts)


# =============================================================================
# V2 ORCHESTRATOR FACTORY
# =============================================================================

def create_orchestrator_v2(
    checkpointer: Optional[BaseCheckpointSaver] = None,
    user_context: Optional[dict[str, Any]] = None,
    auth0_id: Optional[str] = None,
) -> CompiledStateGraph:
    """
    Create the v2 Orchestrator with enhanced capabilities.

    Uses LangChain v1's create_agent (replaces create_react_agent).

    Features:
    - All Cognee RAG tools
    - All Graph query tools
    - Mentor Hub live data tools
    - Cognee memory tools (if auth0_id provided)

    Args:
        checkpointer: Session persistence
        user_context: User details for personalization
        auth0_id: User ID for memory isolation

    Returns:
        Compiled orchestrator graph
    """
    settings = get_settings()

    # Create the LLM with API key from settings
    llm = ChatOpenAI(
        model=settings.default_orchestrator_model,
        api_key=settings.openai_api_key,
        streaming=True,
    )

    # Build prompt with context
    context_section = _build_context_section(user_context)
    full_prompt = ORCHESTRATOR_V2_PROMPT.format(context_section=context_section)

    # Collect all tools
    tools = []

    # 1. Cognee RAG tools
    tools.extend([
        search_text,
        search_chunks,
        search_summaries,
        search_graph,
        search_rag,
    ])

    # 2. Graph query tools (via Cognee)
    tools.extend([
        query_graph,
        find_entity,
        get_graph_schema,
    ])

    # 3. Mentor Hub live data tools
    tools.extend(get_mentor_hub_tools())

    # 4. Cognee memory tools (if authenticated)
    if auth0_id and settings.cognee_memory_enabled:
        try:
            from app.persistence.memory_store import get_memory_tools_for_orchestrator
            memory_tools = get_memory_tools_for_orchestrator(auth0_id)
            tools.extend(memory_tools)
            logger.info(f"Added Cognee memory tools for user: {auth0_id[:20]}...")
        except ImportError as e:
            logger.warning(f"Cognee memory tools not available: {e}")

    logger.info(
        f"Creating orchestrator v2 with {len(tools)} tools, "
        f"model: {settings.default_orchestrator_model}"
    )

    # Create agent using LangChain v1's create_agent
    # This replaces the deprecated langgraph.prebuilt.create_react_agent
    graph = create_agent(
        model=llm,
        tools=tools,
        system_prompt=full_prompt,
        checkpointer=checkpointer,
    )

    return graph


# =============================================================================
# SINGLETON MANAGEMENT
# =============================================================================

_orchestrator_v2_cache: dict[str, CompiledStateGraph] = {}


async def get_orchestrator_v2(
    checkpointer: Optional[BaseCheckpointSaver] = None,
    user_context: Optional[dict[str, Any]] = None,
    auth0_id: Optional[str] = None,
    force_new: bool = False,
) -> CompiledStateGraph:
    """
    Get or create an orchestrator v2 instance.

    For authenticated users, creates a unique orchestrator with their memory.
    For anonymous users, returns a shared instance.

    Args:
        checkpointer: Session persistence
        user_context: User details
        auth0_id: User ID for memory
        force_new: Force new instance

    Returns:
        Compiled orchestrator v2
    """
    cache_key = auth0_id or "anonymous"

    if not force_new and cache_key in _orchestrator_v2_cache:
        logger.debug(f"Returning cached orchestrator v2 for: {cache_key[:20] if auth0_id else 'anonymous'}...")
        return _orchestrator_v2_cache[cache_key]

    orchestrator = create_orchestrator_v2(
        checkpointer=checkpointer,
        user_context=user_context,
        auth0_id=auth0_id,
    )

    _orchestrator_v2_cache[cache_key] = orchestrator
    logger.info(f"Created orchestrator v2 for: {cache_key[:20] if auth0_id else 'anonymous'}...")

    return orchestrator


def clear_orchestrator_v2_cache():
    """Clear the orchestrator v2 cache."""
    global _orchestrator_v2_cache
    _orchestrator_v2_cache = {}
    logger.info("Cleared orchestrator v2 cache")


# =============================================================================
# WORKFLOW SELECTION (for future supervisor pattern)
# =============================================================================

async def run_with_optimal_workflow(
    query: str,
    checkpointer: Optional[BaseCheckpointSaver] = None,
    user_context: Optional[dict[str, Any]] = None,
    auth0_id: Optional[str] = None,
    config: Optional[dict] = None,
) -> dict[str, Any]:
    """
    Run a query with automatic workflow selection.

    Analyzes the query and routes to the optimal workflow:
    - Simple/Complex → Standard orchestrator (parallel tools)
    - Analytical → Evaluator-optimizer for quality
    - Action → Prioritize Mentor Hub tools

    Args:
        query: User query
        checkpointer: Session persistence
        user_context: User context
        auth0_id: User ID
        config: LangGraph config

    Returns:
        Result with answer and metadata
    """
    # Analyze query
    analysis = analyze_query(query)
    logger.info(f"Query analysis: type={analysis['query_type']}, workflow={analysis['workflow']}")

    # For now, use the standard orchestrator v2 for all queries
    # The orchestrator will use the appropriate tools based on the query
    orchestrator = await get_orchestrator_v2(
        checkpointer=checkpointer,
        user_context=user_context,
        auth0_id=auth0_id,
    )

    # Invoke with config
    result = await orchestrator.ainvoke(
        {"messages": [{"role": "user", "content": query}]},
        config=config or {},
    )

    return {
        "messages": result.get("messages", []),
        "query_analysis": analysis,
    }
