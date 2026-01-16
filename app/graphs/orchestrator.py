"""
Root Orchestrator Agent - LangGraph implementation.

Coordinates specialized sub-agents for comprehensive knowledge retrieval:
- Entity Researcher: Graph-focused entity discovery
- Text Researcher: Semantic document search
- Summary Researcher: High-level document summaries
- Deep Reasoning: Complex analytical queries
- Mentor Matcher: Mentor-team recommendations
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Optional

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.redis.aio import AsyncRedisSaver

from app.config import get_settings
from app.tools.graph_tools import query_graph, search_text, find_entity, get_graph_schema
from app.tools.cognee_tools import (
    search_chunks,
    search_summaries,
    search_graph,
    search_rag,
)
from app.tools.subgraph_wrapper import create_subgraph_tool
from app.graphs.subgraphs import (
    create_entity_researcher,
    create_text_researcher,
    create_summary_researcher,
    create_deep_reasoning,
    create_mentor_matcher,
    ENTITY_RESEARCHER_DESCRIPTION,
    TEXT_RESEARCHER_DESCRIPTION,
    SUMMARY_RESEARCHER_DESCRIPTION,
    DEEP_REASONING_DESCRIPTION,
    MENTOR_MATCHER_DESCRIPTION,
)

logger = logging.getLogger(__name__)


# =============================================================================
# ORCHESTRATOR PROMPT
# =============================================================================

ORCHESTRATOR_INSTRUCTION = """You are the Master Orchestrator for the CognoXent mentorship platform.

Your role is to coordinate multiple specialized research agents and tools to provide comprehensive, well-researched answers. You have access to both direct tools AND specialized sub-agents for delegation.

## PlanReAct FRAMEWORK

You MUST follow the PlanReAct structured reasoning format exactly. Use ONLY these markers:

/*PLANNING*/
Analyze the question and create your research plan:
- Break down the question into sub-questions
- Identify entities, relationships, aspects to explore
- Plan which tools/agents to call IN PARALLEL

/*ACTION*/
Execute your planned tool calls. ALWAYS call multiple tools simultaneously.
NEVER call tools one at a time - batch all related calls together.

/*REASONING*/
INTERNAL ANALYSIS ONLY - Users do NOT see this section.
Keep it brief - just a few bullet points:
- What did the searches reveal?
- Are there gaps that need follow-up?
- Do results need deeper analysis?

Example REASONING (brief, internal):
```
/*REASONING*/
- Found DefenX entity with 4 mentors
- Search results show recommendations about data validation and pilot testing
- Have enough info to provide comprehensive answer
```

/*FINAL_ANSWER*/
THE COMPLETE, FORMATTED USER RESPONSE - This is the ONLY section users see!
Put your ENTIRE answer here, including:
- Headers and formatting
- All bullet points and details
- Tables if relevant
- The full polished response

⚠️ CRITICAL: NEVER put the user's answer in /*REASONING*/!
The /*FINAL_ANSWER*/ section must contain the complete response.

---

## CRITICAL: PARALLEL EXECUTION

You MUST call multiple tools in parallel. Examples:

**Simple Query** (e.g., "Tell me about DefenX"):
Call at least 5 tools simultaneously:
- find_entity("defenx")
- search_text("defenx team information")
- search_text("defenx project description")
- search_chunks("defenx overview")
- query_graph("MATCH (e:Entity)-[:is_a]->(:EntityType {name:'team'}) WHERE toLower(e.name) CONTAINS 'defenx' RETURN e")

**Complex Query** (e.g., "What are DefenX's blockers?"):
Call at least 6-8 tools simultaneously:
- find_entity("defenx")
- search_text("defenx challenges blockers")
- search_text("defenx problems difficulties")
- search_text("defenx feedback concerns")
- search_text("defenx needs improvement")
- query_graph("MATCH (t:Entity {name:'defenx'})-[:mentored_by]->(m) RETURN m.name")

---

## YOUR TOOLS

### Direct Tools
- `get_graph_schema()` - Graph structure
- `query_graph(cypher)` - Cypher queries
- `find_entity(name)` - Entity + relationships
- `search_text(query, top_k=20)` - Semantic search
- `search_chunks(query, top_k=20)` - Passage retrieval
- `search_summaries(query)` - Document overviews
- `search_graph(query)` - Graph synthesis
- `search_rag(query)` - RAG completion

### Sub-Agents (delegate complex tasks)
- **entity_researcher** - Graph exploration
- **text_researcher** - Multi-query text search
- **summary_researcher** - High-level overviews
- **deep_reasoning** - Complex analysis
- **mentor_matcher** - Mentor recommendations

### User Memory
- `remember(data)` - Store facts about user
- `recall(query)` - Recall user preferences

---

## MULTI-ROUND RESEARCH

For complex questions:

/*PLANNING*/
Break down question, identify what to search.

/*ACTION*/
[Execute 5-8 parallel tool calls for initial discovery]

/*REASONING*/
Assess results:
- Found: [key findings]
- Missing: [gaps identified]
- Need: [follow-up queries]

/*ACTION*/
[Execute 3-5 more parallel calls to fill gaps]

/*REASONING*/
Evaluate if ready for final answer or need more research.

/*FINAL_ANSWER*/
Synthesized, comprehensive response.

---

## QUERY VARIATION

Use multiple phrasings for each search:
- challenges → problems, blockers, difficulties, obstacles
- feedback → recommendations, suggestions, concerns
- "defenx" → "defenx team", "defenx project", "defenx members"

---

## RESPONSE FORMATTING GUIDELINES

Your final answers should be **comprehensive, well-structured, and visually organized**. Use rich markdown formatting:

### Structure
- Use **headers** (##, ###) to organize major sections
- Use **bullet points** and **numbered lists** for clarity
- Add **blank lines** between paragraphs and sections for readability
- Use **bold** for key terms, names, dates, and important information
- Use *italics* for emphasis or quotes

### Tables
When comparing items or presenting structured data, use tables:
```
| Column 1 | Column 2 | Column 3 |
|----------|----------|----------|
| Data 1   | Data 2   | Data 3   |
```

### Length & Detail
- Provide **thorough, detailed responses** - users want comprehensive information
- Include relevant context and background information
- Explain the "why" behind facts, not just the "what"
- If there are multiple aspects, cover each one with enough detail
- Aim for 200-500 words for complex topics

### What to Include
- **Overview**: Start with a brief summary of key findings
- **Details**: Expand on each point with specifics from the data
- **Context**: Provide relevant background information
- **Relationships**: Explain how entities/concepts connect
- **Gaps**: Note what information wasn't found (if relevant)
"""


# =============================================================================
# ORCHESTRATOR FACTORY
# =============================================================================

def _build_context_section(user_context: Optional[dict[str, Any]] = None) -> str:
    """Build the dynamic context section for the prompt."""
    now = datetime.now()
    context_parts = [
        "",
        "---",
        "",
        "## CURRENT SESSION CONTEXT",
        "",
        f"**Today's date:** {now.strftime('%A, %B %d, %Y')}",
        f"**Current time:** {now.strftime('%I:%M %p')}",
    ]

    if user_context:
        context_parts.append("")
        context_parts.append("**Current user:**")
        if user_context.get("name"):
            context_parts.append(f"- Name: {user_context['name']}")
        if user_context.get("email"):
            context_parts.append(f"- Email: {user_context['email']}")
        if user_context.get("role"):
            context_parts.append(f"- Role: {user_context['role']}")
        if user_context.get("teams"):
            teams = user_context["teams"]
            if isinstance(teams, list):
                teams = ", ".join(teams)
            context_parts.append(f"- Teams: {teams}")
        if user_context.get("cohort"):
            context_parts.append(f"- Cohort: {user_context['cohort']}")
        context_parts.append("")
        context_parts.append("Use this to personalize responses and answer questions about 'my' data.")

    context_parts.append("")
    context_parts.append("Use the date to determine what's \"upcoming\", \"next\", \"past\", etc.")

    return "\n".join(context_parts)


def create_orchestrator(
    checkpointer: Optional[AsyncRedisSaver] = None,
    user_context: Optional[dict[str, Any]] = None,
    auth0_id: Optional[str] = None,
) -> CompiledStateGraph:
    """
    Create the main Orchestrator agent using LangChain v1's create_agent.

    The orchestrator uses a multi-agent architecture:
    - Direct tools for quick targeted searches
    - Specialized sub-agents (wrapped as tools) for delegation
    - create_agent pattern for structured multi-step reasoning

    Args:
        checkpointer: Redis checkpointer for session persistence
        user_context: Optional user details (name, email, role, teams, cohort)
        auth0_id: User's Auth0 ID for per-user memory isolation

    Returns:
        Compiled LangGraph orchestrator
    """
    settings = get_settings()

    # Create the LLM with API key from settings
    llm = ChatOpenAI(
        model=settings.default_orchestrator_model,
        api_key=settings.openai_api_key,
        streaming=True,
    )

    # Build the full prompt with dynamic context
    context_section = _build_context_section(user_context)
    full_prompt = ORCHESTRATOR_INSTRUCTION + context_section

    # Direct tools for quick targeted searches
    direct_tools = [
        get_graph_schema,
        query_graph,
        find_entity,
        search_text,
        search_chunks,
        search_summaries,
        search_graph,
        search_rag,
    ]

    # Add user memory tools if authenticated
    if auth0_id:
        try:
            from app.tools.user_memory import get_user_memory_tools
            memory_tools = get_user_memory_tools(auth0_id)
            direct_tools.extend(memory_tools)
            logger.info(f"Added user memory tools for auth0_id: {auth0_id[:20]}...")
        except ImportError:
            logger.warning("User memory tools not available")

    # Create subgraph tools (wrap sub-agents as callable tools)
    subgraph_tools = [
        create_subgraph_tool(
            name="entity_researcher",
            subgraph=create_entity_researcher(),
            description=ENTITY_RESEARCHER_DESCRIPTION,
        ),
        create_subgraph_tool(
            name="text_researcher",
            subgraph=create_text_researcher(),
            description=TEXT_RESEARCHER_DESCRIPTION,
        ),
        create_subgraph_tool(
            name="summary_researcher",
            subgraph=create_summary_researcher(),
            description=SUMMARY_RESEARCHER_DESCRIPTION,
        ),
        create_subgraph_tool(
            name="deep_reasoning",
            subgraph=create_deep_reasoning(),
            description=DEEP_REASONING_DESCRIPTION,
        ),
        create_subgraph_tool(
            name="mentor_matcher",
            subgraph=create_mentor_matcher(),
            description=MENTOR_MATCHER_DESCRIPTION,
        ),
    ]

    # Combine all tools
    all_tools = direct_tools + subgraph_tools

    logger.info(
        f"Creating orchestrator with model: {settings.default_orchestrator_model}, "
        f"direct_tools: {len(direct_tools)}, subgraph_tools: {len(subgraph_tools)}"
    )

    # Create agent using LangChain v1's create_agent
    # This replaces the deprecated langgraph.prebuilt.create_react_agent
    graph = create_agent(
        model=llm,
        tools=all_tools,
        system_prompt=full_prompt,
        checkpointer=checkpointer,
    )

    logger.info("Created orchestrator graph")
    return graph


# =============================================================================
# SINGLETON MANAGEMENT
# =============================================================================

_orchestrator_cache: dict[str, CompiledStateGraph] = {}


async def get_orchestrator(
    checkpointer: Optional[AsyncRedisSaver] = None,
    user_context: Optional[dict[str, Any]] = None,
    auth0_id: Optional[str] = None,
    force_new: bool = False,
) -> CompiledStateGraph:
    """
    Get or create an orchestrator instance.

    For authenticated users, creates a unique orchestrator with their memory tools.
    For anonymous users, returns a shared orchestrator instance.

    Args:
        checkpointer: Redis checkpointer for session persistence
        user_context: Optional user details
        auth0_id: User's Auth0 ID for per-user memory
        force_new: Force creation of new instance

    Returns:
        Compiled orchestrator graph
    """
    # Create cache key based on auth0_id (or "anonymous")
    cache_key = auth0_id or "anonymous"

    if not force_new and cache_key in _orchestrator_cache:
        logger.debug(f"Returning cached orchestrator for: {cache_key[:20] if auth0_id else 'anonymous'}...")
        return _orchestrator_cache[cache_key]

    # Create new orchestrator
    orchestrator = create_orchestrator(
        checkpointer=checkpointer,
        user_context=user_context,
        auth0_id=auth0_id,
    )

    # Cache it
    _orchestrator_cache[cache_key] = orchestrator
    logger.info(f"Created and cached new orchestrator for: {cache_key[:20] if auth0_id else 'anonymous'}...")

    return orchestrator


def clear_orchestrator_cache():
    """Clear the orchestrator cache (useful for testing)."""
    global _orchestrator_cache
    _orchestrator_cache = {}
    logger.info("Cleared orchestrator cache")
