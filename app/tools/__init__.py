"""Tools for the LangGraph orchestrator."""

from app.tools.graph_tools import (
    query_graph,
    search_text,
    find_entity,
    get_graph_schema,
)
from app.tools.cognee_tools import (
    search_chunks,
    search_graph,
    search_rag,
    search_summaries,
    search_natural_language,
    cognee_add,
)
from app.tools.user_memory import get_user_memory_tools
from app.tools.subgraph_wrapper import create_subgraph_tool
from app.tools.mentor_hub_tools import (
    get_mentor_hub_sessions,
    get_mentor_hub_team,
    search_mentor_hub_mentors,
    get_mentor_hub_tasks,
    get_mentor_hub_user_context,
    get_mentor_hub_tools,
)
from app.tools.firecrawl_tools import (
    firecrawl_scrape,
    firecrawl_search,
    firecrawl_map,
    firecrawl_crawl,
    firecrawl_extract,
)
from app.tools.clarification_tools import get_clarification_tools
from app.tools.tool_registry import (
    ToolRegistry,
    ToolGroup,
    create_default_registry,
)

__all__ = [
    # Graph tools (via Cognee Cypher)
    "query_graph",
    "search_text",
    "find_entity",
    "get_graph_schema",
    # Cognee RAG tools
    "search_chunks",
    "search_graph",
    "search_rag",
    "search_summaries",
    "search_natural_language",
    "cognee_add",
    # Mentor Hub API tools
    "get_mentor_hub_sessions",
    "get_mentor_hub_team",
    "search_mentor_hub_mentors",
    "get_mentor_hub_tasks",
    "get_mentor_hub_user_context",
    "get_mentor_hub_tools",
    # Firecrawl tools
    "firecrawl_scrape",
    "firecrawl_search",
    "firecrawl_map",
    "firecrawl_crawl",
    "firecrawl_extract",
    # Clarification tools
    "get_clarification_tools",
    # User memory
    "get_user_memory_tools",
    # Subgraph wrapper
    "create_subgraph_tool",
    # Tool registry (for Deep Agent)
    "ToolRegistry",
    "ToolGroup",
    "create_default_registry",
]
