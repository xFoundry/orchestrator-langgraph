"""
Cognee-based Memory Store for LangGraph.

Uses cognee-integration-langgraph for cross-session persistent memory
with semantic search via Cognee's knowledge graph.

Features:
- Cross-session memory persistence
- Semantic search via Cognee
- Per-user session isolation
- Automatic knowledge graph construction
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)


def get_cognee_memory_tools(
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> Tuple[Any, Any]:
    """
    Get Cognee memory tools for LangGraph agents.

    Uses cognee-integration-langgraph to provide add/search tools
    that persist across sessions via Cognee's knowledge graph.

    Args:
        session_id: Optional session ID for memory isolation.
                   If not provided, uses user_id or generates UUID.
        user_id: User ID to use as session ID if session_id not provided.

    Returns:
        Tuple of (add_tool, search_tool) for use with create_agent
    """
    try:
        from cognee_integration_langgraph import get_sessionized_cognee_tools

        # Use user_id as session_id for per-user memory isolation
        effective_session = session_id or user_id

        if effective_session:
            add_tool, search_tool = get_sessionized_cognee_tools(
                session_id=effective_session
            )
            logger.info(f"Created Cognee memory tools for session: {effective_session[:20]}...")
        else:
            # Auto-generate session ID
            add_tool, search_tool = get_sessionized_cognee_tools()
            logger.info("Created Cognee memory tools with auto-generated session")

        return add_tool, search_tool

    except ImportError as e:
        logger.error(f"cognee-integration-langgraph not installed: {e}")
        logger.info("Install with: pip install cognee-integration-langgraph")
        raise ImportError(
            "cognee-integration-langgraph is required for memory tools. "
            "Install with: pip install cognee-integration-langgraph"
        ) from e


def get_cognee_memory_tools_for_user(auth0_id: str) -> Tuple[Any, Any]:
    """
    Get user-scoped Cognee memory tools.

    Creates memory tools isolated to a specific user via their Auth0 ID.
    Memory persists across sessions for the same user.

    Args:
        auth0_id: User's Auth0 ID for memory isolation

    Returns:
        Tuple of (add_tool, search_tool)
    """
    # Create a deterministic session ID from auth0_id
    # This ensures the same user always accesses the same memory
    session_id = f"user_{auth0_id}"
    return get_cognee_memory_tools(session_id=session_id)


# =============================================================================
# TOOL WRAPPERS WITH BETTER NAMES
# =============================================================================

def create_remember_tool(session_id: str):
    """
    Create a 'remember' tool that stores information in Cognee.

    This wraps the Cognee add tool with a more intuitive name.

    Args:
        session_id: Session ID for memory isolation

    Returns:
        LangChain tool for remembering information
    """
    from langchain_core.tools import tool

    add_tool, _ = get_cognee_memory_tools(session_id=session_id)

    @tool
    async def remember(information: str) -> str:
        """
        Store information in long-term memory.

        Use this to remember facts, preferences, or context about the user
        that should persist across conversations.

        Args:
            information: The information to remember

        Returns:
            Confirmation message
        """
        await add_tool.ainvoke({"data": information})
        return f"Remembered: {information[:100]}..."

    return remember


def create_recall_tool(session_id: str):
    """
    Create a 'recall' tool that searches Cognee memory.

    This wraps the Cognee search tool with a more intuitive name.

    Args:
        session_id: Session ID for memory isolation

    Returns:
        LangChain tool for recalling information
    """
    from langchain_core.tools import tool

    _, search_tool = get_cognee_memory_tools(session_id=session_id)

    @tool
    async def recall(query: str) -> str:
        """
        Search long-term memory for relevant information.

        Use this to recall facts, preferences, or context about the user
        from previous conversations.

        Args:
            query: What to search for in memory

        Returns:
            Relevant memories found
        """
        result = await search_tool.ainvoke({"query": query})
        return result

    return recall


def get_memory_tools_for_orchestrator(auth0_id: Optional[str] = None) -> list:
    """
    Get memory tools configured for the orchestrator.

    Returns the raw Cognee tools (add/search) for direct use
    in create_agent.

    Args:
        auth0_id: Optional user ID for memory isolation

    Returns:
        List of [add_tool, search_tool]
    """
    if auth0_id:
        add_tool, search_tool = get_cognee_memory_tools_for_user(auth0_id)
    else:
        add_tool, search_tool = get_cognee_memory_tools(session_id="anonymous")

    return [add_tool, search_tool]
