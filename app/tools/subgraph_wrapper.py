"""
Subgraph Wrapper - Wraps LangGraph subgraphs as callable tools.

This allows the main orchestrator to delegate complex tasks to
specialized sub-agents (researchers, deep reasoning, mentor matcher)
by calling them as tools.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from langchain_core.tools import StructuredTool
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SubgraphInput(BaseModel):
    """Input schema for subgraph tool invocation."""

    query: str = Field(..., description="The research query or task to delegate")


def create_subgraph_tool(
    name: str,
    subgraph: CompiledStateGraph,
    description: str,
) -> StructuredTool:
    """
    Wrap a compiled LangGraph subgraph as a callable tool.

    This allows the main orchestrator to invoke specialized sub-agents
    as tools, enabling hierarchical multi-agent orchestration.

    Args:
        name: Tool name (e.g., "entity_researcher")
        subgraph: Compiled LangGraph subgraph
        description: Tool description for the orchestrator

    Returns:
        StructuredTool that invokes the subgraph
    """

    async def invoke_subgraph(query: str) -> str:
        """Invoke the subgraph and return the final response."""
        try:
            logger.info(f"Delegating to {name}: {query[:100]}...")

            # Invoke the subgraph
            result = await subgraph.ainvoke({
                "messages": [{"role": "user", "content": query}]
            })

            # Extract the final answer from the result
            response = _extract_final_answer(result)
            logger.info(f"{name} completed, response length: {len(response)}")

            return response

        except Exception as e:
            logger.error(f"Subgraph {name} error: {e}", exc_info=True)
            return f"Error delegating to {name}: {str(e)}"

    return StructuredTool(
        name=name,
        description=description,
        func=lambda query: None,  # Sync placeholder (not used)
        coroutine=invoke_subgraph,
        args_schema=SubgraphInput,
    )


def _extract_final_answer(result: dict[str, Any]) -> str:
    """
    Extract the final answer text from a subgraph result.

    Handles different result formats from create_agent.
    """
    messages = result.get("messages", [])

    if not messages:
        return "No response generated"

    # Get the last AI message
    for message in reversed(messages):
        # Handle LangChain message objects
        if hasattr(message, "content"):
            content = message.content
            if content and isinstance(content, str):
                return content
        # Handle dict format
        elif isinstance(message, dict):
            content = message.get("content")
            if content and isinstance(content, str):
                return content

    return "No response generated"
