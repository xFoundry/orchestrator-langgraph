"""
Tool Registry - Composable tool management for Deep Agent.

Provides a plug-and-play system for organizing and selecting tools.
Tools are grouped by domain (cognee, mentor_hub, firecrawl, memory)
and can be selectively enabled based on context.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)


@dataclass
class ToolGroup:
    """A group of related tools with metadata."""

    name: str
    tools: list[BaseTool | Callable]
    description: str
    enabled_by_default: bool = True


class ToolRegistry:
    """
    Central registry enabling plug-and-play tool composition.

    Usage:
        registry = ToolRegistry()
        registry.register_group(ToolGroup(
            name="cognee",
            tools=[find_entity, search_text, ...],
            description="Knowledge graph tools"
        ))

        # Get all enabled tools
        tools = registry.get_tools()

        # Get specific groups
        tools = registry.get_tools("cognee", "mentor_hub")
    """

    def __init__(self) -> None:
        self._groups: dict[str, ToolGroup] = {}

    def register_group(self, group: ToolGroup) -> None:
        """Register a tool group."""
        self._groups[group.name] = group
        logger.debug(f"Registered tool group: {group.name} ({len(group.tools)} tools)")

    def get_tools(
        self,
        *group_names: str,
        include_defaults: bool = True,
    ) -> list[BaseTool | Callable]:
        """
        Get tools by group names.

        Args:
            *group_names: Specific groups to include. If empty, includes all enabled groups.
            include_defaults: Whether to include groups enabled by default when no names specified.

        Returns:
            List of tools from the specified or enabled groups.
        """
        tools: list[BaseTool | Callable] = []

        for name, group in self._groups.items():
            # If specific groups requested, only include those
            if group_names and name not in group_names:
                continue
            # If no specific groups, respect enabled_by_default
            if not group_names and not group.enabled_by_default and include_defaults:
                continue
            tools.extend(group.tools)

        return tools

    def get_group(self, name: str) -> Optional[ToolGroup]:
        """Get a specific tool group by name."""
        return self._groups.get(name)

    def list_groups(self) -> list[str]:
        """List all registered group names."""
        return list(self._groups.keys())

    def get_tool_descriptions(self) -> str:
        """Generate tool descriptions for system prompt."""
        lines = []
        for name, group in self._groups.items():
            lines.append(f"\n## {name.replace('_', ' ').title()} Tools")
            lines.append(group.description)
            for tool in group.tools:
                tool_name = getattr(tool, "name", tool.__name__ if callable(tool) else str(tool))
                tool_desc = getattr(tool, "description", "")[:100]
                lines.append(f"- `{tool_name}`: {tool_desc}")
        return "\n".join(lines)


def create_default_registry(user_id: Optional[str] = None) -> ToolRegistry:
    """
    Factory function to create a registry with all default tools.

    Args:
        user_id: User ID for user-specific tools (e.g., memory).

    Returns:
        ToolRegistry populated with all available tool groups.
    """
    registry = ToolRegistry()

    # =========================================================================
    # COGNEE KNOWLEDGE GRAPH TOOLS
    # =========================================================================
    try:
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
        )

        registry.register_group(
            ToolGroup(
                name="cognee",
                tools=[
                    find_entity,
                    search_text,
                    query_graph,
                    get_graph_schema,
                    search_chunks,
                    search_summaries,
                    search_graph,
                    search_rag,
                ],
                description="Knowledge graph and RAG tools for searching program documents, "
                "finding entities, and querying relationships.",
            )
        )
        logger.info("Registered cognee tool group")
    except ImportError as e:
        logger.warning(f"Could not import cognee tools: {e}")

    # =========================================================================
    # MENTOR HUB LIVE DATA TOOLS
    # =========================================================================
    try:
        from app.tools.mentor_hub_tools import get_mentor_hub_tools

        mentor_hub_tools = get_mentor_hub_tools()
        registry.register_group(
            ToolGroup(
                name="mentor_hub",
                tools=mentor_hub_tools,
                description="Live Mentor Hub data including sessions, teams, tasks, and mentors. "
                "Use these for current/real-time information.",
            )
        )
        logger.info("Registered mentor_hub tool group")
    except ImportError as e:
        logger.warning(f"Could not import mentor_hub tools: {e}")

    # =========================================================================
    # FIRECRAWL WEB TOOLS
    # =========================================================================
    try:
        from app.tools.firecrawl_tools import (
            firecrawl_scrape,
            firecrawl_search,
            firecrawl_map,
            firecrawl_crawl,
            firecrawl_extract,
        )

        registry.register_group(
            ToolGroup(
                name="firecrawl",
                tools=[
                    firecrawl_scrape,
                    firecrawl_search,
                    firecrawl_map,
                    firecrawl_crawl,
                    firecrawl_extract,
                ],
                description="Web scraping and search tools for external research. "
                "Use for questions requiring information from the web.",
            )
        )
        logger.info("Registered firecrawl tool group")
    except ImportError as e:
        logger.warning(f"Could not import firecrawl tools: {e}")

    # =========================================================================
    # OUTLINE MCP TOOLS
    # =========================================================================
    try:
        from app.tools.outline_mcp_tools import get_outline_mcp_tools

        outline_tools = get_outline_mcp_tools()
        if outline_tools:
            registry.register_group(
                ToolGroup(
                    name="outline_mcp",
                    tools=outline_tools,
                    description="Outline knowledge base tools (collections, documents, search). "
                    "Use to find and read documents in Outline.",
                )
            )
            logger.info("Registered outline_mcp tool group")
    except ImportError as e:
        logger.warning(f"Could not import outline MCP tools: {e}")

    # =========================================================================
    # USER MEMORY TOOLS (requires user_id)
    # =========================================================================
    if user_id:
        try:
            from app.tools.user_memory import get_user_memory_tools

            memory_tools = get_user_memory_tools(user_id)
            if memory_tools:
                registry.register_group(
                    ToolGroup(
                        name="memory",
                        tools=memory_tools,
                        description="User-specific memory for storing and recalling facts "
                        "about the user across conversations.",
                    )
                )
                logger.info("Registered memory tool group")
        except ImportError as e:
            logger.warning(f"Could not import memory tools: {e}")

    # =========================================================================
    # USER INTERACTION TOOLS (clarifications, UI prompts)
    # =========================================================================
    try:
        from app.tools.clarification_tools import get_clarification_tools

        clarification_tools = get_clarification_tools()
        if clarification_tools:
            registry.register_group(
                ToolGroup(
                    name="interaction",
                    tools=clarification_tools,
                    description="User clarification and structured prompt tools. "
                    "Use to ask multiple-choice questions when details are missing.",
                )
            )
            logger.info("Registered interaction tool group")
    except ImportError as e:
        logger.warning(f"Could not import interaction tools: {e}")

    return registry


# Convenience functions for common tool selections
def get_research_tools(registry: ToolRegistry) -> list:
    """Get tools optimized for research tasks (cognee + firecrawl)."""
    return registry.get_tools("cognee", "firecrawl")


def get_mentor_hub_tools_from_registry(registry: ToolRegistry) -> list:
    """Get Mentor Hub specific tools."""
    return registry.get_tools("mentor_hub")


def get_all_tools(registry: ToolRegistry) -> list:
    """Get all available tools."""
    return registry.get_tools()
