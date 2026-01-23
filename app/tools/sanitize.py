"""
Sanitization utilities for LangGraph tool outputs.

Handles two types of serialization issues:
1. Redis JSON module rejects unescaped control characters (\\u0000-\\u001F).
   Data from external sources (APIs, web scraping, databases) may contain these.
2. SSE streaming fails on non-JSON-serializable objects like AsyncCallbackManager.
   LangChain/LangGraph tools may have internal objects attached.

This module provides sanitization functions to clean data before it enters
the LangGraph state or gets serialized for SSE streaming.

Also provides MCP tool interceptors for sanitizing tool outputs at the source.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Awaitable, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)


def sanitize_for_json(value: Any) -> Any:
    """
    Remove control characters from text that would break JSON serialization.

    Recursively sanitizes strings, dicts, and lists. Safe to call on any value.

    Args:
        value: Any value (string, dict, list, or other)

    Returns:
        Sanitized value with control characters removed from strings
    """
    if value is None:
        return None
    if isinstance(value, str):
        # Remove control characters except:
        # - \\t (tab, \\x09)
        # - \\n (newline, \\x0a)
        # - \\r (carriage return, \\x0d)
        # These are valid in JSON when properly escaped
        return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', value)
    if isinstance(value, dict):
        return {k: sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(item) for item in value]
    return value


def make_json_serializable(value: Any) -> Any:
    """
    Convert any value to a JSON-serializable form.

    Handles non-serializable objects (like AsyncCallbackManager) by converting
    them to string placeholders. Also sanitizes control characters.

    Use this for SSE streaming where tool inputs may contain LangChain internals.

    Args:
        value: Any value that needs to be JSON serialized

    Returns:
        JSON-serializable value
    """
    if value is None:
        return None
    if isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', value)
    if isinstance(value, dict):
        return {k: make_json_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_serializable(item) for item in value]
    # For any other type, check if it's serializable
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        # Non-serializable object - return type name as placeholder
        return f"<{type(value).__name__}>"


def wrap_tool_with_sanitization(tool: "BaseTool") -> "BaseTool":
    """
    Wrap a LangChain tool to sanitize its output for JSON serialization.

    MCP tool responses may contain control characters or invalid escape
    sequences that break Redis JSON checkpointing. This wrapper ensures
    all outputs are sanitized before entering the LangGraph state.

    Args:
        tool: A LangChain BaseTool instance

    Returns:
        A new tool with sanitized outputs
    """
    from langchain_core.tools import StructuredTool

    original_coroutine = getattr(tool, 'coroutine', None)
    original_func = getattr(tool, 'func', None)

    async def sanitized_acall(*args: Any, **kwargs: Any) -> Any:
        """Async call with output sanitization."""
        if original_coroutine:
            result = await original_coroutine(*args, **kwargs)
        elif original_func:
            result = original_func(*args, **kwargs)
        else:
            result = await tool.ainvoke(*args, **kwargs)
        return sanitize_for_json(result)

    def sanitized_call(*args: Any, **kwargs: Any) -> Any:
        """Sync call with output sanitization."""
        if original_func:
            result = original_func(*args, **kwargs)
        else:
            result = tool.invoke(*args, **kwargs)
        return sanitize_for_json(result)

    return StructuredTool(
        name=tool.name,
        description=tool.description or "",
        func=sanitized_call,
        coroutine=sanitized_acall,
        args_schema=getattr(tool, 'args_schema', None),
    )


class SanitizationInterceptor:
    """
    MCP tool interceptor that sanitizes tool output for JSON serialization.

    Intercepts tool call results and removes control characters that would
    break Redis JSON checkpointing. This is the proper way to sanitize
    MCP tool outputs using the langchain-mcp-adapters interceptor pattern.

    Usage:
        client = MultiServerMCPClient(
            servers,
            tool_interceptors=[SanitizationInterceptor()],
        )
    """

    async def __call__(
        self,
        request: Any,  # MCPToolCallRequest
        handler: Callable[[Any], Awaitable[Any]],  # Returns MCPToolCallResult
    ) -> Any:
        """
        Intercept and sanitize MCP tool call results.

        Args:
            request: The MCP tool call request
            handler: The next handler in the chain (executes the actual tool)

        Returns:
            MCPToolCallResult with sanitized content
        """
        # Execute the tool call
        result = await handler(request)

        # Sanitize the result content
        try:
            if hasattr(result, 'content') and result.content:
                sanitized_content = []
                for content_item in result.content:
                    if hasattr(content_item, 'text'):
                        # Sanitize the text content
                        sanitized_text = sanitize_for_json(content_item.text)
                        # Create a new content item with sanitized text
                        # MCP uses TextContent with type and text attributes
                        from mcp.types import TextContent
                        sanitized_content.append(
                            TextContent(type="text", text=sanitized_text)
                        )
                    else:
                        # Keep non-text content as-is
                        sanitized_content.append(content_item)

                # Create new result with sanitized content
                # MCP returns CallToolResult with content and isError attributes
                from mcp.types import CallToolResult
                return CallToolResult(
                    content=sanitized_content,
                    isError=getattr(result, 'isError', False),
                )
        except Exception as e:
            logger.warning(f"Failed to sanitize MCP tool result: {e}")

        return result
