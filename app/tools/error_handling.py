"""
Error handling wrapper for tools.

Wraps tools to catch exceptions and return error messages as results,
allowing the LLM to see errors and potentially retry with a different approach.

Also sanitizes tool outputs to prevent JSON serialization errors in Redis checkpoints.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Type, Union

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from app.tools.sanitize import sanitize_for_json

logger = logging.getLogger(__name__)

# Type alias for tool call schemas (Pydantic model or dict JSON schema)
ArgsSchema = Union[Type[BaseModel], dict[str, Any]]


class ErrorHandlingToolWrapper(BaseTool):
    """
    Wraps a tool to catch exceptions and return them as structured error results.

    Instead of allowing exceptions to propagate and kill the stream,
    this wrapper catches them and returns an error dict that the LLM can see.

    The LLM can then decide to:
    - Retry the same tool with different parameters
    - Try a different approach (e.g., sequential instead of parallel)
    - Report the error to the user

    Implementation notes:
    - We set name, description, args_schema as static attributes during __init__
      to avoid @property which causes infinite recursion in LangChain's schema inspector.
    - For args_schema, MCP tools may return a dict (JSON Schema) instead of a BaseModel
      subclass. We set args_schema=None to pass Pydantic validation, then override
      tool_call_schema property to delegate to the wrapped tool so the LLM sees the
      correct parameter schema including required fields.
    """

    wrapped_tool: BaseTool
    """The underlying tool being wrapped."""

    # Static attributes set during init (NOT @property - causes recursion in LangChain)
    name: str = ""
    description: str = ""
    args_schema: Optional[Type[BaseModel]] = None

    def __init__(self, wrapped_tool: BaseTool, **kwargs: Any) -> None:
        """Initialize the wrapper with static copies of the wrapped tool's attributes."""
        # Get args_schema, but only if it's a proper BaseModel subclass
        # MCP tools may return a raw dict (JSON Schema) which Pydantic rejects
        schema = wrapped_tool.args_schema
        if schema is not None and not (isinstance(schema, type) and issubclass(schema, BaseModel)):
            # Not a valid BaseModel subclass (likely a dict from MCP)
            # Set to None to pass Pydantic validation; tool_call_schema handles the rest
            schema = None

        super().__init__(
            wrapped_tool=wrapped_tool,
            name=wrapped_tool.name,
            description=wrapped_tool.description,
            args_schema=schema,
            **kwargs,
        )

    @property
    def tool_call_schema(self) -> ArgsSchema:
        """
        Delegate to the wrapped tool's tool_call_schema.

        Critical: LangChain's convert_to_openai_tool uses this property to generate
        the schema sent to the LLM. For MCP tools with dict schemas, the wrapped tool
        knows how to return them correctly. By delegating here instead of using our
        args_schema (which may be None), we preserve the full schema including
        required fields.
        """
        return self.wrapped_tool.tool_call_schema

    def get_input_schema(self, config: Optional[Any] = None) -> Type[BaseModel]:
        """
        Delegate schema generation to the wrapped tool.

        This is used for some internal LangChain operations but tool_call_schema
        is what actually gets sent to the LLM.
        """
        return self.wrapped_tool.get_input_schema(config)

    def _run(
        self,
        *args: Any,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Any:
        """Run the wrapped tool with error handling and output sanitization."""
        try:
            # Pass all arguments through, including config for MCP tools
            result = self.wrapped_tool._run(
                *args, run_manager=run_manager, config=config, **kwargs
            )
            # Sanitize output to prevent JSON serialization errors in Redis checkpoints
            return sanitize_for_json(result)
        except Exception as e:
            return self._format_error(e)

    async def _arun(
        self,
        *args: Any,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Any:
        """Async run the wrapped tool with error handling and output sanitization."""
        try:
            # Pass all arguments through, including config for MCP tools
            result = await self.wrapped_tool._arun(
                *args, run_manager=run_manager, config=config, **kwargs
            )
            # Sanitize output to prevent JSON serialization errors in Redis checkpoints
            return sanitize_for_json(result)
        except Exception as e:
            return self._format_error(e)

    def _format_error(self, error: Exception) -> dict[str, Any]:
        """Format an exception into a structured error response."""
        error_str = str(error).lower()
        error_message = str(error)

        # Parse specific error types for better LLM guidance
        if "not_allowed" in error_str:
            # Wrike concurrent request limit or permissions issue
            if "insufficient user rights" in error_str or "license limitations" in error_str:
                suggestion = (
                    "This error may occur when making parallel requests to Wrike task endpoints. "
                    "Try calling this tool sequentially instead of in parallel with other Wrike tools."
                )
            else:
                suggestion = "This action is not allowed. Check permissions or try a different approach."
        elif "429" in error_str or "rate limit" in error_str or "too_many_requests" in error_str:
            suggestion = "Rate limit exceeded. Wait a moment before retrying, or reduce parallel calls."
        elif "401" in error_str or "unauthorized" in error_str:
            suggestion = "Authentication failed. The integration credentials may be invalid or expired."
        elif "403" in error_str or "forbidden" in error_str:
            suggestion = "Access forbidden. Check if the API token has the required permissions."
        elif "404" in error_str or "not found" in error_str:
            suggestion = "Resource not found. Verify the ID or name is correct."
        elif "timeout" in error_str or "timed out" in error_str:
            suggestion = "Request timed out. Try again or simplify the request."
        elif "connection" in error_str:
            suggestion = "Connection error. The service may be temporarily unavailable."
        else:
            suggestion = "Try a different approach or parameters."

        logger.warning(
            f"Tool '{self.name}' failed with error: {error_message}",
            exc_info=True,
        )

        return {
            "error": True,
            "error_type": type(error).__name__,
            "error_message": error_message[:500],  # Truncate for context window
            "suggestion": suggestion,
            "tool_name": self.name,
        }


def wrap_tool_with_error_handling(tool: BaseTool) -> BaseTool:
    """
    Wrap a tool with error handling.

    Args:
        tool: The tool to wrap.

    Returns:
        A wrapped tool that catches exceptions and returns error dicts.
    """
    return ErrorHandlingToolWrapper(wrapped_tool=tool)


def wrap_tools_with_error_handling(tools: list[BaseTool]) -> list[BaseTool]:
    """
    Wrap multiple tools with error handling.

    Args:
        tools: List of tools to wrap.

    Returns:
        List of wrapped tools.
    """
    return [wrap_tool_with_error_handling(tool) for tool in tools]
