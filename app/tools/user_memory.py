"""
User Memory Tools - HTTP-based per-user memory via Cognee Service.

Provides memory tools that call the Cognee service's /memory endpoints.
Each user gets isolated memory storage based on their Auth0 ID.

Uses LangChain StructuredTool to wrap async functions as tools.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
from typing import Any

import httpx
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.config import get_settings

logger = logging.getLogger(__name__)


def _get_auth_headers(body_str: str) -> dict[str, str]:
    """Generate HMAC authentication headers for Cognee service."""
    settings = get_settings()
    if not settings.cognee_secret_key:
        return {"Content-Type": "application/json"}

    timestamp = str(int(time.time()))
    message = f"{timestamp}.{body_str}"
    signature = hmac.new(
        settings.cognee_secret_key.encode("utf-8"),
        message.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()
    return {
        "X-Timestamp": timestamp,
        "X-Signature": signature,
        "Content-Type": "application/json",
    }


# Tool input schemas
class RememberInput(BaseModel):
    """Input schema for remember tool."""

    data: str = Field(..., description="Information to store about the user")


class RecallInput(BaseModel):
    """Input schema for recall tool."""

    query: str = Field(..., description="Search query for user memories")


def get_user_memory_tools(user_id: str) -> list[StructuredTool]:
    """
    Get memory tools for a specific user.

    Creates LangChain StructuredTools that call the Cognee service's memory endpoints.
    These tools can be added to any LangGraph agent.

    Args:
        user_id: User's Auth0 ID (e.g., "auth0|abc123")

    Returns:
        List of [remember, recall] StructuredTools
    """
    settings = get_settings()

    async def remember_impl(data: str) -> str:
        """Store information in the user's personal memory."""
        payload = {
            "user_id": user_id,
            "data": data,
        }
        body_str = json.dumps(payload)

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{settings.cognee_api_url}/memory/add",
                    content=body_str,
                    headers=_get_auth_headers(body_str),
                )
                response.raise_for_status()
                result = response.json()
                return result.get("message", "Information stored successfully")

        except httpx.HTTPError as e:
            logger.error(f"Memory add failed for user {user_id[:20]}...: {e}")
            return f"Failed to store memory: {str(e)}"

    async def recall_impl(query: str) -> str:
        """Search the user's personal memory for relevant information."""
        payload = {
            "user_id": user_id,
            "query": query,
            "top_k": 10,
        }
        body_str = json.dumps(payload)

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{settings.cognee_api_url}/memory/search",
                    content=body_str,
                    headers=_get_auth_headers(body_str),
                )
                response.raise_for_status()
                result = response.json()

                results = result.get("results", [])
                if not results:
                    return "No memories found for this query."

                # Format results as readable text
                formatted = []
                for i, r in enumerate(results, 1):
                    if isinstance(r, dict):
                        content = r.get("content", r.get("text", str(r)))
                    else:
                        content = str(r)
                    formatted.append(f"{i}. {content}")

                return "\n".join(formatted)

        except httpx.HTTPError as e:
            logger.error(f"Memory search failed for user {user_id[:20]}...: {e}")
            return f"Failed to search memory: {str(e)}"

    # Create StructuredTools
    remember_tool = StructuredTool(
        name="remember",
        description=(
            "Store information in the user's personal memory. "
            "Use this to remember important facts, preferences, or context "
            "about the user for future conversations."
        ),
        func=lambda data: None,  # Sync placeholder (not used)
        coroutine=remember_impl,
        args_schema=RememberInput,
    )

    recall_tool = StructuredTool(
        name="recall",
        description=(
            "Search the user's personal memory for relevant information. "
            "Use this to recall facts, preferences, or context previously "
            "stored about the user."
        ),
        func=lambda query: None,  # Sync placeholder (not used)
        coroutine=recall_impl,
        args_schema=RecallInput,
    )

    logger.info(f"Created user memory tools for: {user_id[:20]}...")
    return [remember_tool, recall_tool]
