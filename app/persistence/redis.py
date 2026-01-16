"""Redis checkpointer and store management for LangGraph persistence."""

from __future__ import annotations

import logging
import os
from typing import Any, Optional, Union

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver

from app.config import get_settings

logger = logging.getLogger(__name__)

# Global checkpointer instance and context manager
_checkpointer: Optional[BaseCheckpointSaver] = None
_checkpointer_cm: Optional[Any] = None


async def get_checkpointer() -> BaseCheckpointSaver:
    """
    Get or create the checkpointer singleton.

    Uses Redis Stack if available (requires RediSearch module),
    otherwise falls back to in-memory storage for local development.
    """
    global _checkpointer, _checkpointer_cm

    if _checkpointer is not None:
        return _checkpointer

    settings = get_settings()

    # Check if we should use memory (for local dev without Redis Stack)
    use_memory = os.getenv("USE_MEMORY_CHECKPOINTER", "false").lower() == "true"

    if use_memory or settings.redis_url == "memory://":
        logger.info("Using in-memory checkpointer (sessions won't persist across restarts)")
        _checkpointer = MemorySaver()
        return _checkpointer

    # Try Redis Stack (requires RediSearch module)
    try:
        from langgraph.checkpoint.redis.aio import AsyncRedisSaver

        logger.info(f"Initializing Redis Stack checkpointer at {settings.redis_url}")

        # from_conn_string returns an async context manager
        _checkpointer_cm = AsyncRedisSaver.from_conn_string(settings.redis_url)
        _checkpointer = await _checkpointer_cm.__aenter__()

        logger.info("Redis Stack checkpointer initialized successfully")
        return _checkpointer

    except Exception as e:
        # Fall back to memory if Redis Stack isn't available
        logger.warning(f"Redis Stack not available ({e}), falling back to in-memory checkpointer")
        logger.info("To use Redis persistence, run: docker run -d -p 6379:6379 redis/redis-stack-server")
        _checkpointer = MemorySaver()
        return _checkpointer


async def close_checkpointer() -> None:
    """Close the checkpointer connection."""
    global _checkpointer, _checkpointer_cm

    if _checkpointer_cm is not None:
        logger.info("Closing Redis checkpointer connection")
        try:
            await _checkpointer_cm.__aexit__(None, None, None)
        except Exception as e:
            logger.warning(f"Error closing checkpointer: {e}")
        _checkpointer_cm = None

    _checkpointer = None
    logger.info("Checkpointer closed")


def get_thread_config(
    thread_id: str,
    user_id: str = "anonymous",
    tenant_id: str = "default",
) -> dict:
    """
    Create LangGraph config with thread ID for session persistence.

    Args:
        thread_id: Unique identifier for the conversation thread
        user_id: User identifier for personalization
        tenant_id: Tenant identifier for multi-tenancy

    Returns:
        Configuration dict for LangGraph invoke/stream calls
    """
    return {
        "configurable": {
            "thread_id": thread_id,
            "user_id": user_id,
            "tenant_id": tenant_id,
        }
    }
