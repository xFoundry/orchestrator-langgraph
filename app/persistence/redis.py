"""Redis checkpointer and store management for LangGraph persistence."""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

from app.config import get_settings

logger = logging.getLogger(__name__)

# Global checkpointer instance and context manager
_checkpointer: Optional[BaseCheckpointSaver] = None
_checkpointer_cm: Optional[Any] = None
_store: Optional[BaseStore] = None
_store_cm: Optional[Any] = None


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


async def get_store() -> BaseStore:
    """
    Get or create the LangGraph store singleton.

    Uses Redis if available, otherwise falls back to in-memory storage.
    """
    global _store, _store_cm

    if _store is not None:
        return _store

    settings = get_settings()

    use_memory = os.getenv("USE_MEMORY_STORE", "false").lower() == "true"
    postgres_url = os.getenv("POSTGRES_URL") or settings.postgres_url
    if use_memory or settings.redis_url == "memory://":
        logger.info("Using in-memory store (memories won't persist across restarts)")
        _store = InMemoryStore()
        return _store

    if postgres_url:
        try:
            from langgraph.store.postgres import AsyncPostgresStore

            logger.info("Initializing Postgres store")
            _store_cm = AsyncPostgresStore.from_conn_string(postgres_url)
            _store = await _store_cm.__aenter__()
            await _store.setup()
            logger.info("Postgres store initialized successfully")
            return _store
        except Exception as e:
            logger.warning(f"Postgres store not available ({e}), falling back to Redis store")

    try:
        try:
            from langgraph.store.redis.aio import AsyncRedisStore
        except ImportError:
            from langgraph.store.redis import AsyncRedisStore

        logger.info(f"Initializing Redis store at {settings.redis_url}")
        _store_cm = AsyncRedisStore.from_conn_string(settings.redis_url)
        _store = await _store_cm.__aenter__()
        logger.info("Redis store initialized successfully")
        return _store

    except Exception as e:
        logger.warning(f"Redis store not available ({e}), using in-memory store")
        _store = InMemoryStore()
        return _store


async def close_store() -> None:
    """Close the store connection if it was initialized."""
    global _store, _store_cm

    if _store_cm is not None:
        logger.info("Closing Redis store connection")
        try:
            await _store_cm.__aexit__(None, None, None)
        except Exception as e:
            logger.warning(f"Error closing store: {e}")
        _store_cm = None

    _store = None
    logger.info("Store closed")


def get_thread_config(
    thread_id: str,
    user_id: str = "anonymous",
    tenant_id: str = "default",
    recursion_limit: int | None = None,
) -> dict:
    """
    Create LangGraph config with thread ID for session persistence.

    Args:
        thread_id: Unique identifier for the conversation thread
        user_id: User identifier for personalization
        tenant_id: Tenant identifier for multi-tenancy
        recursion_limit: Maximum graph steps before termination (default from settings)

    Returns:
        Configuration dict for LangGraph invoke/stream calls
    """
    from app.config import get_settings

    settings = get_settings()
    limit = recursion_limit if recursion_limit is not None else settings.recursion_limit

    return {
        "configurable": {
            "thread_id": thread_id,
            "user_id": user_id,
            "tenant_id": tenant_id,
        },
        "recursion_limit": limit,
    }
