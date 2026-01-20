"""
Redis Store Backend - Persistent storage for Deep Agent filesystem.

Provides cross-session memory persistence via LangGraph's Store abstraction.
Files written to /memories/ are persisted in Redis and available across
different conversation threads.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from app.config import get_settings

logger = logging.getLogger(__name__)


class RedisStoreBackend:
    """
    Backend that persists files to Redis via LangGraph's Store abstraction.

    Uses namespace prefixes for user isolation:
    - /memories/{path} -> persistent cross-conversation memory
    - Each user gets their own namespace

    This backend is designed to work with Deep Agent's CompositeBackend
    to provide persistent storage for the /memories/ path while letting
    other paths use ephemeral StateBackend.
    """

    def __init__(
        self,
        store: Any,  # BaseStore from langgraph.store.base
        user_id: str,
        namespace_prefix: str = "deepagent",
    ) -> None:
        """
        Initialize the Redis store backend.

        Args:
            store: LangGraph Store instance (e.g., AsyncRedisStore)
            user_id: User ID for namespace isolation
            namespace_prefix: Prefix for all namespaces (default: "deepagent")
        """
        self.store = store
        self.user_id = user_id
        self.namespace_prefix = namespace_prefix

    def _get_namespace_and_key(self, path: str) -> tuple[str, str]:
        """
        Convert a file path to (namespace, key) for the store.

        Examples:
            /memories/facts.md -> (deepagent:memories:{user_id}, facts.md)
            /memories/projects/ai.md -> (deepagent:memories:{user_id}, projects/ai.md)
        """
        # Remove leading slash and split
        clean_path = path.lstrip("/")
        parts = clean_path.split("/", 1)

        if len(parts) == 2:
            folder, filename = parts
            namespace = f"{self.namespace_prefix}:{folder}:{self.user_id}"
            return namespace, filename
        else:
            # Single-level path
            namespace = f"{self.namespace_prefix}:{self.user_id}"
            return namespace, parts[0] if parts else ""

    async def read_file(self, path: str) -> Optional[str]:
        """
        Read a file from the Redis store.

        Args:
            path: File path (e.g., /memories/facts.md)

        Returns:
            File content as string, or None if not found
        """
        namespace, key = self._get_namespace_and_key(path)
        try:
            result = await self.store.aget(namespace, key)
            if result and hasattr(result, "value"):
                value = result.value
                if isinstance(value, dict):
                    return value.get("content")
                return str(value)
            return None
        except Exception as e:
            logger.warning(f"Failed to read {path} from store: {e}")
            return None

    async def write_file(self, path: str, content: str) -> bool:
        """
        Write a file to the Redis store.

        Args:
            path: File path (e.g., /memories/facts.md)
            content: File content to write

        Returns:
            True if successful, False otherwise
        """
        namespace, key = self._get_namespace_and_key(path)
        try:
            await self.store.aput(
                namespace,
                key,
                {"content": content, "path": path},
            )
            logger.debug(f"Wrote {len(content)} bytes to {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to write {path} to store: {e}")
            return False

    async def ls(self, path: str = "/") -> list[str]:
        """
        List files in a directory.

        Args:
            path: Directory path to list (e.g., /memories/)

        Returns:
            List of file names in the directory
        """
        namespace, _ = self._get_namespace_and_key(path)
        try:
            items = await self.store.asearch(namespace)
            return [item.key for item in items]
        except Exception as e:
            logger.warning(f"Failed to list {path}: {e}")
            return []

    async def delete_file(self, path: str) -> bool:
        """
        Delete a file from the store.

        Args:
            path: File path to delete

        Returns:
            True if successful, False otherwise
        """
        namespace, key = self._get_namespace_and_key(path)
        try:
            await self.store.adelete(namespace, key)
            logger.debug(f"Deleted {path}")
            return True
        except Exception as e:
            logger.warning(f"Failed to delete {path}: {e}")
            return False

    async def exists(self, path: str) -> bool:
        """Check if a file exists."""
        content = await self.read_file(path)
        return content is not None


def create_redis_store_backend(
    user_id: str,
    redis_url: Optional[str] = None,
) -> Optional[RedisStoreBackend]:
    """
    Factory function to create a Redis store backend.

    Args:
        user_id: User ID for namespace isolation
        redis_url: Redis connection URL (uses settings default if not provided)

    Returns:
        RedisStoreBackend instance, or None if Redis is not available
    """
    settings = get_settings()
    url = redis_url or settings.redis_url

    try:
        from langgraph.store.redis import AsyncRedisStore

        store = AsyncRedisStore.from_conn_string(url)
        return RedisStoreBackend(store=store, user_id=user_id)
    except ImportError:
        logger.warning("langgraph.store.redis not available, using ephemeral storage")
        return None
    except Exception as e:
        logger.warning(f"Could not create Redis store: {e}")
        return None
