"""Redis checkpointer and store management for LangGraph persistence."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional, Sequence

from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    PendingWrite,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langchain_core.runnables import RunnableConfig

from app.config import get_settings

logger = logging.getLogger(__name__)

# Regex to remove control characters that crash Redis/JSON
# Keeps: \t (tab, 0x09), \n (newline, 0x0A), \r (carriage return, 0x0D)
# Removes: 0x00-0x08, 0x0B, 0x0C, 0x0E-0x1F, 0x7F (DEL)
import re
CONTROL_CHAR_REGEX = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
# Also match escaped Unicode control characters in JSON strings (e.g., \u0000, \u001f)
# These survive json.dumps and need to be removed from the JSON string itself
# Keep: \u0009 (tab), \u000a (newline), \u000d (carriage return)
# Remove: \u0000-\u0008, \u000b, \u000c, \u000e-\u001f, \u007f
def _remove_escaped_control_chars(json_str: str) -> str:
    """Remove escaped Unicode control characters from JSON string, keeping tab/newline/CR."""
    def replacer(match: re.Match) -> str:
        # Get the matched escape sequence like \u0000
        seq = match.group(0).lower()
        # Keep tab, newline, carriage return
        if seq in ('\\u0009', '\\u000a', '\\u000d'):
            return match.group(0)
        # Remove all other control characters
        return ''
    # Match all \u00XX patterns where XX is 00-1f or 7f
    return re.sub(r'\\u00[0-1][0-9a-fA-F]|\\u007[fF]', replacer, json_str, flags=re.IGNORECASE)

# Global checkpointer instance and context manager
_checkpointer: Optional[BaseCheckpointSaver] = None
_checkpointer_cm: Optional[Any] = None
_store: Optional[BaseStore] = None
_store_cm: Optional[Any] = None


class SanitizingCheckpointSaver(BaseCheckpointSaver):
    """
    Checkpoint saver wrapper that sanitizes data before writing to Redis.

    Redis JSON module rejects unescaped control characters (\\u0000-\\u001F).
    Data from external APIs, web scraping, or tool outputs may contain these
    characters. This wrapper sanitizes all channel_values before checkpoint
    writes to prevent Redis JSON errors.

    This is the proper fix for errors like:
    "Command JSON.SET caused error: ('invalid escape at line 1 column N',)"
    """

    def __init__(self, wrapped: BaseCheckpointSaver):
        """
        Initialize with a wrapped checkpointer.

        Args:
            wrapped: The underlying checkpointer (e.g., AsyncRedisSaver)
        """
        # Don't call super().__init__() as it may have required args
        self._wrapped = wrapped
        # Copy serde from wrapped if available
        if hasattr(wrapped, 'serde'):
            self.serde = wrapped.serde

    @property
    def config_specs(self):
        """Delegate config_specs to wrapped checkpointer."""
        return self._wrapped.config_specs

    def _sanitize_value(self, value: Any, depth: int = 0) -> Any:
        """
        Recursively sanitize a value, removing control characters from strings.

        IMPORTANT: Unlike sanitize_for_json which uses JSON round-trip with default=str,
        this method preserves object types (like LangChain messages). It only touches
        actual strings, leaving complex objects intact for proper serialization.

        Args:
            value: Any value to sanitize
            depth: Current recursion depth (prevents infinite recursion)

        Returns:
            Value with control characters removed from any string content
        """
        # Prevent infinite recursion on circular references
        if depth > 100:
            return value

        if value is None:
            return None

        if isinstance(value, str):
            # Remove raw control characters
            sanitized = CONTROL_CHAR_REGEX.sub('', value)
            # Also check for escaped sequences that might cause issues when re-serialized
            # These can appear if the string was previously JSON-encoded
            if '\\u00' in sanitized or '\\u007' in sanitized:
                sanitized = _remove_escaped_control_chars(sanitized)
            return sanitized

        if isinstance(value, bytes):
            # Bytes might contain control chars when decoded
            try:
                decoded = value.decode('utf-8', errors='replace')
                return CONTROL_CHAR_REGEX.sub('', decoded).encode('utf-8')
            except Exception:
                return value

        if isinstance(value, dict):
            return {self._sanitize_value(k, depth + 1): self._sanitize_value(v, depth + 1)
                    for k, v in value.items()}

        if isinstance(value, list):
            return [self._sanitize_value(item, depth + 1) for item in value]

        if isinstance(value, tuple):
            return tuple(self._sanitize_value(item, depth + 1) for item in value)

        if isinstance(value, set):
            return {self._sanitize_value(item, depth + 1) for item in value}

        # For primitive types, return as-is
        if isinstance(value, (int, float, bool, type(None))):
            return value

        # For objects with known string attributes (LangChain messages, Pydantic models, etc.)
        # Try to sanitize any string attributes we can find
        try:
            # Handle objects with __dict__ (regular Python objects)
            if hasattr(value, '__dict__'):
                for attr_name in list(vars(value).keys()):
                    try:
                        attr_value = getattr(value, attr_name)
                        if isinstance(attr_value, str):
                            sanitized = CONTROL_CHAR_REGEX.sub('', attr_value)
                            setattr(value, attr_name, sanitized)
                        elif isinstance(attr_value, (dict, list, tuple)):
                            sanitized = self._sanitize_value(attr_value, depth + 1)
                            setattr(value, attr_name, sanitized)
                    except (AttributeError, TypeError):
                        # Attribute is read-only or can't be set, skip
                        pass

            # Handle Pydantic models (v1 and v2)
            if hasattr(value, 'model_fields') or hasattr(value, '__fields__'):
                fields = getattr(value, 'model_fields', None) or getattr(value, '__fields__', {})
                for field_name in fields:
                    try:
                        field_value = getattr(value, field_name, None)
                        if isinstance(field_value, str):
                            sanitized = CONTROL_CHAR_REGEX.sub('', field_value)
                            # Try direct assignment first
                            try:
                                setattr(value, field_name, sanitized)
                            except Exception:
                                # For frozen models, try __dict__ assignment
                                if hasattr(value, '__dict__'):
                                    value.__dict__[field_name] = sanitized
                        elif field_value is not None:
                            self._sanitize_value(field_value, depth + 1)
                    except Exception:
                        pass

            # Handle dataclasses
            if hasattr(value, '__dataclass_fields__'):
                for field_name in value.__dataclass_fields__:
                    try:
                        field_value = getattr(value, field_name)
                        if isinstance(field_value, str):
                            sanitized = CONTROL_CHAR_REGEX.sub('', field_value)
                            object.__setattr__(value, field_name, sanitized)
                        elif field_value is not None:
                            self._sanitize_value(field_value, depth + 1)
                    except Exception:
                        pass

        except Exception as e:
            logger.debug(f"Could not sanitize object attributes: {e}")

        return value

    def _sanitize_checkpoint(self, checkpoint: Checkpoint) -> Checkpoint:
        """
        Sanitize checkpoint data to remove JSON-breaking control characters.

        Uses recursive sanitization that preserves LangChain message objects
        while removing control characters from string content.

        Args:
            checkpoint: The checkpoint dict to sanitize

        Returns:
            Sanitized checkpoint with control characters removed
        """
        if not checkpoint:
            return checkpoint

        # Create a shallow copy to avoid mutating the original
        sanitized = dict(checkpoint)

        # Sanitize channel_values - this is where tool results and messages live
        if "channel_values" in sanitized:
            try:
                sanitized["channel_values"] = self._sanitize_value(sanitized["channel_values"])
            except Exception as e:
                logger.warning(f"Failed to sanitize channel_values: {e}")

        # Also sanitize pending_sends if present
        if "pending_sends" in sanitized:
            try:
                sanitized["pending_sends"] = self._sanitize_value(sanitized["pending_sends"])
            except Exception as e:
                logger.warning(f"Failed to sanitize pending_sends: {e}")

        return sanitized

    def _sanitize_metadata(self, metadata: CheckpointMetadata) -> CheckpointMetadata:
        """
        Sanitize checkpoint metadata to remove control characters.

        Args:
            metadata: The metadata dict to sanitize

        Returns:
            Sanitized metadata with control characters removed
        """
        if not metadata:
            return metadata

        return self._sanitize_value(dict(metadata))

    def _aggressive_sanitize(self, value: Any) -> Any:
        """
        Aggressive JSON-level sanitization as a fallback.

        Serializes the value to JSON (converting any non-serializable objects to strings),
        sanitizes the resulting JSON string, and deserializes back.

        This catches control characters that slip through object-level sanitization,
        such as those in complex nested structures or custom objects.

        Args:
            value: Any value to sanitize

        Returns:
            Sanitized value with all control characters removed
        """
        if value is None:
            return None

        try:
            # Serialize to JSON, converting non-serializable objects to strings
            # Use ensure_ascii=True to escape all non-ASCII chars (safer for Redis)
            def default_handler(obj):
                try:
                    # Try to get dict representation
                    if hasattr(obj, '__dict__'):
                        return obj.__dict__
                    # Try to get string representation
                    return str(obj)
                except Exception:
                    return f"<{type(obj).__name__}>"

            json_str = json.dumps(value, default=default_handler, ensure_ascii=True)

            # Sanitize raw control characters (shouldn't be present after json.dumps, but just in case)
            sanitized_json = CONTROL_CHAR_REGEX.sub('', json_str)

            # Also remove escaped Unicode control characters like \u0000, \u001f, \u007f
            # These are the JSON-escaped forms that Redis JSON will reject
            # But keep \u0009 (tab), \u000a (newline), \u000d (carriage return)
            sanitized_json = _remove_escaped_control_chars(sanitized_json)

            # Deserialize back
            return json.loads(sanitized_json)

        except Exception as e:
            logger.warning(f"Aggressive sanitization failed, returning original: {e}")
            return value

    def _sanitize_writes(self, writes: Sequence[tuple[str, Any]]) -> list[tuple[str, Any]]:
        """Sanitize pending writes data using type-preserving sanitization."""
        sanitized_writes = []
        for channel, value in writes:
            try:
                sanitized_value = self._sanitize_value(value)
                sanitized_writes.append((channel, sanitized_value))
            except Exception as e:
                logger.warning(f"Failed to sanitize write for channel {channel}: {e}")
                sanitized_writes.append((channel, value))
        return sanitized_writes

    # Delegate all read operations to wrapped checkpointer
    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        return self._wrapped.get_tuple(config)

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        return await self._wrapped.aget_tuple(config)

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ):
        return self._wrapped.list(config, filter=filter, before=before, limit=limit)

    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ):
        async for item in self._wrapped.alist(config, filter=filter, before=before, limit=limit):
            yield item

    # Write operations - sanitize before delegating
    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        sanitized_checkpoint = self._sanitize_checkpoint(checkpoint)
        sanitized_metadata = self._sanitize_metadata(metadata)
        try:
            return self._wrapped.put(config, sanitized_checkpoint, sanitized_metadata, new_versions)
        except Exception as e:
            error_str = str(e).lower()
            if 'control character' in error_str or 'invalid escape' in error_str:
                logger.warning(f"Control character error in put, attempting aggressive sanitization: {e}")
                sanitized_checkpoint = self._aggressive_sanitize(sanitized_checkpoint)
                sanitized_metadata = self._aggressive_sanitize(sanitized_metadata)
                return self._wrapped.put(config, sanitized_checkpoint, sanitized_metadata, new_versions)
            raise

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        sanitized_checkpoint = self._sanitize_checkpoint(checkpoint)
        sanitized_metadata = self._sanitize_metadata(metadata)
        try:
            return await self._wrapped.aput(config, sanitized_checkpoint, sanitized_metadata, new_versions)
        except Exception as e:
            error_str = str(e).lower()
            if 'control character' in error_str or 'invalid escape' in error_str:
                # First sanitization wasn't enough - do aggressive JSON-level sanitization
                logger.warning(f"Control character error in aput, attempting aggressive sanitization: {e}")
                sanitized_checkpoint = self._aggressive_sanitize(sanitized_checkpoint)
                sanitized_metadata = self._aggressive_sanitize(sanitized_metadata)
                return await self._wrapped.aput(config, sanitized_checkpoint, sanitized_metadata, new_versions)
            raise

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        sanitized_writes = self._sanitize_writes(writes)
        try:
            return self._wrapped.put_writes(config, sanitized_writes, task_id, task_path)
        except Exception as e:
            error_str = str(e).lower()
            if 'control character' in error_str or 'invalid escape' in error_str:
                logger.warning(f"Control character error in put_writes, attempting aggressive sanitization: {e}")
                sanitized_writes = [(ch, self._aggressive_sanitize(v)) for ch, v in sanitized_writes]
                return self._wrapped.put_writes(config, sanitized_writes, task_id, task_path)
            raise

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        sanitized_writes = self._sanitize_writes(writes)
        try:
            return await self._wrapped.aput_writes(config, sanitized_writes, task_id, task_path)
        except Exception as e:
            error_str = str(e).lower()
            if 'control character' in error_str or 'invalid escape' in error_str:
                logger.warning(f"Control character error in aput_writes, attempting aggressive sanitization: {e}")
                sanitized_writes = [(ch, self._aggressive_sanitize(v)) for ch, v in sanitized_writes]
                return await self._wrapped.aput_writes(config, sanitized_writes, task_id, task_path)
            raise

    # Delegate setup method if available
    async def asetup(self) -> None:
        if hasattr(self._wrapped, 'asetup'):
            await self._wrapped.asetup()

    def setup(self) -> None:
        if hasattr(self._wrapped, 'setup'):
            self._wrapped.setup()

    # Delegate version methods required by LangGraph
    def get_next_version(self, current: Optional[str], channel: ChannelVersions) -> str:
        """Delegate version generation to wrapped checkpointer."""
        return self._wrapped.get_next_version(current, channel)

    # Delegate any other attributes/methods to wrapped checkpointer
    def __getattr__(self, name: str) -> Any:
        """Fallback: delegate any unknown attributes to the wrapped checkpointer."""
        return getattr(self._wrapped, name)


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
        redis_saver = await _checkpointer_cm.__aenter__()

        # Wrap with sanitizing layer to prevent JSON control character errors
        # This fixes: "Command JSON.SET caused error: ('invalid escape at line N column M',)"
        _checkpointer = SanitizingCheckpointSaver(redis_saver)

        logger.info("Redis Stack checkpointer initialized with sanitization layer")
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
