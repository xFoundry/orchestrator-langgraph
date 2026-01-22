"""Thread management API endpoints.

Provides REST endpoints for:
- Listing threads for a user
- Getting thread details
- Getting thread messages (from checkpointer)
- Updating thread metadata (title, starred status)
- Deleting threads
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional, Any

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from app.persistence.redis import get_store, get_checkpointer, get_thread_config

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/threads", tags=["threads"])

# =============================================================================
# Models
# =============================================================================


class ThreadMetadata(BaseModel):
    """Thread metadata stored in LangGraph store."""

    id: str = Field(..., description="Thread ID")
    title: str = Field(..., description="Thread title (from first message or user-set)")
    created_at: str = Field(..., description="ISO timestamp of creation")
    updated_at: str = Field(..., description="ISO timestamp of last update")
    last_message_at: Optional[str] = Field(None, description="ISO timestamp of last message")
    message_count: int = Field(0, description="Number of messages in thread")
    is_starred: bool = Field(False, description="Whether thread is starred")
    is_archived: bool = Field(False, description="Whether thread is archived")
    archived_at: Optional[str] = Field(None, description="ISO timestamp when thread was archived")
    user_id: str = Field(..., description="Owner user ID")
    tenant_id: str = Field("default", description="Tenant ID")
    first_message: Optional[str] = Field(None, description="First user message (for title generation)")


class ThreadResponse(BaseModel):
    """Response for a single thread."""

    id: str
    title: str
    created_at: str
    updated_at: str
    last_message_at: Optional[str] = None
    message_count: int = 0
    is_starred: bool = False
    is_archived: bool = False


class ThreadListResponse(BaseModel):
    """Response for thread listing."""

    threads: list[ThreadResponse] = Field(default_factory=list)
    total: int = Field(0, description="Total number of threads")
    hasMore: bool = Field(False, description="Whether there are more threads to load")


class UpdateThreadRequest(BaseModel):
    """Request to update thread metadata."""

    title: Optional[str] = None
    is_starred: Optional[bool] = None
    is_archived: Optional[bool] = None


class ThreadMessage(BaseModel):
    """A message in a thread."""

    id: str
    role: str  # "user" | "assistant" | "system" | "tool"
    content: str | list[dict[str, Any]]
    tool_calls: Optional[list[dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None  # Tool name for tool messages
    created_at: Optional[str] = None


class ThreadMessagesResponse(BaseModel):
    """Response for thread messages."""

    thread_id: str
    messages: list[ThreadMessage] = Field(default_factory=list)


# =============================================================================
# Namespace Helpers
# =============================================================================


def get_thread_metadata_namespace(tenant_id: str, user_id: str) -> tuple[str, ...]:
    """Get namespace for thread metadata storage."""
    return (tenant_id, "users", user_id, "thread_metadata")


# =============================================================================
# Internal Functions
# =============================================================================


async def get_or_create_thread_metadata(
    thread_id: str,
    user_id: str,
    tenant_id: str = "default",
    first_message: Optional[str] = None,
) -> ThreadMetadata:
    """
    Get existing thread metadata or create new entry.

    Called when a message is sent to ensure thread is tracked.
    """
    store = await get_store()
    namespace = get_thread_metadata_namespace(tenant_id, user_id)

    # Try to get existing metadata (await for async stores)
    result = await store.aget(namespace, thread_id)

    now = datetime.now(timezone.utc).isoformat()

    if result and result.value:
        # Update existing thread
        metadata = ThreadMetadata(**result.value)
        metadata.updated_at = now
        metadata.last_message_at = now
        metadata.message_count += 1

        # Update in store (await for async stores)
        await store.aput(namespace, thread_id, metadata.model_dump())
        logger.info(f"[Threads] UPDATED existing thread: id={thread_id}, title={metadata.title}, msg_count={metadata.message_count}")

        return metadata

    # Create new thread metadata
    # Generate title from first message or use default
    title = "New conversation"
    if first_message:
        # Use first 50 chars of message as title
        title = first_message[:50].strip()
        if len(first_message) > 50:
            title += "..."

    metadata = ThreadMetadata(
        id=thread_id,
        title=title,
        created_at=now,
        updated_at=now,
        last_message_at=now,
        message_count=1,
        is_starred=False,
        user_id=user_id,
        tenant_id=tenant_id,
        first_message=first_message,
    )

    # Store metadata (await for async stores)
    await store.aput(namespace, thread_id, metadata.model_dump())
    logger.info(f"[Threads] Created NEW thread metadata: id={thread_id}, title={metadata.title}, namespace={namespace}")

    # Verify it was stored
    verify = await store.aget(namespace, thread_id)
    if verify:
        logger.info(f"[Threads] ✓ Verified thread {thread_id} was stored successfully")
    else:
        logger.error(f"[Threads] ✗ FAILED to verify thread {thread_id} storage!")

    return metadata


async def update_thread_activity(
    thread_id: str,
    user_id: str,
    tenant_id: str = "default",
    message_text: Optional[str] = None,
) -> None:
    """
    Update thread activity timestamp and message count.

    Called after each message to keep metadata current.
    """
    await get_or_create_thread_metadata(
        thread_id=thread_id,
        user_id=user_id,
        tenant_id=tenant_id,
        first_message=message_text,
    )


# =============================================================================
# Endpoints
# =============================================================================


@router.get("/debug", response_model=dict)
async def debug_threads(
    user_id: str = Query(..., description="User ID"),
    tenant_id: str = Query("default", description="Tenant ID"),
    test_write: bool = Query(False, description="Test writing multiple items"),
):
    """DEBUG endpoint to see raw Redis data and test store behavior."""
    store = await get_store()
    namespace = get_thread_metadata_namespace(tenant_id, user_id)

    logger.critical(f"===== DEBUG_THREADS: namespace={namespace}, test_write={test_write} =====")

    # Optionally test writing multiple items
    if test_write:
        logger.critical("===== TESTING: Writing 3 test threads =====")
        for i in range(1, 4):
            test_thread_id = f"test-thread-{i}"
            test_data = {
                "id": test_thread_id,
                "title": f"Test Thread {i}",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "last_message_at": datetime.now(timezone.utc).isoformat(),
                "message_count": 1,
                "is_starred": False,
                "user_id": user_id,
                "tenant_id": tenant_id,
            }
            await store.aput(namespace, test_thread_id, test_data)
            logger.critical(f"  - Wrote {test_thread_id}")

        logger.critical("===== TESTING: Verifying writes =====")
        for i in range(1, 4):
            test_thread_id = f"test-thread-{i}"
            result = await store.aget(namespace, test_thread_id)
            if result:
                logger.critical(f"  - ✓ Found {test_thread_id}")
            else:
                logger.critical(f"  - ✗ NOT FOUND: {test_thread_id}")

    # Get ALL items without any filtering
    items = await store.asearch(namespace)

    raw_data = []
    for item in items:
        raw_data.append({
            "key": item.key,
            "value_type": str(type(item.value)),
            "value_keys": list(item.value.keys()) if isinstance(item.value, dict) else None,
            "value": item.value if isinstance(item.value, dict) else str(item.value)[:100]
        })

    logger.critical(f"===== DEBUG: Found {len(items)} raw items after operations =====")
    for data in raw_data:
        logger.critical(f"  - {data['key']}: {data.get('value_keys', 'N/A')}")

    return {
        "namespace": namespace,
        "total_items": len(items),
        "items": raw_data,
        "test_write_performed": test_write
    }


@router.get("", response_model=ThreadListResponse)
async def list_threads(
    user_id: str = Query(..., description="User ID"),
    tenant_id: str = Query("default", description="Tenant ID"),
    limit: int = Query(50, description="Maximum threads to return", ge=1, le=100),
    offset: int = Query(0, description="Offset for pagination", ge=0),
    include_archived: bool = Query(False, description="Include archived threads"),
    archived_only: bool = Query(False, description="Return only archived threads"),
):
    """
    List threads for a user.

    Returns threads sorted by last_message_at (most recent first).
    By default, archived threads are excluded. Use include_archived=true to include them,
    or archived_only=true to get only archived threads.
    """
    logger.critical(f"===== LIST_THREADS CALLED: user_id={user_id[:20]}..., tenant_id={tenant_id}, limit={limit}, offset={offset} =====")

    store = await get_store()
    namespace = get_thread_metadata_namespace(tenant_id, user_id)

    logger.debug(f"[Threads] Listing threads for user_id={user_id[:20]}..., namespace={namespace}")

    try:
        # Get all thread metadata for this user (use async search)
        items = await store.asearch(namespace)
        logger.info(f"[Threads] asearch returned {len(items)} raw items from namespace {namespace}")

        # Debug: log all item keys
        if items:
            logger.info(f"[Threads] Raw item keys: {[item.key for item in items]}")

        # Parse into ThreadMetadata objects with strict validation
        threads: list[ThreadMetadata] = []
        for item in items:
            # Skip items that don't have the expected thread ID format
            # Thread IDs should start with "thread-" to distinguish from artifacts
            if not item.key.startswith("thread-"):
                logger.debug(f"[Threads] Skipping non-thread item with key: {item.key}")
                continue

            # Validate that item.value is a dict and has required ThreadMetadata fields
            if not isinstance(item.value, dict):
                logger.debug(f"[Threads] Skipping item {item.key}: value is not a dict")
                continue

            required_fields = {"id", "title", "updated_at", "user_id"}
            if not required_fields.issubset(item.value.keys()):
                logger.debug(f"[Threads] Skipping item {item.key}: missing required fields {required_fields - item.value.keys()}")
                continue

            try:
                metadata = ThreadMetadata(**item.value)
                threads.append(metadata)
                logger.info(f"[Threads] Successfully parsed thread: {item.key} - title: {metadata.title}")
            except Exception as e:
                logger.warning(f"[Threads] Failed to parse thread metadata {item.key}: {e}")
                continue

        logger.info(f"[Threads] After filtering: {len(threads)} valid threads found")

        # Filter by archive status
        if archived_only:
            threads = [t for t in threads if t.is_archived]
            logger.info(f"[Threads] Filtered to archived only: {len(threads)} threads")
        elif not include_archived:
            threads = [t for t in threads if not t.is_archived]
            logger.info(f"[Threads] Filtered out archived: {len(threads)} threads")

        # Sort by last_message_at (most recent first)
        threads.sort(
            key=lambda t: t.last_message_at or t.updated_at,
            reverse=True
        )

        # Apply pagination
        total = len(threads)
        threads = threads[offset:offset + limit]
        has_more = (offset + len(threads)) < total

        # Convert to response format
        response_threads = [
            ThreadResponse(
                id=t.id,
                title=t.title,
                created_at=t.created_at,
                updated_at=t.updated_at,
                last_message_at=t.last_message_at,
                message_count=t.message_count,
                is_starred=t.is_starred,
                is_archived=t.is_archived,
            )
            for t in threads
        ]

        return ThreadListResponse(
            threads=response_threads,
            total=total,
            hasMore=has_more,
        )

    except Exception as e:
        logger.error(f"Error listing threads: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list threads: {e}")


@router.get("/{thread_id}", response_model=ThreadResponse)
async def get_thread(
    thread_id: str,
    user_id: str = Query(..., description="User ID"),
    tenant_id: str = Query("default", description="Tenant ID"),
):
    """Get thread metadata by ID."""
    store = await get_store()
    namespace = get_thread_metadata_namespace(tenant_id, user_id)

    result = await store.aget(namespace, thread_id)

    if not result or not result.value:
        raise HTTPException(status_code=404, detail=f"Thread not found: {thread_id}")

    metadata = ThreadMetadata(**result.value)

    return ThreadResponse(
        id=metadata.id,
        title=metadata.title,
        created_at=metadata.created_at,
        updated_at=metadata.updated_at,
        last_message_at=metadata.last_message_at,
        message_count=metadata.message_count,
        is_starred=metadata.is_starred,
        is_archived=metadata.is_archived,
    )


@router.get("/{thread_id}/messages", response_model=ThreadMessagesResponse)
async def get_thread_messages(
    thread_id: str,
    request: Request,
    user_id: str = Query(..., description="User ID"),
    tenant_id: str = Query("default", description="Tenant ID"),
):
    """
    Get messages from a thread's checkpoint.

    Retrieves the conversation history stored in the LangGraph checkpointer.
    This allows resuming conversations from where they left off.
    """
    checkpointer = request.app.state.checkpointer

    if not checkpointer:
        raise HTTPException(status_code=503, detail="Checkpointer not available")

    try:
        # Build config for the thread
        config = get_thread_config(
            thread_id=thread_id,
            user_id=user_id,
            tenant_id=tenant_id,
        )

        # Get the latest checkpoint for this thread
        # For async checkpointers, use aget_tuple
        checkpoint_tuple = None
        try:
            # Try async method first (Redis checkpointer)
            checkpoint_tuple = await checkpointer.aget_tuple(config)
        except AttributeError:
            # Fall back to sync method (MemorySaver)
            checkpoint_tuple = checkpointer.get_tuple(config)

        if not checkpoint_tuple or not checkpoint_tuple.checkpoint:
            logger.debug(f"No checkpoint found for thread {thread_id}")
            return ThreadMessagesResponse(thread_id=thread_id, messages=[])

        # Extract messages from checkpoint channel_values
        checkpoint = checkpoint_tuple.checkpoint
        channel_values = checkpoint.get("channel_values", {})
        raw_messages = channel_values.get("messages", [])

        logger.debug(f"Found {len(raw_messages)} messages in thread {thread_id}")

        # Convert to ThreadMessage format
        messages: list[ThreadMessage] = []
        for i, msg in enumerate(raw_messages):
            # Handle different message formats (LangChain messages, dicts, etc.)
            tool_calls = None
            tool_call_id = None
            tool_name = None

            if hasattr(msg, "type"):
                # LangChain message object
                role = msg.type  # "human" -> "user", "ai" -> "assistant", "tool" -> "tool"
                if role == "human":
                    role = "user"
                elif role == "ai":
                    role = "assistant"
                content = msg.content
                msg_id = getattr(msg, "id", f"{thread_id}-{i}")

                # Extract tool calls from AI messages
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    tool_calls = [
                        {
                            "id": tc.get("id", f"tool-{i}-{j}") if isinstance(tc, dict) else getattr(tc, "id", f"tool-{i}-{j}"),
                            "name": tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", ""),
                            "args": tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {}),
                        }
                        for j, tc in enumerate(msg.tool_calls)
                    ]

                # Extract tool call ID from tool messages
                if hasattr(msg, "tool_call_id"):
                    tool_call_id = msg.tool_call_id
                if hasattr(msg, "name"):
                    tool_name = msg.name

            elif isinstance(msg, dict):
                role = msg.get("role", msg.get("type", "unknown"))
                if role == "human":
                    role = "user"
                elif role == "ai":
                    role = "assistant"
                content = msg.get("content", "")
                msg_id = msg.get("id", f"{thread_id}-{i}")

                # Extract tool calls from dict
                if "tool_calls" in msg and msg["tool_calls"]:
                    tool_calls = msg["tool_calls"]
                if "tool_call_id" in msg:
                    tool_call_id = msg["tool_call_id"]
                if "name" in msg:
                    tool_name = msg["name"]

            else:
                logger.warning(f"Unknown message format: {type(msg)}")
                continue

            messages.append(ThreadMessage(
                id=msg_id,
                role=role,
                content=content,
                tool_calls=tool_calls,
                tool_call_id=tool_call_id,
                name=tool_name,
            ))

        return ThreadMessagesResponse(
            thread_id=thread_id,
            messages=messages,
        )

    except Exception as e:
        logger.error(f"Error getting thread messages: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get thread messages: {e}")


@router.patch("/{thread_id}", response_model=ThreadResponse)
async def update_thread(
    thread_id: str,
    request: UpdateThreadRequest,
    user_id: str = Query(..., description="User ID"),
    tenant_id: str = Query("default", description="Tenant ID"),
):
    """Update thread metadata (title, starred status)."""
    store = await get_store()
    namespace = get_thread_metadata_namespace(tenant_id, user_id)

    result = await store.aget(namespace, thread_id)

    if not result or not result.value:
        raise HTTPException(status_code=404, detail=f"Thread not found: {thread_id}")

    metadata = ThreadMetadata(**result.value)

    # Apply updates
    now = datetime.now(timezone.utc).isoformat()

    if request.title is not None:
        metadata.title = request.title
    if request.is_starred is not None:
        metadata.is_starred = request.is_starred
    if request.is_archived is not None:
        metadata.is_archived = request.is_archived
        # Set archived_at timestamp when archiving, clear it when unarchiving
        if request.is_archived:
            metadata.archived_at = now
        else:
            metadata.archived_at = None

    metadata.updated_at = now

    # Save back
    await store.aput(namespace, thread_id, metadata.model_dump())

    return ThreadResponse(
        id=metadata.id,
        title=metadata.title,
        created_at=metadata.created_at,
        updated_at=metadata.updated_at,
        last_message_at=metadata.last_message_at,
        message_count=metadata.message_count,
        is_starred=metadata.is_starred,
        is_archived=metadata.is_archived,
    )


@router.delete("/{thread_id}")
async def delete_thread(
    thread_id: str,
    user_id: str = Query(..., description="User ID"),
    tenant_id: str = Query("default", description="Tenant ID"),
):
    """
    Delete a thread's metadata.

    Note: This only deletes the metadata. The actual checkpoint data
    remains in the checkpointer (for potential recovery).
    """
    store = await get_store()
    namespace = get_thread_metadata_namespace(tenant_id, user_id)

    result = await store.aget(namespace, thread_id)

    if not result or not result.value:
        raise HTTPException(status_code=404, detail=f"Thread not found: {thread_id}")

    # Delete metadata
    await store.adelete(namespace, thread_id)

    return {"success": True, "message": f"Thread {thread_id} deleted"}
