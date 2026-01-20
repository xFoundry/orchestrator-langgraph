"""Synchronous chat endpoint for the orchestrator."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from langgraph.errors import GraphRecursionError

from app.models.chat import ChatRequest, ChatResponse, Citation, AgentActivity
from app.persistence.redis import get_thread_config

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest, req: Request) -> ChatResponse:
    """
    Send a message to the orchestrator and get a response.

    This is the synchronous endpoint that waits for the full response.
    For real-time streaming, use POST /chat/stream instead.
    """
    try:
        # Import here to avoid circular imports
        from app.graphs.orchestrator import get_orchestrator

        checkpointer = req.app.state.checkpointer

        # Get user context
        user_context = None
        auth0_id = None
        if request.user_context:
            user_context = request.user_context.model_dump(exclude_none=True)
            auth0_id = request.user_context.auth0_id

        # Get or create orchestrator
        graph = await get_orchestrator(
            checkpointer=checkpointer,
            auth0_id=auth0_id,
        )

        # Build thread config
        thread_id = request.thread_id or request.session_id or str(uuid.uuid4())
        config = get_thread_config(
            thread_id=thread_id,
            user_id=request.user_id or "anonymous",
            tenant_id=request.tenant_id,
        )

        # Build message with memory instruction if needed
        message = request.message
        if request.use_memory and auth0_id:
            message = (
                "[MEMORY ENABLED: Before responding, you MUST call recall() to retrieve "
                "any relevant memories or facts stored about this user.]\n\n"
                f"{message}"
            )

        # Invoke the orchestrator
        result = await graph.ainvoke(
            {"messages": [{"role": "user", "content": message}]},
            config=config,
        )

        # Extract response from result
        messages = result.get("messages", [])
        response_text = ""
        if messages:
            last_message = messages[-1]
            if hasattr(last_message, "content"):
                response_text = last_message.content
            elif isinstance(last_message, dict):
                response_text = last_message.get("content", "")

        return ChatResponse(
            message=response_text or "No response generated",
            citations=[],  # Would need event mapper for citations
            agent_trace=[],
            thread_id=thread_id,
            status="success",
        )

    except GraphRecursionError as e:
        logger.warning(f"Graph recursion limit reached: {e}")
        raise HTTPException(
            status_code=500,
            detail=(
                "The request required too many processing steps. "
                "This can happen with complex queries that trigger many sub-tasks. "
                "Try simplifying your request or breaking it into smaller questions."
            ),
        )

    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def chat_health() -> dict[str, Any]:
    """Health check for the chat endpoint."""
    return {
        "status": "healthy",
        "service": "orchestrator-chat",
    }
