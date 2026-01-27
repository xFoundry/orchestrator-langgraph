"""Streaming chat endpoint with SSE for real-time agent activity."""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import AsyncGenerator

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from langgraph.errors import GraphRecursionError

from app.models.chat import ChatRequest
from app.persistence.redis import get_thread_config
from app.streaming import LangGraphEventMapper, SSEEvent, SSEEventType, ErrorData
from app.streaming.deep_agent_event_mapper import DeepAgentEventMapper

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


async def sse_event_generator(
    request: ChatRequest,
    req: Request,
) -> AsyncGenerator[str, None]:
    """
    Generate SSE events from orchestrator execution.

    Yields formatted SSE strings as the orchestrator processes the request.
    Each event follows the SSE format:
        event: <event_type>
        data: <json_payload>

    Args:
        request: Chat request with message and configuration
        req: FastAPI request for accessing app state

    Yields:
        SSE-formatted strings for streaming response
    """
    thread_id = request.thread_id or request.session_id or str(uuid.uuid4())

    # Select mapper based on orchestrator type
    use_deep_agent = getattr(request, "use_deep_agent", False)
    if use_deep_agent:
        mapper = DeepAgentEventMapper()
    else:
        mapper = LangGraphEventMapper()

    try:
        checkpointer = req.app.state.checkpointer
        store = getattr(req.app.state, "store", None)

        # Get user context
        auth0_id = None
        user_context_dict = None
        if request.user_context:
            auth0_id = request.user_context.auth0_id
            user_context_dict = {
                "name": request.user_context.name,
                "email": request.user_context.email,
                "role": request.user_context.role,
                "teams": request.user_context.teams,
                "cohort": request.user_context.cohort,
            }

        # Orchestrator selection priority: deep_agent > v3 > v2 > v1
        use_v3 = getattr(request, "use_v3", True)
        use_v2 = getattr(request, "use_v2", False)

        if use_deep_agent:
            # Deep Agent orchestrator (planning, subagents, context management)
            from app.graphs.orchestrator_deep_agent import get_orchestrator_deep_agent
            graph = await get_orchestrator_deep_agent(
                checkpointer=checkpointer,
                store=store,
                user_context=user_context_dict,
                auth0_id=auth0_id,
                selected_tools=request.selected_tools,
            )
            logger.info("Using Deep Agent orchestrator")
        elif use_v3:
            from app.graphs.orchestrator_v3 import get_orchestrator_v3
            graph = await get_orchestrator_v3(
                checkpointer=checkpointer,
                user_context=user_context_dict,
                auth0_id=auth0_id,
            )
            logger.info("Using orchestrator v3 (supervisor pattern)")
        elif use_v2:
            from app.graphs.orchestrator_v2 import get_orchestrator_v2
            graph = await get_orchestrator_v2(
                checkpointer=checkpointer,
                user_context=user_context_dict,
                auth0_id=auth0_id,
            )
            logger.info("Using orchestrator v2")
        else:
            from app.graphs.orchestrator import get_orchestrator
            graph = await get_orchestrator(
                checkpointer=checkpointer,
                user_context=user_context_dict,
                auth0_id=auth0_id,
            )
            logger.info("Using orchestrator v1")

        # Build thread config
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

        # Build input state based on orchestrator version
        if use_deep_agent:
            # Deep Agent uses simple message format (handles state internally)
            input_state = {"messages": [{"role": "user", "content": message}]}
        elif use_v3:
            # V3 uses explicit query field and structured state
            input_state = {
                "messages": [{"role": "user", "content": message}],
                "query": message,
                "user_context": user_context_dict,
                "canvas_id": request.canvas_id,
                "chat_block_id": request.chat_block_id,
                "auto_artifacts": request.auto_artifacts,
                "context_artifacts": request.context_artifacts or [],
                "selected_tools": request.selected_tools or [],
                "plan": None,
                "planned_calls": [],
                "tool_results": [],
                "evaluation": None,
                "retry_count": 0,
                "max_retries": 2,
                "final_answer": None,
                "confidence": 0.0,
                "citations": [],
            }
        else:
            # V1/V2 use simple message format
            input_state = {"messages": [{"role": "user", "content": message}]}

        # Stream events from orchestrator
        async for event in graph.astream_events(
            input_state,
            config=config,
            version="v2",
        ):
            # Map LangGraph event to SSE events
            sse_events = mapper.map_event(event)

            for sse_event in sse_events:
                yield sse_event.to_sse_string()

            # Allow other tasks to run
            await asyncio.sleep(0)

        # Emit final complete event
        complete_event = mapper.create_complete_event(thread_id)
        yield complete_event.to_sse_string()

    except asyncio.CancelledError:
        logger.info("SSE stream cancelled by client disconnect")
        raise

    except GraphRecursionError as e:
        logger.warning(f"Graph recursion limit reached: {e}")
        error_msg = (
            "The request required too many processing steps. "
            "This can happen with complex queries that trigger many sub-tasks. "
            "Try simplifying your request or breaking it into smaller questions."
        )
        error_event = mapper.create_error_event(error_msg)
        yield error_event.to_sse_string()

    except Exception as e:
        logger.error(f"SSE streaming error: {e}", exc_info=True)
        error_event = mapper.create_error_event(str(e))
        yield error_event.to_sse_string()


@router.post("/stream")
async def chat_stream(request: ChatRequest, req: Request) -> StreamingResponse:
    """
    Stream agent activity in real-time via Server-Sent Events.

    This endpoint provides real-time feedback as the orchestrator processes
    your request. Events are streamed as they occur, including agent activity,
    tool calls, partial response text, and citations.

    ## Event Types

    - `agent_activity`: Agent actions (tool calls, delegations, thinking)
    - `text_chunk`: Partial response text as it's generated
    - `citation`: Source citations extracted from responses
    - `tool_result`: Results from tool executions
    - `thinking`: Internal reasoning phases (planning, action, reasoning)
    - `complete`: Final response with full message and metadata
    - `error`: Error information if something goes wrong

    ## Example Usage

    ```javascript
    const response = await fetch('/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: 'What feedback did mentors give?' })
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const text = decoder.decode(value);
        // Parse SSE events from text
        // Each event has format: "event: <type>\\ndata: <json>\\n\\n"
    }
    ```
    """
    return StreamingResponse(
        sse_event_generator(request, req),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


@router.get("/stream/health")
async def stream_health() -> dict:
    """Health check for the streaming endpoint."""
    return {
        "status": "healthy",
        "service": "orchestrator-chat-stream",
        "features": ["sse", "agent_activity", "text_chunks", "citations", "thinking"],
    }
