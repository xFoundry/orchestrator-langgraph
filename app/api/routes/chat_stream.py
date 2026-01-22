"""Streaming chat endpoint with SSE for real-time agent activity."""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import AsyncGenerator, Any, Union

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from langgraph.errors import GraphRecursionError

from app.models.chat import (
    ChatRequest,
    ContentBlock,
    TextContentBlock,
    ImageContentBlock,
    FileContentBlock,
)
from app.persistence.redis import get_thread_config
from app.streaming import LangGraphEventMapper, SSEEvent, SSEEventType, ErrorData
from app.streaming.deep_agent_event_mapper import DeepAgentEventMapper
from app.api.routes.threads import update_thread_activity

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


def build_message_content(request: ChatRequest) -> Union[str, list[dict[str, Any]]]:
    """
    Build message content from ChatRequest, supporting both legacy text and multimodal.
    
    Returns:
        - str: If only text content (legacy or single text block)
        - list[dict]: If multimodal content (images, files, mixed)
    
    Output format follows LangChain standard content blocks:
    - Text: {"type": "text", "text": "..."}
    - Image (base64): {"type": "image", "base64": "...", "mime_type": "image/jpeg"}
    - Image (URL): {"type": "image", "url": "..."}
    - File (base64): {"type": "file", "base64": "...", "mime_type": "application/pdf"}
    - File (URL): {"type": "file", "url": "..."}
    """
    # If multimodal content is provided, use it (takes precedence)
    if request.content:
        content_blocks: list[dict[str, Any]] = []
        
        for block in request.content:
            if isinstance(block, dict):
                # Already a dict, pass through (handle raw dicts from frontend)
                content_blocks.append(block)
            elif isinstance(block, TextContentBlock):
                content_blocks.append({"type": "text", "text": block.text})
            elif isinstance(block, ImageContentBlock):
                img_block: dict[str, Any] = {"type": "image"}
                if block.base64:
                    img_block["base64"] = block.base64
                    if block.mime_type:
                        img_block["mime_type"] = block.mime_type
                elif block.url:
                    img_block["url"] = block.url
                if block.metadata:
                    img_block["metadata"] = block.metadata
                content_blocks.append(img_block)
            elif isinstance(block, FileContentBlock):
                file_block: dict[str, Any] = {"type": "file"}
                if block.base64:
                    file_block["base64"] = block.base64
                    if block.mime_type:
                        file_block["mime_type"] = block.mime_type
                elif block.url:
                    file_block["url"] = block.url
                elif block.file_id:
                    file_block["file_id"] = block.file_id
                if block.metadata:
                    file_block["metadata"] = block.metadata
                content_blocks.append(file_block)
        
        # Optimization: if only a single text block, return as string
        if len(content_blocks) == 1 and content_blocks[0].get("type") == "text":
            return content_blocks[0]["text"]
        
        return content_blocks
    
    # Fallback to legacy message field
    return request.message or ""


def extract_text_from_content(content: Union[str, list[dict[str, Any]]]) -> str:
    """Extract text content for query field (v3 orchestrator compatibility)."""
    if isinstance(content, str):
        return content
    
    text_parts = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            text_parts.append(block.get("text", ""))
    
    return " ".join(text_parts)


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

            # Get model override from request
            model_override = request.model_override
            if model_override:
                logger.info(f"Using model override from request: {model_override}")

            graph = await get_orchestrator_deep_agent(
                checkpointer=checkpointer,
                store=store,
                user_context=user_context_dict,
                auth0_id=auth0_id,
                selected_tools=request.selected_tools,
                model_override=model_override,
            )
            logger.info(f"Using Deep Agent orchestrator (model: {model_override or 'default'})")
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

        # Build message content (supports multimodal: images, PDFs, text)
        message_content = build_message_content(request)

        # Extract text for query field and memory instructions
        text_content = extract_text_from_content(message_content)

        # Track thread metadata for thread history
        await update_thread_activity(
            thread_id=thread_id,
            user_id=request.user_id or "anonymous",
            tenant_id=request.tenant_id,
            message_text=text_content,
        )

        # Add memory instruction if needed (prepend to text content)
        if request.use_memory and auth0_id:
            memory_instruction = (
                "[MEMORY ENABLED: Before responding, you MUST call recall() to retrieve "
                "any relevant memories or facts stored about this user.]\n\n"
            )
            if isinstance(message_content, str):
                message_content = memory_instruction + message_content
            else:
                # For multimodal: prepend instruction as text block
                message_content = [
                    {"type": "text", "text": memory_instruction + text_content}
                ] + [b for b in message_content if b.get("type") != "text"]
                # Update text_content for query field
                text_content = memory_instruction + text_content
        
        # Log multimodal content info
        if isinstance(message_content, list):
            block_types = [b.get("type", "unknown") for b in message_content]
            logger.info(f"Multimodal message with blocks: {block_types}")

        # Build input state based on orchestrator version
        if use_deep_agent:
            # Deep Agent uses simple message format (handles state internally)
            input_state = {"messages": [{"role": "user", "content": message_content}]}
        elif use_v3:
            # V3 uses explicit query field and structured state
            input_state = {
                "messages": [{"role": "user", "content": message_content}],
                "query": text_content,  # Query is text-only for routing/planning
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
            input_state = {"messages": [{"role": "user", "content": message_content}]}

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

    ## Multimodal Support

    This endpoint supports multimodal input (images and PDFs) via content blocks.
    
    **Supported formats:**
    - Images: JPEG, PNG, GIF, WEBP (via base64 or URL)
    - Documents: PDF (via base64 or URL)
    
    **Model compatibility:**
    - GPT-5.2+: Images ✅, PDFs ❌ (needs preprocessing)
    - Claude 4.5+: Images ✅, PDFs ✅
    - Gemini: Images ✅, PDFs ✅

    ## Event Types

    - `agent_activity`: Agent actions (tool calls, delegations, thinking)
    - `text_chunk`: Partial response text as it's generated
    - `citation`: Source citations extracted from responses
    - `tool_result`: Results from tool executions
    - `thinking`: Internal reasoning phases (planning, action, reasoning)
    - `complete`: Final response with full message and metadata
    - `error`: Error information if something goes wrong

    ## Example Usage (Text Only - Legacy)

    ```javascript
    const response = await fetch('/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: 'What feedback did mentors give?' })
    });
    ```

    ## Example Usage (Multimodal)

    ```javascript
    const response = await fetch('/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            content: [
                { type: 'text', text: 'What is in this image?' },
                { type: 'image', data: 'base64...', mimeType: 'image/jpeg' }
            ]
        })
    });
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
        "features": [
            "sse",
            "agent_activity", 
            "text_chunks", 
            "citations", 
            "thinking",
            "multimodal",
        ],
        "multimodal": {
            "supported_types": ["image/jpeg", "image/png", "image/gif", "image/webp", "application/pdf"],
            "input_formats": ["base64", "url"],
        },
    }
