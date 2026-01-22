"""Middleware that emits LangGraph UI events for deep agent tool calls."""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, Awaitable, Callable, Optional

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.callbacks.manager import adispatch_custom_event
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.config import get_stream_writer
from langgraph.runtime import Runtime

_logger = logging.getLogger(__name__)


def _emit_custom_event(data: dict[str, Any]) -> bool:
    """Emit a custom UI event using the stream writer from context.
    
    Returns True if emission succeeded, False otherwise.
    """
    try:
        writer = get_stream_writer()
        if writer:
            writer(data)
            return True
    except Exception as e:
        _logger.debug(f"[UI Middleware] get_stream_writer failed: {e}")
    
    # Fallback: try adispatch_custom_event (won't work in all contexts)
    try:
        # This is synchronous fallback - won't work in async context
        # but we try anyway for compatibility
        _logger.debug("[UI Middleware] Falling back to adispatch_custom_event")
    except Exception as e:
        _logger.debug(f"[UI Middleware] adispatch_custom_event fallback also failed: {e}")
    
    return False


async def _async_emit_custom_event(data: dict[str, Any]) -> bool:
    """Async version of emit_custom_event that tries multiple methods.
    
    Returns True if emission succeeded, False otherwise.
    """
    # First try get_stream_writer (preferred method)
    try:
        writer = get_stream_writer()
        if writer:
            writer(data)
            _logger.debug(f"[UI Middleware] Emitted via stream_writer: {data.get('name', 'unknown')}")
            return True
    except Exception as e:
        _logger.debug(f"[UI Middleware] get_stream_writer failed: {e}")
    
    # Fallback to adispatch_custom_event
    try:
        await adispatch_custom_event("ui", data)
        _logger.debug(f"[UI Middleware] Emitted via adispatch_custom_event: {data.get('name', 'unknown')}")
        return True
    except Exception as e:
        _logger.debug(f"[UI Middleware] adispatch_custom_event failed: {e}")
    
    return False


# Type alias for tool call request (may vary between versions)
try:
    from langchain.agents.middleware.types import ToolCallRequest
except ImportError:
    from langchain.tools.tool_node import ToolCallRequest


def _summarize_result(result: ToolMessage | Any) -> str:
    if isinstance(result, ToolMessage):
        content = result.content
        if isinstance(content, str):
            return content[:400]
        return str(content)[:400]
    return str(result)[:400]


def _extract_content(tool_args: dict[str, Any]) -> str:
    for key in ("content", "new_content", "new_string"):
        value = tool_args.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _artifact_id(path: str) -> str:
    return f"artifact_{path.replace('/', '_')}"


async def _emit_artifact_stream(
    *,
    path: str,
    filename: str,
    content: str,
    action: str,
    saved: bool,
) -> None:
    """Emit progressive UI updates for artifact content."""
    artifact_id = _artifact_id(path)
    chunk_size = 1000
    delay_seconds = 0.02

    if not content:
        await _async_emit_custom_event({
            "type": "ui",
            "id": artifact_id,
            "name": "artifact",
            "props": {
                "id": artifact_id,
                "title": filename,
                "path": path,
                "content": "",
                "action": action,
                "saved": saved,
                "streaming": False,
            },
            "metadata": {"merge": True},
        })
        return

    for idx in range(chunk_size, len(content) + chunk_size, chunk_size):
        is_final = idx >= len(content)
        await _async_emit_custom_event({
            "type": "ui",
            "id": artifact_id,
            "name": "artifact",
            "props": {
                "id": artifact_id,
                "title": filename,
                "path": path,
                "content": content[:idx],
                "action": action,
                "saved": saved,
                "streaming": not is_final,
            },
            "metadata": {"merge": True},
        })
        if not is_final:
            await asyncio.sleep(delay_seconds)


class DeepAgentUIMiddleware(AgentMiddleware):
    """Emit UI messages for tool calls/results for Agent Chat UI."""

    def __init__(self):
        super().__init__()
        self._thinking_emitted: set[str] = set()
        _logger.info("[DeepAgentUIMiddleware] Initialized")

    def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """Preprocess messages to normalize multimodal content blocks.
        
        The TypeScript SDK (agent-ui) sends content blocks with:
        - `data` field for base64 content
        - `mimeType` field (camelCase)
        
        The Python LangChain SDK expects:
        - `base64` field for base64 content
        - `mime_type` field (snake_case)
        
        This hook normalizes the format before messages are sent to the model.
        """
        messages = state.get("messages", [])
        if not messages:
            return None
        
        # Debug: Log incoming message structure
        for i, msg in enumerate(messages):
            msg_type = type(msg).__name__
            content = getattr(msg, "content", None)
            if isinstance(content, list):
                block_types = []
                for block in content:
                    if isinstance(block, dict):
                        block_types.append(f"{block.get('type', 'unknown')}({list(block.keys())})")
                    else:
                        block_types.append(type(block).__name__)
                _logger.debug(f"[UI Middleware] Message[{i}] ({msg_type}): content blocks = {block_types}")
            elif isinstance(content, str):
                _logger.debug(f"[UI Middleware] Message[{i}] ({msg_type}): text content ({len(content)} chars)")
        
        modified = False
        for msg in messages:
            content = getattr(msg, "content", None)
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        block_modified = self._normalize_content_block(block)
                        if block_modified:
                            modified = True
        
        if modified:
            _logger.info("[UI Middleware] Normalized multimodal content blocks")
            return {"messages": messages}
        
        return None
    
    def _normalize_content_block(self, block: dict[str, Any]) -> bool:
        """Normalize a content block from TypeScript to Python format.
        
        Returns True if the block was modified.
        """
        modified = False
        block_type = block.get("type")
        
        # Normalize image blocks
        if block_type == "image":
            # Convert 'data' to 'base64'
            if "data" in block and "base64" not in block:
                block["base64"] = block.pop("data")
                modified = True
                _logger.debug("[UI Middleware] Normalized image: data -> base64")
            
            # Convert 'mimeType' to 'mime_type'
            if "mimeType" in block and "mime_type" not in block:
                block["mime_type"] = block.pop("mimeType")
                modified = True
                _logger.debug("[UI Middleware] Normalized image: mimeType -> mime_type")
        
        # Normalize file blocks (PDFs, etc.)
        elif block_type == "file":
            # Convert 'data' to 'base64'
            if "data" in block and "base64" not in block:
                block["base64"] = block.pop("data")
                modified = True
                _logger.debug("[UI Middleware] Normalized file: data -> base64")
            
            # Convert 'mimeType' to 'mime_type'
            if "mimeType" in block and "mime_type" not in block:
                block["mime_type"] = block.pop("mimeType")
                modified = True
                _logger.debug("[UI Middleware] Normalized file: mimeType -> mime_type")
        
        return modified

    async def awrap_model_call(
        self,
        request: Any,  # ModelRequest
        handler: Callable[[Any], Awaitable[Any]],
    ) -> Any:
        """Wrap model call to extract thinking from response (async version).
        
        DeepAgents uses awrap_model_call for async contexts like astream/ainvoke.
        
        This method also:
        1. Normalizes multimodal content blocks (TypeScript -> Python format)
        2. Extracts thinking/reasoning from responses
        """
        _logger.info("[UI Middleware] awrap_model_call - calling model")
        
        # =====================================================================
        # PRE-PROCESSING: Normalize multimodal content blocks in the request
        # 
        # TypeScript SDK (agent-ui) uses:
        #   { type: "image", data: "base64...", mimeType: "image/jpeg" }
        # 
        # OpenAI/OpenRouter expects:
        #   { type: "image_url", image_url: { url: "data:image/jpeg;base64,..." } }
        # 
        # LangChain standard format:
        #   { type: "image", base64: "...", mime_type: "image/jpeg" }
        # =====================================================================
        if hasattr(request, 'messages'):
            multimodal_found = False
            for msg in request.messages:
                content = getattr(msg, 'content', None)
                if isinstance(content, list):
                    new_content = []
                    for block in content:
                        if isinstance(block, dict):
                            block_type = block.get('type')
                            # Log what we found
                            _logger.debug(f"[UI Middleware] Found {block_type} block with keys: {list(block.keys())}")
                            
                            # Handle image blocks - convert to OpenAI format
                            if block_type == 'image':
                                multimodal_found = True
                                # Get base64 data (could be in 'data' or 'base64')
                                base64_data = block.get('data') or block.get('base64')
                                # Get mime type (could be 'mimeType' or 'mime_type')
                                mime_type = block.get('mimeType') or block.get('mime_type') or 'image/jpeg'
                                
                                if base64_data:
                                    # Convert to OpenAI image_url format
                                    new_block = {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:{mime_type};base64,{base64_data}"
                                        }
                                    }
                                    new_content.append(new_block)
                                    _logger.info(f"[UI Middleware] Converted image to OpenAI format (mime: {mime_type})")
                                elif block.get('url'):
                                    # URL-based image
                                    new_block = {
                                        "type": "image_url",
                                        "image_url": {"url": block['url']}
                                    }
                                    new_content.append(new_block)
                                    _logger.info(f"[UI Middleware] Converted image URL to OpenAI format")
                                else:
                                    new_content.append(block)
                            
                            # Handle file blocks (PDFs) - convert to OpenRouter file format
                            elif block_type == 'file':
                                multimodal_found = True
                                base64_data = block.get('data') or block.get('base64')
                                mime_type = block.get('mimeType') or block.get('mime_type') or 'application/pdf'
                                # Get filename from metadata or use default
                                metadata = block.get('metadata', {})
                                filename = metadata.get('filename') or metadata.get('name') or 'document.pdf'
                                
                                if base64_data:
                                    # OpenRouter expects: { type: "file", file: { filename, file_data } }
                                    # See: https://openrouter.ai/docs/guides/overview/multimodal/pdfs
                                    new_block = {
                                        "type": "file",
                                        "file": {
                                            "filename": filename,
                                            "file_data": f"data:{mime_type};base64,{base64_data}"
                                        }
                                    }
                                    new_content.append(new_block)
                                    _logger.info(f"[UI Middleware] Converted file to OpenRouter format (filename: {filename}, mime: {mime_type})")
                                else:
                                    new_content.append(block)
                            
                            # Keep text blocks as-is
                            elif block_type == 'text':
                                new_content.append(block)
                            else:
                                new_content.append(block)
                        else:
                            new_content.append(block)
                    
                    # Update the message content
                    if multimodal_found and hasattr(msg, 'content'):
                        msg.content = new_content
                        _logger.info(f"[UI Middleware] Updated message with {len(new_content)} content blocks")
            
            if multimodal_found:
                _logger.info("[UI Middleware] Multimodal content detected and converted to OpenAI format")
        
        # Call the actual model
        response = await handler(request)
        _logger.info(f"[UI Middleware] awrap_model_call - got response: {type(response)}")
        
        # =====================================================================
        # POST-PROCESSING: Extract thinking from response
        # =====================================================================
        if hasattr(response, 'message'):
            msg = response.message
            thinking = self._extract_thinking_from_message(msg)
            if thinking:
                _logger.info(f"[UI Middleware] Found thinking ({len(thinking)} chars)")
                # Emit via runtime if available
                if hasattr(request, 'runtime') and hasattr(request.runtime, 'stream_writer'):
                    self._emit_thinking_via_runtime(request.runtime, thinking)
        
        return response

    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """Extract and emit thinking content from the model response.
        
        This hook runs after each model call to extract any thinking/reasoning
        content and emit it as a custom UI event via runtime.stream_writer().
        
        NOTE: This must be synchronous - async def returns a coroutine which LangGraph rejects.
        """
        messages = state.get("messages", [])
        if not messages:
            _logger.debug("[UI Middleware] No messages in state")
            return None

        # Get the last message (most recent AI response)
        last_msg = messages[-1]
        _logger.debug(f"[UI Middleware] Last message type: {type(last_msg).__name__}")
        
        if not isinstance(last_msg, AIMessage):
            return None

        # Debug: Log what we're looking for
        _logger.debug(f"[UI Middleware] additional_kwargs: {getattr(last_msg, 'additional_kwargs', {})}")
        _logger.debug(f"[UI Middleware] response_metadata: {getattr(last_msg, 'response_metadata', {})}")
        
        # Debug: Log ALL message attributes to find where reasoning is
        try:
            all_attrs = [attr for attr in dir(last_msg) if not attr.startswith('_')]
            interesting_attrs = {}
            for attr in all_attrs:
                try:
                    val = getattr(last_msg, attr)
                    if not callable(val) and val is not None:
                        interesting_attrs[attr] = type(val).__name__
                        # Log potentially interesting values
                        if attr in ['content', 'reasoning', 'thinking', 'output', 'content_blocks', 'tool_calls']:
                            _logger.debug(f"[UI Middleware] msg.{attr} = {val}")
                except Exception:
                    pass
            _logger.debug(f"[UI Middleware] All non-None attributes: {interesting_attrs}")
        except Exception as e:
            _logger.debug(f"[UI Middleware] Error logging attributes: {e}")

        thinking_content = self._extract_thinking_from_message(last_msg)
        if thinking_content:
            _logger.info(f"[UI Middleware] Found thinking content ({len(thinking_content)} chars)")
            self._emit_thinking_via_runtime(runtime, thinking_content)
        else:
            _logger.debug("[UI Middleware] No thinking content found in AI message")

        return None
    
    def _emit_thinking_via_runtime(self, runtime: Runtime, content: str) -> None:
        """Emit thinking content as a custom UI event via runtime.stream_writer()."""
        # Deduplicate by content hash
        content_hash = str(hash(content))
        if content_hash in self._thinking_emitted:
            return
        self._thinking_emitted.add(content_hash)
        
        # Emit as custom event using runtime.stream_writer
        event_id = f"ui_thinking_{uuid.uuid4().hex[:8]}"
        runtime.stream_writer({
            "type": "ui",
            "id": event_id,
            "name": "thinking",
            "props": {
                "phase": "reasoning",
                "content": content,
            },
            "metadata": {"merge": True},
        })

    def _extract_thinking_from_message(self, message: AIMessage) -> str | None:
        """Extract thinking content from an AI message.
        
        Supports:
        - GPT-5.x / o-series: Reasoning in content_blocks with type="reasoning"
        - Claude: Extended thinking in message.thinking or additional_kwargs
        """
        # Debug: Log message structure
        _logger.debug(f"[UI Middleware] Message content type: {type(message.content)}")
        
        # =============================================================
        # GPT-5.x / o-series: Check content_blocks for reasoning
        # When using reasoning={effort:..., summary:...}, OpenAI returns
        # content_blocks with type="reasoning" containing summary text
        # =============================================================
        content_blocks = getattr(message, "content_blocks", None)
        if content_blocks:
            _logger.debug(f"[UI Middleware] Found content_blocks: {len(content_blocks)} blocks")
            # Log all block types for debugging
            for i, block in enumerate(content_blocks):
                if isinstance(block, dict):
                    _logger.debug(f"[UI Middleware] content_blocks[{i}] type: {block.get('type')}, keys: {list(block.keys())}")
                else:
                    _logger.debug(f"[UI Middleware] content_blocks[{i}] is {type(block).__name__}: {str(block)[:100]}")
            reasoning_texts = []
            for block in content_blocks:
                if isinstance(block, dict):
                    block_type = block.get("type")
                    # Type "reasoning" with nested summary/reasoning
                    if block_type == "reasoning":
                        summary = block.get("summary", [])
                        if summary:
                            for item in summary:
                                if isinstance(item, dict) and item.get("type") == "summary_text":
                                    reasoning_texts.append(str(item.get("text", "")))
                                elif isinstance(item, str):
                                    reasoning_texts.append(item)
                        reasoning_text = block.get("reasoning")
                        if reasoning_text:
                            reasoning_texts.append(str(reasoning_text))
                    # Direct "summary_text" type block
                    elif block_type == "summary_text":
                        if "text" in block:
                            reasoning_texts.append(str(block["text"]))
                    # "thinking" type block
                    elif block_type == "thinking":
                        if "thinking" in block:
                            reasoning_texts.append(str(block["thinking"]))
                        elif "text" in block:
                            reasoning_texts.append(str(block["text"]))
            if reasoning_texts:
                result = "\n\n".join(reasoning_texts)
                _logger.info(f"[UI Middleware] Extracted reasoning from content_blocks: {len(result)} chars")
                return result
        
        # =============================================================
        # Also check message.content if it's a list of content blocks
        # =============================================================
        content = message.content
        if isinstance(content, list):
            _logger.debug(f"[UI Middleware] Checking message.content list: {len(content)} items")
            # Log all block types for debugging
            for i, block in enumerate(content):
                if isinstance(block, dict):
                    _logger.debug(f"[UI Middleware] content[{i}] type: {block.get('type')}, keys: {list(block.keys())}")
                else:
                    _logger.debug(f"[UI Middleware] content[{i}] is {type(block).__name__}: {str(block)[:100]}")
            reasoning_texts = []
            for block in content:
                if isinstance(block, dict):
                    block_type = block.get("type")
                    # Type "reasoning" with nested summary/reasoning
                    if block_type == "reasoning":
                        summary = block.get("summary", [])
                        if summary:
                            for item in summary:
                                if isinstance(item, dict) and item.get("type") == "summary_text":
                                    reasoning_texts.append(str(item.get("text", "")))
                                elif isinstance(item, str):
                                    reasoning_texts.append(item)
                        reasoning_text = block.get("reasoning")
                        if reasoning_text:
                            reasoning_texts.append(str(reasoning_text))
                    # Direct "summary_text" type block
                    elif block_type == "summary_text":
                        if "text" in block:
                            reasoning_texts.append(str(block["text"]))
                    # "thinking" type block
                    elif block_type == "thinking":
                        if "thinking" in block:
                            reasoning_texts.append(str(block["thinking"]))
                        elif "text" in block:
                            reasoning_texts.append(str(block["text"]))
            if reasoning_texts:
                result = "\n\n".join(reasoning_texts)
                _logger.info(f"[UI Middleware] Extracted reasoning from content list: {len(result)} chars")
                return result
        
        # =============================================================
        # Check for direct reasoning_content attribute (older API)
        # =============================================================
        reasoning_content = getattr(message, "reasoning_content", None)
        if reasoning_content:
            _logger.info(f"[UI Middleware] Found reasoning_content attribute: {len(str(reasoning_content))} chars")
            return str(reasoning_content)
        
        # =============================================================
        # Check for reasoning_details (OpenRouter format for Anthropic/OpenAI reasoning)
        # OpenRouter returns reasoning in message.reasoning_details array
        # =============================================================
        reasoning_details = getattr(message, "reasoning_details", None)
        if reasoning_details:
            _logger.debug(f"[UI Middleware] Found reasoning_details: {len(reasoning_details)} items")
            reasoning_texts = []
            for detail in reasoning_details:
                if isinstance(detail, dict):
                    detail_type = detail.get("type", "")
                    # reasoning.text - contains raw reasoning text
                    if detail_type == "reasoning.text":
                        text = detail.get("text", "")
                        if text:
                            reasoning_texts.append(text)
                    # reasoning.summary - contains summary of reasoning
                    elif detail_type == "reasoning.summary":
                        summary = detail.get("summary", "")
                        if summary:
                            reasoning_texts.append(f"**Summary:** {summary}")
                    # reasoning.encrypted - redacted content
                    elif detail_type == "reasoning.encrypted":
                        reasoning_texts.append("[Reasoning content redacted]")
            if reasoning_texts:
                result = "\n\n".join(reasoning_texts)
                _logger.info(f"[UI Middleware] Extracted reasoning from reasoning_details: {len(result)} chars")
                return result
        
        # Also check additional_kwargs for reasoning_details (some LangChain versions put it there)
        additional = getattr(message, "additional_kwargs", {}) or {}
        if "reasoning_details" in additional:
            reasoning_details = additional["reasoning_details"]
            _logger.debug(f"[UI Middleware] Found reasoning_details in additional_kwargs")
            reasoning_texts = []
            for detail in reasoning_details:
                if isinstance(detail, dict):
                    detail_type = detail.get("type", "")
                    if detail_type == "reasoning.text":
                        text = detail.get("text", "")
                        if text:
                            reasoning_texts.append(text)
                    elif detail_type == "reasoning.summary":
                        summary = detail.get("summary", "")
                        if summary:
                            reasoning_texts.append(f"**Summary:** {summary}")
            if reasoning_texts:
                result = "\n\n".join(reasoning_texts)
                _logger.info(f"[UI Middleware] Extracted reasoning from additional_kwargs.reasoning_details: {len(result)} chars")
                return result
        
        # =============================================================
        # Check response_metadata for reasoning (OpenAI Responses API)
        # =============================================================
        response_metadata = getattr(message, "response_metadata", {}) or {}
        if "reasoning" in response_metadata:
            reasoning_data = response_metadata["reasoning"]
            _logger.debug(f"[UI Middleware] Found reasoning in response_metadata: {type(reasoning_data)}")
            if isinstance(reasoning_data, str):
                return reasoning_data
            if isinstance(reasoning_data, dict):
                # Check for summary
                summary = reasoning_data.get("summary", [])
                if summary:
                    texts = []
                    for item in summary:
                        if isinstance(item, dict) and item.get("type") == "summary_text":
                            texts.append(str(item.get("text", "")))
                        elif isinstance(item, str):
                            texts.append(item)
                    if texts:
                        return "\n\n".join(texts)
        
        # Also check for output in response_metadata (OpenAI Responses API format)
        if "output" in response_metadata:
            output = response_metadata["output"]
            _logger.debug(f"[UI Middleware] Found output in response_metadata: {type(output)}")
            if isinstance(output, list):
                for item in output:
                    if isinstance(item, dict):
                        item_type = item.get("type")
                        if item_type == "reasoning":
                            summary = item.get("summary", [])
                            if summary:
                                texts = []
                                for s in summary:
                                    if isinstance(s, dict) and s.get("type") == "summary_text":
                                        texts.append(str(s.get("text", "")))
                                if texts:
                                    return "\n\n".join(texts)
        
        # =============================================================
        # Claude extended thinking (message.thinking or additional_kwargs)
        # =============================================================
        thinking = getattr(message, "thinking", None)
        if thinking:
            if isinstance(thinking, str):
                return thinking
            if isinstance(thinking, dict) and "content" in thinking:
                return str(thinking["content"])
            if isinstance(thinking, list):
                texts = []
                for block in thinking:
                    if isinstance(block, str):
                        texts.append(block)
                    elif isinstance(block, dict):
                        if block.get("type") == "thinking":
                            texts.append(str(block.get("thinking", "")))
                        elif "text" in block:
                            texts.append(str(block["text"]))
                if texts:
                    return "\n".join(texts)

        # Check additional_kwargs for thinking
        additional = getattr(message, "additional_kwargs", {}) or {}

        if "thinking" in additional:
            thinking_data = additional["thinking"]
            if isinstance(thinking_data, str):
                return thinking_data
            if isinstance(thinking_data, dict) and "content" in thinking_data:
                return str(thinking_data["content"])
            if isinstance(thinking_data, list):
                texts = []
                for block in thinking_data:
                    if isinstance(block, str):
                        texts.append(block)
                    elif isinstance(block, dict):
                        if block.get("type") == "thinking":
                            texts.append(str(block.get("thinking", "")))
                        elif "text" in block:
                            texts.append(str(block["text"]))
                if texts:
                    return "\n".join(texts)

        # GPT-5.2 / o1 / o3 reasoning_content in additional_kwargs
        if "reasoning_content" in additional:
            return str(additional["reasoning_content"])

        # Check response_metadata
        response_meta = getattr(message, "response_metadata", {}) or {}
        if "thinking" in response_meta:
            return str(response_meta["thinking"])
        if "reasoning_content" in response_meta:
            return str(response_meta["reasoning_content"])

        return None

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Any]],
    ) -> ToolMessage | Any:
        tool_call = request.tool_call or {}
        tool_name = tool_call.get("name") or getattr(request.tool, "name", "tool")
        tool_args = tool_call.get("args") or tool_call.get("arguments") or {}

        path = tool_args.get("path") or tool_args.get("file_path") or ""
        if tool_name in {"write_file", "edit_file"} and isinstance(path, str):
            if path.startswith("/artifacts/"):
                filename = path.split("/")[-1] if path else "artifact"
                content = _extract_content(tool_args)
                artifact_id = _artifact_id(path)
                action = "create" if tool_name == "write_file" else "update"
                saved = "/artifacts/saved/" in path

                # Emit IMMEDIATELY to show artifact in list with streaming indicator
                await _async_emit_custom_event({
                    "type": "ui",
                    "id": artifact_id,
                    "name": "artifact",
                    "props": {
                        "id": artifact_id,
                        "title": filename,
                        "path": path,
                        "content": "",  # Start empty
                        "action": action,
                        "saved": saved,
                        "streaming": True,  # Indicates streaming has started
                    },
                    "metadata": {"merge": True},
                })

                # Then stream the content progressively in background
                asyncio.create_task(
                    _emit_artifact_stream(
                        path=path,
                        filename=filename,
                        content=content,
                        action=action,
                        saved=saved,
                    )
                )

        # Emit tool_call UI event (skip for request_clarifications - it uses interrupt)
        if tool_name != "request_clarifications":
            await _async_emit_custom_event({
                "type": "ui",
                "id": f"ui_tool_call_{tool_call.get('id', '')}",
                "name": "tool_call",
                "props": {
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                },
                "metadata": {"merge": True},
            })
        else:
            # Emit a special "asking" UI event for clarifications
            await _async_emit_custom_event({
                "type": "ui",
                "id": f"ui_clarification_{tool_call.get('id', '')}",
                "name": "thinking",
                "props": {
                    "phase": "clarification",
                    "content": f"Asking for clarification: {tool_args.get('title', 'Question')}",
                    "agent": "DeepAgent",
                },
                "metadata": {"merge": True},
            })

        if tool_name == "write_todos":
            # Extract todo descriptions for display
            todos = tool_args.get("todos", [])
            todo_count = len(todos) if isinstance(todos, list) else 0
            
            # Emit a thinking event showing what the agent is planning
            await _async_emit_custom_event({
                "type": "ui",
                "id": f"ui_planning_{tool_call.get('id', '')}",
                "name": "thinking",
                "props": {
                    "phase": "planning",
                    "content": f"Planning {todo_count} tasks..." if todo_count else "Updating task list...",
                    "agent": "DeepAgent",
                },
                "metadata": {"merge": True},
            })
            
            # Also emit the plan event for the trace panel
            await _async_emit_custom_event({
                "type": "ui",
                "id": f"ui_plan_{tool_call.get('id', '')}",
                "name": "plan",
                "props": {
                    "tool_calls": [{"tool_name": tool_name, "tool_args": tool_args}],
                },
                "metadata": {"merge": True},
            })

        if tool_name == "task":
            # Extract subagent name and task description for display
            subagent_name = tool_args.get('agent') or tool_args.get('name') or 'subagent'
            task_desc = tool_args.get('task') or tool_args.get('prompt') or ''
            # Truncate task description for display
            task_preview = (task_desc[:100] + "...") if len(task_desc) > 100 else task_desc
            
            await _async_emit_custom_event({
                "type": "ui",
                "id": f"ui_delegate_{tool_call.get('id', '')}",
                "name": "thinking",
                "props": {
                    "phase": "delegate",
                    "content": f"Delegating to {subagent_name}: {task_preview}" if task_preview else f"Delegating to {subagent_name}",
                    "agent": "DeepAgent",
                },
                "metadata": {"merge": True},
            })

        result = await handler(request)

        success = True
        if isinstance(result, ToolMessage) and getattr(result, "status", None) == "error":
            success = False

        await _async_emit_custom_event({
            "type": "ui",
            "id": f"ui_tool_result_{tool_call.get('id', '')}",
            "name": "tool_result",
            "props": {
                "tool_name": tool_name,
                "success": success,
                "summary": _summarize_result(result),
            },
            "metadata": {"merge": True},
        })

        return result
