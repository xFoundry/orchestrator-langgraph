"""Maps LangGraph streaming events to SSE events."""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Optional

from app.tools.sanitize import make_json_serializable, sanitize_for_json
from app.streaming.sse_events import (
    SSEEvent,
    SSEEventType,
    AgentActivityData,
    TextChunkData,
    CitationData,
    RichCitationData,
    ToolResultData,
    ThinkingData,
    ArtifactData,
    ThreadTitleData,
    CompleteData,
    ErrorData,
)

logger = logging.getLogger(__name__)


class LangGraphEventMapper:
    """
    Maps LangGraph astream_events to SSE events.

    Handles event types:
    - on_chat_model_stream: Token streaming → text_chunk
    - on_tool_start: Tool invocation → agent_activity
    - on_tool_end: Tool completion → tool_result
    - on_custom_event: Custom events → thinking, citation

    Accumulates text and citations for the final complete event.
    """

    # Internal ReAct agent node names to filter out
    # These are implementation details, not meaningful to users
    INTERNAL_NODE_NAMES = {
        "agent",
        "tools",
        "call_model",
        "Prompt",
        "LangGraph",
        "RunnableSequence",
        "RunnableLambda",
        "ChannelWrite",
        "ChannelRead",
        "__start__",
        "__end__",
    }

    # Our meaningful subgraph names that we DO want to show
    SUBGRAPH_NAMES = {
        "entity_researcher",
        "text_researcher",
        "summary_researcher",
        "deep_reasoning",
        "mentor_matcher",
    }

    # V3 Orchestrator node names - these are the main workflow stages
    V3_NODE_NAMES = {
        "planner": "Planner",
        # "tool_worker" intentionally excluded - we show actual tool names instead
        "evaluator": "Evaluator",
        "increment_retry": "Retry",
        "synthesizer": "Synthesizer",
    }

    def __init__(self) -> None:
        self.accumulated_text: str = ""
        self.citations: list[RichCitationData] = []
        self.agent_trace: list[AgentActivityData] = []
        self._seen_citation_sources: set[str] = set()
        self._current_agent: str = "orchestrator"
        self._current_invocation_id: Optional[str] = None  # For parallel subagent tracking
        self._active_tools: set[str] = set()  # Track tools currently running
        self._handoff_summary: Optional[str] = None
        self._handoff_recent_messages: Optional[list[dict[str, Any]]] = None
        # Accumulate thinking content per-invocation to emit as single event (not per-chunk)
        # Dict maps (invocation_id or agent_name) -> {content, agent, invocation_id}
        # Using invocation_id as key enables parallel subagent support
        self._accumulated_thinking: dict[str, dict[str, Any]] = {}
        # Accumulate UI events for persistence (so they can be restored when loading thread)
        # These are stored alongside the thread and used to reconstruct the UI on reload
        self._ui_events: list[dict[str, Any]] = []
        # Sequence number for tool timeline ordering
        self._tool_sequence: int = 0

    def _get_invocation_id_for_event(self, event: dict[str, Any]) -> Optional[str]:
        """
        Get the invocation_id for an event. Override in subclasses for parallel tracking.

        The base implementation just returns the current invocation_id.
        Subclasses can override this to do more sophisticated lookup based on
        run_id mapping for parallel subagent support.

        Args:
            event: The LangGraph event

        Returns:
            The invocation_id for this event, or None
        """
        return self._current_invocation_id

    def map_event(self, event: dict[str, Any]) -> list[SSEEvent]:
        """
        Map a single LangGraph event to zero or more SSE events.

        Args:
            event: Event dict from graph.astream_events()

        Returns:
            List of SSE events to emit
        """
        sse_events: list[SSEEvent] = []
        kind = event.get("event")
        timestamp = time.time()
        metadata = event.get("metadata", {})

        # Determine current agent from event metadata (stateless, robust)
        self._current_agent = self._get_agent_name(event)

        if kind == "on_chat_model_stream":
            # Token streaming from LLM
            chunk = event.get("data", {}).get("chunk")
            if chunk and hasattr(chunk, "content") and chunk.content:
                raw_content = chunk.content

                # Handle content as either string or list of content blocks
                # GPT-5.x and deepagents return list format: [{'type': 'text', 'text': '...', 'index': 0}]
                # Claude extended thinking: [{'type': 'thinking', 'thinking': '...'}]
                # GPT reasoning: [{'type': 'reasoning', 'summary': [...]}] or reasoning_content attribute
                content = ""
                thinking_content = ""
                if isinstance(raw_content, str):
                    content = raw_content
                elif isinstance(raw_content, list):
                    # Extract text, thinking, and reasoning from content blocks
                    for block in raw_content:
                        if isinstance(block, dict):
                            block_type = block.get("type", "")
                            if block_type == "text":
                                text = block.get("text", "")
                                if text:
                                    content += text
                            elif block_type == "thinking":
                                # Claude extended thinking
                                thinking = block.get("thinking", "")
                                if thinking:
                                    thinking_content += thinking
                            elif block_type == "reasoning":
                                # GPT-5.x reasoning blocks - may have summary array
                                summary = block.get("summary", [])
                                if isinstance(summary, list):
                                    for item in summary:
                                        if isinstance(item, dict) and item.get("type") == "summary_text":
                                            thinking_content += item.get("text", "")
                                elif isinstance(summary, str):
                                    thinking_content += summary

                # Check for reasoning_content on chunk object (GPT/OpenAI format)
                if hasattr(chunk, "reasoning_content") and chunk.reasoning_content:
                    thinking_content += str(chunk.reasoning_content)

                # Check additional_kwargs for reasoning (alternative format)
                if hasattr(chunk, "additional_kwargs"):
                    additional = chunk.additional_kwargs or {}
                    if "reasoning_content" in additional:
                        thinking_content += str(additional["reasoning_content"])
                    # OpenRouter reasoning_details format
                    if "reasoning_details" in additional:
                        details = additional["reasoning_details"]
                        if isinstance(details, dict) and "content" in details:
                            thinking_content += str(details["content"])
                        elif isinstance(details, str):
                            thinking_content += details

                # Accumulate thinking content per-invocation (don't emit per-chunk to avoid UI spam)
                # Will emit when response text starts or at stream end
                # Use invocation_id as key for parallel subagent support, fallback to agent name
                if thinking_content:
                    agent = self._current_agent
                    invocation_id = self._get_invocation_id_for_event(event)
                    # Use invocation_id as key if available, else agent name (for orchestrator)
                    thinking_key = invocation_id or agent
                    if thinking_key not in self._accumulated_thinking:
                        self._accumulated_thinking[thinking_key] = {
                            "content": "",
                            "agent": agent,
                            "invocation_id": invocation_id,
                        }
                    self._accumulated_thinking[thinking_key]["content"] += thinking_content

                if content:
                    # Sanitize content to remove control characters that cause garbled output
                    # This fixes garbled markdown in subagent responses
                    content = sanitize_for_json(content) if isinstance(content, str) else content

                    # Text content means response has started - emit accumulated thinking first
                    # Use invocation_id as key for parallel subagent support
                    agent = self._current_agent
                    invocation_id = self._get_invocation_id_for_event(event)
                    thinking_key = invocation_id or agent
                    if thinking_key in self._accumulated_thinking:
                        thinking_data = self._accumulated_thinking[thinking_key]
                        if thinking_data.get("content"):
                            sse_events.append(
                                SSEEvent(
                                    event=SSEEventType.THINKING,
                                    data=ThinkingData(
                                        phase="reasoning",
                                        content=thinking_data["content"],
                                        agent=thinking_data.get("agent", agent),
                                        timestamp=timestamp,
                                        invocation_id=thinking_data.get("invocation_id"),
                                    ),
                                )
                            )
                            # Reset content after emitting (keep entry for potential more thinking)
                            self._accumulated_thinking[thinking_key]["content"] = ""
                    tags = set(event.get("tags") or [])
                    tags.update(metadata.get("tags") or [])
                    purpose = metadata.get("purpose")
                    if isinstance(metadata.get("metadata"), dict) and not purpose:
                        purpose = metadata["metadata"].get("purpose")
                    if "handoff_summary" in tags or purpose == "handoff_summary":
                        logger.debug("Filtering handoff summary stream output")
                        return sse_events

                    # Only accumulate and emit text from final response nodes
                    # Filter out internal LLM calls (planner, evaluator JSON output)
                    node_name = metadata.get("langgraph_node", "")

                    # V3 orchestrator: only show synthesizer output
                    # V1/V2 orchestrator: show agent output (default behavior)
                    is_v3_internal_node = node_name in ("planner", "evaluator", "increment_retry")

                    if not is_v3_internal_node:
                        # Only emit text for user-facing nodes (synthesizer, or v1/v2 agent)
                        self.accumulated_text += content

                        # Emit text chunk
                        sse_events.append(
                            SSEEvent(
                                event=SSEEventType.TEXT_CHUNK,
                                data=TextChunkData(
                                    chunk=content,
                                    agent=self._current_agent,
                                    is_partial=True,
                                    invocation_id=invocation_id,
                                ),
                            )
                        )

                        # Extract citations from streamed text
                        new_citations = self._extract_citations(content)
                        for citation in new_citations:
                            sse_events.append(
                                SSEEvent(event=SSEEventType.CITATION, data=citation)
                            )
                    else:
                        # Log that we're filtering internal output
                        logger.debug(f"Filtering internal LLM output from node: {node_name}")

        elif kind == "on_tool_start":
            # Tool invocation started
            tool_name = event.get("name", "unknown")
            tool_input = event.get("data", {}).get("input", {})
            invocation_id = self._get_invocation_id_for_event(event)

            # Track this tool as active
            self._active_tools.add(tool_name)

            # Increment sequence number for timeline ordering
            self._tool_sequence += 1

            # Create human-readable description
            tool_description = self._get_tool_description(tool_name, tool_input)

            # Sanitize tool_input to remove non-serializable objects (e.g., AsyncCallbackManager)
            safe_tool_input = make_json_serializable(tool_input)

            activity = AgentActivityData(
                agent=self._current_agent,
                action="tool_call",
                tool_name=tool_name,
                tool_args=safe_tool_input if isinstance(safe_tool_input, dict) else {"input": safe_tool_input},
                details=tool_description,
                timestamp=timestamp,
                invocation_id=invocation_id,
                sequence_number=self._tool_sequence,  # For timeline ordering
                description=tool_description,  # AI-readable description for active indicator
            )
            self.agent_trace.append(activity)
            sse_events.append(
                SSEEvent(event=SSEEventType.AGENT_ACTIVITY, data=activity)
            )

        elif kind == "on_tool_end":
            # Tool execution completed
            tool_name = event.get("name", "unknown")
            output = event.get("data", {}).get("output")
            invocation_id = self._get_invocation_id_for_event(event)

            # Check if the tool returned an error
            tool_success = True
            if isinstance(output, dict) and output.get("error"):
                tool_success = False
                # Include suggestion in summary for better error display
                error_msg = output.get("error_message", "Unknown error")
                suggestion = output.get("suggestion", "")
                result_summary = f"Error: {error_msg}"
                if suggestion:
                    result_summary += f" ({suggestion})"
            else:
                result_summary = self._summarize_tool_result(tool_name, output)

            # Also emit as agent_activity with "tool_complete" action for inline tool timeline
            # This allows the frontend to show tool completion in the inline timeline
            sse_events.append(
                SSEEvent(
                    event=SSEEventType.AGENT_ACTIVITY,
                    data=AgentActivityData(
                        agent=self._current_agent,
                        action="tool_complete",
                        tool_name=tool_name,
                        timestamp=timestamp,
                        invocation_id=invocation_id,
                        result_summary=result_summary,
                        success=tool_success,
                        # Include sanitized result details for expanded view (truncated for safety)
                        result_details=make_json_serializable(output) if output and not isinstance(output, str) or (isinstance(output, str) and len(output) < 5000) else None,
                    ),
                )
            )

            # Keep the legacy TOOL_RESULT event for backwards compatibility
            sse_events.append(
                SSEEvent(
                    event=SSEEventType.TOOL_RESULT,
                    data=ToolResultData(
                        agent=self._current_agent,
                        tool_name=tool_name,
                        result_summary=result_summary,
                        success=tool_success,
                        invocation_id=invocation_id,
                    ),
                )
            )

            # Extract citations from tool results
            tool_citations = self._extract_citations_from_tool_result(tool_name, output)
            for citation in tool_citations:
                sse_events.append(
                    SSEEvent(event=SSEEventType.CITATION, data=citation)
                )

        elif kind == "on_custom_event":
            # Custom events emitted via get_stream_writer()
            custom_data = event.get("data", {})
            event_type = custom_data.get("type")
            invocation_id = self._get_invocation_id_for_event(event)

            if event_type == "thinking":
                sse_events.append(
                    SSEEvent(
                        event=SSEEventType.THINKING,
                        data=ThinkingData(
                            phase=custom_data.get("phase", "reasoning"),
                            content=custom_data.get("content", ""),
                            agent=custom_data.get("agent", self._current_agent),
                            timestamp=timestamp,
                            invocation_id=invocation_id,
                        ),
                    )
                )
                # Also add to trace
                activity = AgentActivityData(
                    agent=custom_data.get("agent", self._current_agent),
                    action="thinking",
                    details=f"[{custom_data.get('phase', '').upper()}] {custom_data.get('content', '')[:100]}...",
                    timestamp=timestamp,
                    invocation_id=invocation_id,
                )
                self.agent_trace.append(activity)

            elif event_type == "citation":
                source = custom_data.get("source", "")
                if source and source not in self._seen_citation_sources:
                    self._seen_citation_sources.add(source)
                    # Handle custom citations - may have rich data or just basic
                    # source_number enables inline [source:N] citation markers
                    citation = RichCitationData(
                        source=source,
                        entity_type=custom_data.get("entity_type", "entity"),
                        display_name=custom_data.get("display_name", source),
                        content=custom_data.get("content", ""),
                        confidence=custom_data.get("confidence", 1.0),
                        group_key=custom_data.get("group_key", "entities"),
                        source_number=custom_data.get("source_number"),
                        metadata=custom_data.get("metadata"),
                    )
                    self.citations.append(citation)
                    sse_events.append(
                        SSEEvent(event=SSEEventType.CITATION, data=citation)
                    )
            elif event_type == "artifact":
                artifact = ArtifactData(
                    id=custom_data.get("id", ""),
                    artifact_type=custom_data.get("artifact_type")
                    or custom_data.get("artifactType")
                    or "document",
                    title=custom_data.get("title", "Artifact"),
                    summary=custom_data.get("summary"),
                    payload=custom_data.get("payload")
                    if custom_data.get("payload") is not None
                    else custom_data.get("data", {}),
                    origin=custom_data.get("origin"),
                    created_at=custom_data.get("created_at")
                    or custom_data.get("createdAt"),
                )
                sse_events.append(
                    SSEEvent(event=SSEEventType.ARTIFACT, data=artifact)
                )
            elif event_type == "handoff":
                self._handoff_summary = custom_data.get("summary")
                self._handoff_recent_messages = custom_data.get("recent_messages")

            elif event_type == "ui":
                # UI events can contain various frontend updates
                ui_name = custom_data.get("name")
                ui_props = custom_data.get("props", {})

                if ui_name == "thread_title":
                    # AI-generated thread title
                    title = ui_props.get("title")
                    if title:
                        sse_events.append(
                            SSEEvent(
                                event=SSEEventType.THREAD_TITLE,
                                data=ThreadTitleData(title=title),
                            )
                        )
                elif ui_name == "thinking":
                    # Thinking/reasoning content from middleware
                    content = ui_props.get("content", "")
                    phase = ui_props.get("phase", "reasoning")
                    if content:
                        sse_events.append(
                            SSEEvent(
                                event=SSEEventType.THINKING,
                                data=ThinkingData(
                                    phase=phase,
                                    content=content,
                                    agent=self._current_agent,
                                    timestamp=timestamp,
                                    invocation_id=invocation_id,
                                ),
                            )
                        )
                        # Also add to trace
                        activity = AgentActivityData(
                            agent=self._current_agent,
                            action="thinking",
                            details=f"[{phase.upper()}] {content[:100]}...",
                            timestamp=timestamp,
                            invocation_id=invocation_id,
                        )
                        self.agent_trace.append(activity)

        elif kind == "on_chain_start":
            # Subgraph/chain invocation - only show meaningful delegations
            chain_name = event.get("name", "")
            invocation_id = self._get_invocation_id_for_event(event)

            # Skip internal ReAct nodes - these are implementation noise
            if not chain_name or chain_name in self.INTERNAL_NODE_NAMES:
                return sse_events

            # Handle v3 orchestrator node transitions
            if chain_name in self.V3_NODE_NAMES:
                readable_name = self.V3_NODE_NAMES[chain_name]
                action_details = self._get_v3_node_description(chain_name)
                activity = AgentActivityData(
                    agent=readable_name,
                    action="thinking" if chain_name in ("planner", "evaluator", "synthesizer") else "working",
                    tool_name=chain_name,
                    details=action_details,
                    timestamp=timestamp,
                    invocation_id=invocation_id,
                )
                self.agent_trace.append(activity)
                sse_events.append(
                    SSEEvent(event=SSEEventType.AGENT_ACTIVITY, data=activity)
                )

            # Only emit events for meaningful subgraph delegations
            elif chain_name in self.SUBGRAPH_NAMES:
                readable_name = self._get_subgraph_display_name(chain_name)
                activity = AgentActivityData(
                    agent=self._current_agent,
                    action="delegate",
                    tool_name=chain_name,
                    details=f"Delegating to {readable_name}",
                    timestamp=timestamp,
                    invocation_id=invocation_id,
                )
                self.agent_trace.append(activity)
                sse_events.append(
                    SSEEvent(event=SSEEventType.AGENT_ACTIVITY, data=activity)
                )

        return sse_events

    def _extract_citations(self, text: str) -> list[RichCitationData]:
        """Extract [Source: xxx] citations from text."""
        new_citations: list[RichCitationData] = []
        pattern = r"\[Source:\s*([^\]]+)\]"

        for match in re.finditer(pattern, text, re.IGNORECASE):
            source = match.group(1).strip()
            if source in self._seen_citation_sources:
                continue
            self._seen_citation_sources.add(source)

            # Get surrounding context
            start = max(0, match.start() - 100)
            end = min(len(text), match.end() + 100)
            context = text[start:end].strip()

            citation = RichCitationData(
                source=f"text:{source}",
                entity_type="document",
                display_name=source[:50] if len(source) > 50 else source,
                content=context,
                confidence=1.0,
                group_key="documents",
            )
            self.citations.append(citation)
            new_citations.append(citation)

        return new_citations

    def _summarize_tool_result(self, tool_name: str, result: Any) -> str:
        """Create a human-readable summary of a tool result."""
        if result is None:
            return "No result"

        if isinstance(result, str):
            # Handle raw JSON strings (common with MCP tool results)
            if result.startswith('{"') or result.startswith('['):
                try:
                    import json
                    parsed = json.loads(result)
                    return self._summarize_tool_result(tool_name, parsed)
                except (json.JSONDecodeError, ValueError):
                    pass
            return result[:200] if len(result) > 200 else result

        if isinstance(result, dict):
            # Knowledge graph tools
            if tool_name == "query_graph":
                count = result.get("count", 0)
                return f"Query returned {count} results"
            elif tool_name == "find_entity":
                if result.get("found"):
                    return f"Found entity: {result.get('name', 'unknown')}"
                return result.get("message", "Entity not found")
            elif tool_name == "search_text":
                count = result.get("count", 0)
                return f"Found {count} text chunks"

            # Firecrawl web research tools
            elif tool_name == "firecrawl_search":
                results = result.get("results") or result.get("data", [])
                if isinstance(results, list):
                    count = len(results)
                    # Extract first result title for context
                    if count > 0 and isinstance(results[0], dict):
                        title = results[0].get("title", "")[:50]
                        return f"Found {count} web results" + (f": {title}..." if title else "")
                    return f"Found {count} web results"
                return "Web search completed"

            elif tool_name == "firecrawl_scrape":
                # Scrape returns page content
                title = result.get("title") or result.get("metadata", {}).get("title", "")
                url = result.get("url", "")
                if title:
                    return f"Scraped: {title[:60]}..."
                elif url:
                    return f"Scraped page: {url[:60]}..."
                return "Page scraped successfully"

            elif tool_name == "firecrawl_map":
                # Map returns list of URLs on a site
                urls = result.get("urls") or result.get("links", [])
                if isinstance(urls, list):
                    return f"Found {len(urls)} URLs on site"
                return "Site map retrieved"

            elif tool_name == "firecrawl_extract":
                # Extract returns structured data
                data = result.get("data") or result.get("extracted", {})
                if isinstance(data, dict):
                    keys = list(data.keys())[:3]
                    return f"Extracted: {', '.join(keys)}..." if keys else "Data extracted"
                return "Structured data extracted"

            # Mentor Hub tools
            elif tool_name == "get_mentor_hub_sessions":
                sessions = result.get("sessions", [])
                return f"Found {len(sessions)} sessions"

            elif tool_name == "get_mentor_hub_tasks":
                tasks = result.get("tasks", [])
                return f"Found {len(tasks)} tasks"

            elif tool_name == "search_mentor_hub_mentors":
                mentors = result.get("mentors", [])
                return f"Found {len(mentors)} mentors"

            elif tool_name == "get_mentor_hub_team":
                name = result.get("name", "")
                if name:
                    return f"Team: {name}"
                return "Team details retrieved"

            # Generic results handling
            elif "results" in result:
                return f"Found {len(result['results'])} results"
            elif "data" in result and isinstance(result["data"], list):
                return f"Retrieved {len(result['data'])} items"
            elif "content" in result:
                content = result["content"]
                if isinstance(content, str) and len(content) > 100:
                    return f"Retrieved content ({len(content)} chars)"
                return "Content retrieved"

        # List results
        if isinstance(result, list):
            return f"Retrieved {len(result)} items"

        return str(result)[:200]

    def _extract_citations_from_tool_result(
        self, tool_name: str, result: Any
    ) -> list[RichCitationData]:
        """Extract rich citations from tool results for user-friendly display.

        Handles all tool types:
        - Mentor Hub: tasks, sessions, teams, mentors
        - Knowledge base: search_text, search_chunks, find_entity, query_graph
        """
        new_citations: list[RichCitationData] = []

        if not isinstance(result, dict):
            return new_citations

        # =================================================================
        # MENTOR HUB: Tasks
        # =================================================================
        if tool_name == "get_mentor_hub_tasks" and "tasks" in result:
            for task in result.get("tasks", [])[:10]:
                task_id = task.get("id") or task.get("task_id", "unknown")
                source = f"task:{task_id}"
                if source in self._seen_citation_sources:
                    continue
                self._seen_citation_sources.add(source)

                # Build tooltip content
                content_parts = []
                if task.get("status"):
                    content_parts.append(f"Status: {task['status']}")
                if task.get("assignee"):
                    content_parts.append(f"Assigned to: {task['assignee']}")
                if task.get("due_date"):
                    content_parts.append(f"Due: {task['due_date']}")
                content = " | ".join(content_parts) if content_parts else task.get("description", "")[:100]

                citation = RichCitationData(
                    source=source,
                    entity_type="task",
                    display_name=task.get("title") or task.get("name", "Untitled task"),
                    content=content,
                    confidence=1.0,
                    group_key="tasks",
                    metadata={
                        "status": task.get("status"),
                        "due_date": task.get("due_date"),
                        "assignee": task.get("assignee"),
                        "priority": task.get("priority"),
                    },
                )
                self.citations.append(citation)
                new_citations.append(citation)

        # =================================================================
        # MENTOR HUB: Sessions
        # =================================================================
        elif tool_name == "get_mentor_hub_sessions" and "sessions" in result:
            for session in result.get("sessions", [])[:10]:
                session_id = session.get("id") or session.get("session_id", "unknown")
                source = f"session:{session_id}"
                if source in self._seen_citation_sources:
                    continue
                self._seen_citation_sources.add(source)

                # Format display name: "Jan 15 session with Dr. Smith"
                display_parts = []
                if session.get("scheduled_start"):
                    try:
                        from datetime import datetime
                        dt = datetime.fromisoformat(
                            session["scheduled_start"].replace("Z", "+00:00")
                        )
                        display_parts.append(dt.strftime("%b %d"))
                    except (ValueError, TypeError):
                        pass
                display_parts.append("session")
                if session.get("mentor"):
                    display_parts.append(f"with {session['mentor']}")
                display_name = " ".join(display_parts)

                citation = RichCitationData(
                    source=source,
                    entity_type="session",
                    display_name=display_name,
                    content=session.get("summary") or session.get("agenda") or "No summary available",
                    confidence=1.0,
                    group_key="sessions",
                    metadata={
                        "scheduled_start": session.get("scheduled_start"),
                        "mentor": session.get("mentor"),
                        "team": session.get("team"),
                        "type": session.get("type"),
                        "status": session.get("status"),
                    },
                )
                self.citations.append(citation)
                new_citations.append(citation)

        # =================================================================
        # MENTOR HUB: Team
        # =================================================================
        elif tool_name == "get_mentor_hub_team":
            if result.get("name") and "error" not in result:
                source = f"team:{result.get('id', 'unknown')}"
                if source not in self._seen_citation_sources:
                    self._seen_citation_sources.add(source)

                    member_count = result.get("active_member_count") or len(result.get("active_members", []))
                    content = f"{member_count} active members"
                    if result.get("description"):
                        content = result["description"][:150]

                    citation = RichCitationData(
                        source=source,
                        entity_type="team",
                        display_name=result.get("name"),
                        content=content,
                        confidence=1.0,
                        group_key="teams",
                        metadata={
                            "status": result.get("status"),
                            "member_count": member_count,
                            "cohorts": result.get("cohorts"),
                        },
                    )
                    self.citations.append(citation)
                    new_citations.append(citation)

        # =================================================================
        # MENTOR HUB: Mentors
        # =================================================================
        elif tool_name == "search_mentor_hub_mentors" and "mentors" in result:
            for mentor in result.get("mentors", [])[:10]:
                mentor_id = mentor.get("id", "unknown")
                source = f"mentor:{mentor_id}"
                if source in self._seen_citation_sources:
                    continue
                self._seen_citation_sources.add(source)

                bio_snippet = mentor.get("bio", "")[:100] if mentor.get("bio") else "Mentor profile"

                citation = RichCitationData(
                    source=source,
                    entity_type="mentor",
                    display_name=mentor.get("name", "Unknown mentor"),
                    content=bio_snippet,
                    confidence=1.0,
                    group_key="mentors",
                    metadata={
                        "email": mentor.get("email"),
                        "cohorts": mentor.get("cohorts"),
                    },
                )
                self.citations.append(citation)
                new_citations.append(citation)

        # =================================================================
        # KNOWLEDGE BASE: find_entity
        # =================================================================
        elif tool_name == "find_entity" and result.get("found"):
            name = result.get("name", "unknown")
            source = f"entity:{name}"
            if source not in self._seen_citation_sources:
                self._seen_citation_sources.add(source)
                citation = RichCitationData(
                    source=source,
                    entity_type="entity",
                    display_name=name,
                    content=result.get("description", "")[:200] or "Entity from knowledge graph",
                    confidence=1.0,
                    group_key="entities",
                )
                self.citations.append(citation)
                new_citations.append(citation)

        # =================================================================
        # KNOWLEDGE BASE: search_text, search_chunks, search_summaries
        # =================================================================
        elif tool_name in ("search_text", "search_chunks", "search_summaries"):
            results = result.get("results", [])
            for r in results[:5]:
                if isinstance(r, dict):
                    text = r.get("text", "")
                    chunk_id = r.get("id", "")[:8] if r.get("id") else str(hash(text))[:8]
                    source = f"doc:{chunk_id}"

                    if source in self._seen_citation_sources:
                        continue
                    self._seen_citation_sources.add(source)

                    # Extract document name from metadata or use excerpt
                    doc_name = (
                        r.get("document_name")
                        or r.get("_document_name")
                        or r.get("name")
                        or text[:50] + "..." if len(text) > 50 else text or "Document"
                    )
                    excerpt = text[:150] + "..." if len(text) > 150 else text

                    citation = RichCitationData(
                        source=source,
                        entity_type="document",
                        display_name=doc_name,
                        content=excerpt,
                        confidence=0.85,
                        group_key="documents",
                        metadata={
                            "excerpt": excerpt,
                            "search_type": tool_name,
                        },
                    )
                    self.citations.append(citation)
                    new_citations.append(citation)

        # =================================================================
        # KNOWLEDGE BASE: query_graph
        # =================================================================
        elif tool_name == "query_graph":
            results = result.get("results", [])
            for r in results[:3]:
                if isinstance(r, dict):
                    name = r.get("name") or r.get("id", "")[:20]
                    if name:
                        source = f"graph:{name}"
                        if source not in self._seen_citation_sources:
                            self._seen_citation_sources.add(source)
                            citation = RichCitationData(
                                source=source,
                                entity_type="entity",
                                display_name=name,
                                content=r.get("text", r.get("description", ""))[:200] or "Graph query result",
                                confidence=0.9,
                                group_key="entities",
                            )
                            self.citations.append(citation)
                            new_citations.append(citation)

        return new_citations

    def capture_ui_event(self, event: SSEEvent) -> None:
        """
        Capture a UI event for persistence.

        These events are stored alongside the thread so the UI can be
        reconstructed when loading historical conversations.

        Only captures events that contain meaningful UI state:
        - agent_activity: Tool calls and agent actions
        - tool_result: Tool results with summaries
        - thinking: Reasoning/planning content
        - artifact: Created artifacts
        """
        # Only persist events that help reconstruct the UI
        persistable_types = {
            SSEEventType.AGENT_ACTIVITY,
            SSEEventType.TOOL_RESULT,
            SSEEventType.THINKING,
            SSEEventType.ARTIFACT,
            SSEEventType.TODO,
        }

        if event.event in persistable_types:
            # Store as serializable dict
            event_dict = {
                "type": event.event.value if hasattr(event.event, 'value') else str(event.event),
                "data": event.data.model_dump() if hasattr(event.data, 'model_dump') else event.data,
                "timestamp": time.time(),
            }
            self._ui_events.append(event_dict)

    def get_ui_events(self) -> list[dict[str, Any]]:
        """Get accumulated UI events for persistence."""
        return self._ui_events

    def flush_pending_events(self) -> list[SSEEvent]:
        """Flush any pending accumulated events (e.g., thinking content).

        Call this before create_complete_event to ensure all content is emitted.
        """
        events: list[SSEEvent] = []
        timestamp = time.time()

        # Emit any remaining accumulated thinking for all invocations
        # Dict structure: {thinking_key: {content, agent, invocation_id}}
        for thinking_key, thinking_data in list(self._accumulated_thinking.items()):
            content = thinking_data.get("content", "")
            if content:
                event = SSEEvent(
                    event=SSEEventType.THINKING,
                    data=ThinkingData(
                        phase="reasoning",
                        content=content,
                        agent=thinking_data.get("agent", "orchestrator"),
                        timestamp=timestamp,
                        invocation_id=thinking_data.get("invocation_id"),
                    ),
                )
                events.append(event)
                # Also capture for persistence
                self.capture_ui_event(event)
        # Clear all accumulated thinking
        self._accumulated_thinking = {}

        return events

    def create_complete_event(self, thread_id: Optional[str] = None) -> SSEEvent:
        """Create the final complete event with accumulated data."""
        return SSEEvent(
            event=SSEEventType.COMPLETE,
            data=CompleteData(
                status="success",
                thread_id=thread_id,
                full_message=self.accumulated_text,
                citations=self.citations,
                agent_trace=self.agent_trace,
                handoff_summary=self._handoff_summary,
                handoff_recent_messages=self._handoff_recent_messages,
            ),
        )

    def create_error_event(self, message: str, code: str = "STREAM_ERROR") -> SSEEvent:
        """Create an error event."""
        return SSEEvent(
            event=SSEEventType.ERROR,
            data=ErrorData(
                message=message,
                code=code,
                recoverable=False,
            ),
        )

    def _get_tool_description(self, tool_name: str, tool_input: Any) -> str:
        """Create a human-readable description for a tool call."""
        descriptions = {
            # Knowledge graph tools
            "query_graph": "Querying knowledge graph",
            "find_entity": "Finding entity",
            "search_text": "Searching documents",
            "search_chunks": "Searching text passages",
            "search_summaries": "Searching summaries",
            "search_graph": "Graph-enhanced search",
            "search_rag": "RAG search",
            "get_graph_schema": "Getting graph schema",
            "remember": "Storing memory",
            "recall": "Recalling memories",
            # Mentor Hub live data tools
            "get_mentor_hub_sessions": "Fetching sessions",
            "get_mentor_hub_team": "Fetching team details",
            "search_mentor_hub_mentors": "Searching mentors",
            "get_mentor_hub_tasks": "Fetching tasks",
            "get_mentor_hub_user_context": "Getting user context",
            # Subgraph tools
            "entity_researcher": "Entity research",
            "text_researcher": "Text research",
            "summary_researcher": "Summary research",
            "deep_reasoning": "Deep reasoning",
            "mentor_matcher": "Mentor matching",
        }

        base_desc = descriptions.get(tool_name, f"Calling {tool_name}")

        # Add context from input if available
        if isinstance(tool_input, dict):
            if "query" in tool_input:
                query = str(tool_input["query"])[:50]
                return f"{base_desc}: {query}..."
            elif "name" in tool_input:
                return f"{base_desc}: {tool_input['name']}"
            elif "cypher" in tool_input or "cypher_query" in tool_input:
                cypher = str(tool_input.get("cypher") or tool_input.get("cypher_query", ""))[:40]
                return f"{base_desc}: {cypher}..."
            elif "team_id" in tool_input:
                return f"{base_desc}: {tool_input['team_id']}"
            elif "expertise" in tool_input:
                return f"{base_desc}: {tool_input['expertise']}"

        return base_desc

    def _get_subgraph_display_name(self, subgraph_name: str) -> str:
        """Get a human-readable display name for a subgraph."""
        display_names = {
            "entity_researcher": "Entity Researcher",
            "text_researcher": "Text Researcher",
            "summary_researcher": "Summary Researcher",
            "deep_reasoning": "Deep Reasoning Agent",
            "mentor_matcher": "Mentor Matcher",
        }
        return display_names.get(subgraph_name, subgraph_name)

    def _get_v3_node_description(self, node_name: str) -> str:
        """Get a human-readable description for v3 orchestrator nodes."""
        descriptions = {
            "planner": "Analyzing query and planning research strategy...",
            # tool_worker excluded - we show actual tool names instead
            "evaluator": "Evaluating results quality...",
            "increment_retry": "Preparing to retry with refined strategy...",
            "synthesizer": "Synthesizing final answer from research...",
        }
        return descriptions.get(node_name, f"Running {node_name}...")

    def _get_agent_name(self, event: dict[str, Any]) -> str:
        """
        Determine the agent name from event metadata.

        This is a stateless approach - we look at the langgraph_node in the
        event metadata to determine which agent emitted the event. This is
        more robust than tracking start/end states.

        Subclasses can override this to customize agent name resolution.
        """
        metadata = event.get("metadata", {})
        node_name = metadata.get("langgraph_node", "")
        agent_name = metadata.get("lc_agent_name", "")

        # Prefer lc_agent_name if set, otherwise use node_name
        name_to_map = agent_name or node_name or "orchestrator"
        return self._map_agent_name(name_to_map)

    def _map_agent_name(self, node_name: str) -> str:
        """Map internal LangGraph node names to user-friendly agent names."""
        # Map internal ReAct node names to meaningful names
        internal_to_friendly = {
            "agent": "orchestrator",
            "tools": "orchestrator",
            "call_model": "orchestrator",
            "__start__": "orchestrator",
            "__end__": "orchestrator",
        }

        # If it's an internal name, map it
        if node_name in internal_to_friendly:
            return internal_to_friendly[node_name]

        # If it's a v3 orchestrator node, use the display name
        if node_name in self.V3_NODE_NAMES:
            return self.V3_NODE_NAMES[node_name]

        # If it's a subgraph name, use the display name
        if node_name in self.SUBGRAPH_NAMES:
            return self._get_subgraph_display_name(node_name)

        # Otherwise return as-is (could be a custom node)
        return node_name
