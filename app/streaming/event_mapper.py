"""Maps LangGraph streaming events to SSE events."""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Optional

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
        self._active_tools: set[str] = set()  # Track tools currently running
        self._handoff_summary: Optional[str] = None
        self._handoff_recent_messages: Optional[list[dict[str, Any]]] = None

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

        # Track current agent from metadata, mapping internal names to user-friendly ones
        metadata = event.get("metadata", {})
        node_name = metadata.get("langgraph_node", "orchestrator")
        if node_name:
            self._current_agent = self._map_agent_name(node_name)

        if kind == "on_chat_model_stream":
            # Token streaming from LLM
            chunk = event.get("data", {}).get("chunk")
            if chunk and hasattr(chunk, "content") and chunk.content:
                content = chunk.content
                if isinstance(content, str) and content:
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

            # Track this tool as active
            self._active_tools.add(tool_name)

            # Create human-readable description
            tool_description = self._get_tool_description(tool_name, tool_input)

            activity = AgentActivityData(
                agent=self._current_agent,
                action="tool_call",
                tool_name=tool_name,
                tool_args=tool_input if isinstance(tool_input, dict) else {"input": tool_input},
                details=tool_description,
                timestamp=timestamp,
            )
            self.agent_trace.append(activity)
            sse_events.append(
                SSEEvent(event=SSEEventType.AGENT_ACTIVITY, data=activity)
            )

        elif kind == "on_tool_end":
            # Tool execution completed
            tool_name = event.get("name", "unknown")
            output = event.get("data", {}).get("output")

            result_summary = self._summarize_tool_result(tool_name, output)

            sse_events.append(
                SSEEvent(
                    event=SSEEventType.TOOL_RESULT,
                    data=ToolResultData(
                        agent=self._current_agent,
                        tool_name=tool_name,
                        result_summary=result_summary,
                        success=True,
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

            if event_type == "thinking":
                sse_events.append(
                    SSEEvent(
                        event=SSEEventType.THINKING,
                        data=ThinkingData(
                            phase=custom_data.get("phase", "reasoning"),
                            content=custom_data.get("content", ""),
                            agent=custom_data.get("agent", self._current_agent),
                            timestamp=timestamp,
                        ),
                    )
                )
                # Also add to trace
                activity = AgentActivityData(
                    agent=custom_data.get("agent", self._current_agent),
                    action="thinking",
                    details=f"[{custom_data.get('phase', '').upper()}] {custom_data.get('content', '')[:100]}...",
                    timestamp=timestamp,
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

        elif kind == "on_chain_start":
            # Subgraph/chain invocation - only show meaningful delegations
            chain_name = event.get("name", "")

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
            return result[:200] if len(result) > 200 else result

        if isinstance(result, dict):
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
            elif "results" in result:
                return f"Found {len(result['results'])} results"

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
