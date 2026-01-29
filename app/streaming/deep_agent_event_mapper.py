"""
Deep Agent Event Mapper - Extended SSE event mapping for Deep Agents.

Extends the base LangGraphEventMapper to handle Deep Agent-specific events:
- write_todos: Planning/task decomposition -> thinking events
- task: Subagent delegation -> agent_activity events
- Filesystem operations: Context management (filtered for internal ops)
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from typing import Any

from app.streaming.event_mapper import LangGraphEventMapper
from app.streaming.sse_events import (
    SSEEvent,
    SSEEventType,
    AgentActivityData,
    ArtifactData,
    ThinkingData,
    TodoData,
    TodoItemData,
)
from app.tools.sanitize import sanitize_for_json

logger = logging.getLogger(__name__)


class DeepAgentEventMapper(LangGraphEventMapper):
    """
    Extended event mapper for Deep Agent streaming.

    Handles Deep Agent-specific tool calls:
    - write_todos: Emits thinking events with task breakdown
    - task: Emits agent_activity events for subagent delegation
    - Filesystem ops: Filters internal context files, shows user-relevant ones
    """

    # Deep Agent built-in tool names
    DEEP_AGENT_TOOLS = {
        "write_todos",
        "read_todos",
        "ls",
        "read_file",
        "write_file",
        "edit_file",
        "glob",
        "grep",
        "task",
        "subagent",  # deepagents library uses this name for task delegation
    }

    # Internal paths to filter (don't show to user)
    INTERNAL_PATHS = {
        "/context/",
        "/.cache/",
        "/.tmp/",
    }

    def __init__(self) -> None:
        super().__init__()
        self._current_todos: list[dict] = []
        # Stack for nested subagent tracking (sequential): [{name: str, invocation_id: str, run_id: str}]
        self._active_subagents: list[dict[str, str]] = []
        # Map run_id -> invocation_id for parallel subagent tracking
        # When multiple subagents run in parallel, each has a unique run_id
        self._run_to_invocation: dict[str, str] = {}
        # Map run_id -> agent_name for parallel tracking
        self._run_to_agent: dict[str, str] = {}
        self._current_invocation_id: str | None = None  # Current invocation ID for event attribution (fallback)
        self._artifact_writes: dict[str, dict[str, str]] = {}
        self._artifact_write_queue: list[dict[str, str]] = []
        self._event_sequence: int = 0  # Track event order for debugging

    def _get_invocation_id_for_event(self, event: dict) -> str | None:
        """
        Look up the invocation_id for an event based on its context.

        For parallel subagent tracking, we need to map events to their specific
        invocation even when multiple subagents of the same type are running.

        Priority:
        1. Direct run_id match in _run_to_invocation
        2. Config metadata (injected when subagent is invoked)
        3. Parent invocation from metadata
        4. Parent run context from checkpoint_ns or parent_ids
        5. Match by agent name from active subagents stack
        6. Current invocation_id (fallback)

        Returns:
            The invocation_id for this event, or None if not in subagent context
        """
        run_id = event.get("run_id") or event.get("data", {}).get("run_id")
        metadata = event.get("metadata", {})

        # Priority 1: Direct run_id lookup
        if run_id and run_id in self._run_to_invocation:
            return self._run_to_invocation[run_id]

        # Priority 2: Config metadata (injected by execute_subagent)
        # This is the key fix for parallel subagent event routing
        configurable = metadata.get("configurable", {})
        if configurable.get("invocation_id"):
            return configurable["invocation_id"]

        # Priority 3: Parent invocation from metadata (set when delegating)
        if metadata.get("parent_invocation_id"):
            return metadata["parent_invocation_id"]

        # Priority 4: Try to find parent run context from checkpoint_ns or parent_ids
        parent_ids = metadata.get("parent_ids", [])
        for parent_id in parent_ids:
            if parent_id in self._run_to_invocation:
                return self._run_to_invocation[parent_id]

        # Priority 5: Try matching by agent name from active subagents
        agent_name = self._get_agent_name(event)
        for subagent in reversed(self._active_subagents):
            if subagent.get("name") == agent_name:
                return subagent.get("invocation_id")

        # Priority 6: Fallback to current invocation_id
        return self._current_invocation_id

    def _log_sse_events(self, events: list[SSEEvent]) -> list[SSEEvent]:
        """Log SSE events being emitted for sequence debugging."""
        seq = self._event_sequence
        for sse in events:
            event_type = sse.event.value if hasattr(sse.event, 'value') else str(sse.event)
            if event_type == "text_chunk":
                data = sse.data
                chunk_preview = ""
                if hasattr(data, 'chunk'):
                    chunk_preview = data.chunk[:50].replace("\n", " ") if data.chunk else ""
                agent = getattr(data, 'agent', '?')
                logger.info(f"[SSE-OUT:{seq:04d}] {event_type} agent={agent} chunk=\"{chunk_preview}...\"")
            elif event_type == "agent_activity":
                data = sse.data
                action = getattr(data, 'action', '?')
                agent = getattr(data, 'agent', '?')
                tool = getattr(data, 'tool_name', '')
                logger.info(f"[SSE-OUT:{seq:04d}] {event_type} action={action} agent={agent} tool={tool}")
            else:
                logger.debug(f"[SSE-OUT:{seq:04d}] {event_type}")
        return events

    def map_event(self, event: dict[str, Any]) -> list[SSEEvent]:
        """
        Map event with Deep Agent-specific handling.

        Intercepts Deep Agent tool calls to provide better user feedback,
        then delegates to parent for standard events.
        """
        sse_events: list[SSEEvent] = []
        kind = event.get("event")
        timestamp = time.time()

        # Always update current agent from event metadata (stateless approach)
        # This ensures correct attribution even when we handle events directly
        self._current_agent = self._get_agent_name(event)

        # Log incoming LangGraph event for sequence debugging
        self._event_sequence += 1
        seq = self._event_sequence
        if kind in ("on_tool_start", "on_tool_end"):
            tool_name = event.get("name", "?")
            logger.info(f"[SEQ:{seq:04d}] {kind} tool={tool_name} agent={self._current_agent}")
        elif kind == "on_chat_model_stream":
            chunk = event.get("data", {}).get("chunk")
            content_preview = ""
            if chunk and hasattr(chunk, "content"):
                raw = chunk.content
                if isinstance(raw, str):
                    content_preview = raw[:50].replace("\n", " ")
                elif isinstance(raw, list) and raw:
                    content_preview = str(raw[0])[:50]
            logger.info(f"[SEQ:{seq:04d}] {kind} agent={self._current_agent} preview=\"{content_preview}...\"")
        else:
            logger.debug(f"[SEQ:{seq:04d}] {kind} agent={self._current_agent}")

        # =====================================================================
        # HANDLE TOOL STARTS
        # =====================================================================
        if kind == "on_tool_start":
            tool_name = event.get("name", "")
            tool_input = event.get("data", {}).get("input", {})
            run_id = event.get("run_id") or event.get("data", {}).get("run_id")

            if tool_name in ("write_file", "edit_file"):
                path = tool_input.get("path") or tool_input.get("file_path", "")
                if path.startswith("/artifacts/"):
                    content = (
                        tool_input.get("content")
                        or tool_input.get("new_content")
                        or tool_input.get("new_string")
                        or ""
                    )
                    entry = {"path": path, "content": content}
                    if run_id:
                        self._artifact_writes[run_id] = entry
                    else:
                        self._artifact_write_queue.append(entry)

            # -----------------------------------------------------------------
            # write_todos: Emit thinking event with task breakdown
            # -----------------------------------------------------------------
            if tool_name == "write_todos":
                todos = tool_input.get("todos", [])
                self._current_todos = todos

                # Format todos for display
                todo_text = self._format_todos(todos)

                # Emit thinking event
                sse_events.append(
                    SSEEvent(
                        event=SSEEventType.THINKING,
                        data=ThinkingData(
                            phase="planning",
                            content=todo_text,
                            agent="Planner",
                            timestamp=timestamp,
                        ),
                    )
                )

                # Also emit agent activity for trace
                sse_events.append(
                    SSEEvent(
                        event=SSEEventType.AGENT_ACTIVITY,
                        data=AgentActivityData(
                            agent="Planner",
                            action="planning",
                            tool_name="write_todos",
                            details=f"Created {len(todos)} tasks",
                            timestamp=timestamp,
                        ),
                    )
                )

                # Emit dedicated TODO event for UI progress panels
                # Convert raw todo dicts to TodoItemData objects
                todo_items = [
                    TodoItemData(
                        id=todo.get("id", f"todo-{i}"),
                        content=todo.get("content", ""),
                        status=todo.get("status", "pending"),
                        activeForm=todo.get("activeForm"),
                    )
                    for i, todo in enumerate(todos)
                ]

                # Get the invocation_id for proper grouping in progress panel
                # This fixes the issue where write_todos from subagents fell into "Planner" bucket
                invocation_id = self._get_invocation_id_for_event(event)

                # Determine the owning agent - use current agent context
                # For subagents, this will be the subagent name; for orchestrator, it's "Planner"
                owning_agent = self._current_agent if self._active_subagents else "Planner"

                sse_events.append(
                    SSEEvent(
                        event=SSEEventType.TODO,
                        data=TodoData(
                            todos=todo_items,
                            timestamp=timestamp,
                            agent=owning_agent,
                            invocation_id=invocation_id,
                        ),
                    )
                )
                return sse_events

            # -----------------------------------------------------------------
            # task/subagent: Emit subagent delegation event and track subagent context
            # deepagents library uses "subagent" tool name, Claude Code uses "task"
            # -----------------------------------------------------------------
            if tool_name in ("task", "subagent"):
                # Try multiple fields to get the subagent name
                subagent_name = (
                    tool_input.get("agent") or
                    tool_input.get("name") or
                    tool_input.get("subagent_type") or
                    tool_input.get("agent_type") or
                    tool_input.get("type") or
                    tool_input.get("description", "")[:50].split()[0] if tool_input.get("description") else None
                )

                # If still no name, try to infer from task description
                task_description = tool_input.get("task") or tool_input.get("prompt", "")
                if not subagent_name or subagent_name == "subagent":
                    subagent_name = self._infer_subagent_name(task_description, tool_input)

                # Final fallback
                if not subagent_name:
                    subagent_name = "assistant"

                # NEW: Get AI-generated display_name if provided, otherwise generate one
                # Format: "Role – Concise task description"
                display_name = tool_input.get("display_name")
                if not display_name:
                    # Generate display_name from subagent_name and task description
                    role_name = self._get_readable_agent_name(subagent_name)
                    task_summary = self._summarize_task_description(task_description)
                    display_name = f"{role_name} – {task_summary}" if task_summary else role_name

                # Generate unique invocation ID for this delegation
                # Format: {sanitized_name}_{short_uuid} to enable parallel same-agent tracking
                # Sanitize to ensure only alphanumeric, underscore, hyphen (safe for JSON/Redis)
                safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', subagent_name)[:50]
                invocation_id = f"{safe_name}_{uuid.uuid4().hex[:8]}"

                # Emit delegation event BEFORE switching context
                # Truncate task description for display and sanitize to remove control chars
                display_task = task_description[:100]
                if len(task_description) > 100:
                    display_task += "..."
                safe_details = sanitize_for_json(f"Delegating: {display_task}")

                sse_events.append(
                    SSEEvent(
                        event=SSEEventType.AGENT_ACTIVITY,
                        data=AgentActivityData(
                            agent=self._current_agent,
                            action="delegate",
                            tool_name=subagent_name,
                            details=safe_details,
                            timestamp=timestamp,
                            invocation_id=invocation_id,
                            display_name=display_name,  # NEW: AI-generated or inferred name
                        ),
                    )
                )

                # Now switch to subagent context - all subsequent tool calls
                # will be attributed to this subagent until task completes
                self._active_subagents.append({
                    "name": subagent_name,
                    "invocation_id": invocation_id,
                    "run_id": run_id or "",
                })
                self._current_agent = subagent_name
                self._current_invocation_id = invocation_id

                # Store run_id -> invocation_id mapping for parallel subagent tracking
                # This allows us to route events to the correct invocation when multiple
                # subagents of the same type run in parallel
                if run_id:
                    self._run_to_invocation[run_id] = invocation_id
                    self._run_to_agent[run_id] = subagent_name

                logger.info(f"[SSE-OUT:{self._event_sequence:04d}] >>> DELEGATE to {subagent_name} (invocation_id={invocation_id}, run_id={run_id})")
                return self._log_sse_events(sse_events)

            # -----------------------------------------------------------------
            # Filesystem operations: Filter internal, show user-relevant
            # -----------------------------------------------------------------
            if tool_name in ("ls", "read_file", "write_file", "edit_file", "glob", "grep"):
                path = tool_input.get("path", "") or tool_input.get("pattern", "")

                # Filter internal paths
                for internal_path in self.INTERNAL_PATHS:
                    if path.startswith(internal_path):
                        logger.debug(f"Filtering internal filesystem op: {tool_name} {path}")
                        return sse_events

                # Artifact writes to /artifacts/ are handled in on_tool_end
                # (content is already captured at lines 97-110 above)
                if path.startswith("/artifacts/") and tool_name in ("write_file", "edit_file"):
                    logger.debug(f"Artifact write started: {path}")
                    return sse_events

                # Build action description based on tool and path
                if path.startswith("/memories/"):
                    action_desc = {
                        "read_file": "Reading memory",
                        "write_file": "Saving memory",
                        "ls": "Listing memories",
                        "edit_file": "Updating memory",
                    }.get(tool_name, tool_name)
                    action_type = "memory"
                else:
                    # General file operations
                    action_desc = {
                        "read_file": "Reading file",
                        "write_file": "Writing file",
                        "edit_file": "Editing file",
                        "ls": "Listing directory",
                        "glob": "Searching files",
                        "grep": "Searching content",
                    }.get(tool_name, tool_name)
                    action_type = "file"

                # Emit activity event for all visible file operations
                # Sanitize path in case it contains control characters
                safe_details = sanitize_for_json(f"{action_desc}: {path}" if path else action_desc)
                sse_events.append(
                    SSEEvent(
                        event=SSEEventType.AGENT_ACTIVITY,
                        data=AgentActivityData(
                            agent=self._current_agent,
                            action=action_type,
                            tool_name=tool_name,
                            details=safe_details,
                            timestamp=timestamp,
                            invocation_id=self._current_invocation_id,
                        ),
                    )
                )
                return sse_events

            # -----------------------------------------------------------------
            # read_todos: Just log, don't emit
            # -----------------------------------------------------------------
            if tool_name == "read_todos":
                logger.debug("Agent reading todos")
                return sse_events

        # =====================================================================
        # HANDLE TOOL ENDS
        # =====================================================================
        if kind == "on_tool_end":
            tool_name = event.get("name", "")
            run_id = event.get("run_id") or event.get("data", {}).get("run_id")

            if tool_name in ("write_file", "edit_file"):
                entry = None
                if run_id and run_id in self._artifact_writes:
                    entry = self._artifact_writes.pop(run_id)
                elif self._artifact_write_queue:
                    entry = self._artifact_write_queue.pop(0)
                else:
                    output = event.get("data", {}).get("output")
                    if isinstance(output, str) and "/artifacts/" in output:
                        start = output.find("/artifacts/")
                        end = output.find(" ", start)
                        path = output[start:] if end == -1 else output[start:end]
                        entry = {"path": path, "content": ""}

                if entry and entry.get("path", "").startswith("/artifacts/"):
                    path = entry["path"]
                    title = path.split("/")[-1] or "Artifact file"
                    content = entry.get("content", "")
                    is_saved = "/artifacts/saved/" in path
                    action = "create" if tool_name == "write_file" else "update"

                    # Sanitize content to remove control characters that break Redis JSON
                    safe_content = sanitize_for_json(content) if content else ""

                    # Emit artifact event
                    sse_events.append(
                        SSEEvent(
                            event=SSEEventType.ARTIFACT,
                            data=ArtifactData(
                                id=path,
                                artifact_type="file",
                                title=title,
                                summary=path,
                                payload={"path": path, "content": safe_content, "action": action, "saved": is_saved},
                                created_at=timestamp,
                            ),
                        )
                    )

                    # Emit activity for trace
                    action_desc = "Created" if tool_name == "write_file" else "Updated"
                    if is_saved:
                        action_desc = f"Saved permanently: {title}"
                    else:
                        action_desc = f"{action_desc}: {title}"

                    sse_events.append(
                        SSEEvent(
                            event=SSEEventType.AGENT_ACTIVITY,
                            data=AgentActivityData(
                                agent=self._current_agent,
                                action="artifact",
                                tool_name=tool_name,
                                details=sanitize_for_json(action_desc),
                                timestamp=timestamp,
                                invocation_id=self._current_invocation_id,
                            ),
                        )
                    )
                    return sse_events

            # task/subagent completion: Subagent returned, restore previous context
            if tool_name in ("task", "subagent"):
                # Pop the completed subagent from the stack
                if self._active_subagents:
                    subagent_info = self._active_subagents.pop()
                    subagent_name = subagent_info["name"]
                    completed_invocation_id = subagent_info["invocation_id"]

                    sse_events.append(
                        SSEEvent(
                            event=SSEEventType.AGENT_ACTIVITY,
                            data=AgentActivityData(
                                agent=subagent_name,
                                action="completed",
                                tool_name=subagent_name,  # Use agent name for frontend matching
                                details="Subagent completed task",
                                timestamp=timestamp,
                                invocation_id=completed_invocation_id,
                            ),
                        )
                    )

                # Restore context to parent agent (previous subagent or orchestrator)
                if self._active_subagents:
                    parent_info = self._active_subagents[-1]
                    self._current_agent = parent_info["name"]
                    self._current_invocation_id = parent_info["invocation_id"]
                else:
                    self._current_agent = "orchestrator"
                    self._current_invocation_id = None

                logger.info(f"[SSE-OUT:{self._event_sequence:04d}] <<< COMPLETED subagent, restored to {self._current_agent} (invocation_id={self._current_invocation_id})")
                return self._log_sse_events(sse_events)

            # write_todos completion: Could update progress display
            if tool_name == "write_todos":
                logger.debug("Todo list updated")
                return sse_events

            # Skip other Deep Agent internal tool completions
            if tool_name in self.DEEP_AGENT_TOOLS:
                return sse_events

        # =====================================================================
        # DELEGATE TO PARENT FOR STANDARD EVENTS
        # =====================================================================
        parent_events = super().map_event(event)
        return self._log_sse_events(parent_events)

    def _format_todos(self, todos: list[dict]) -> str:
        """Format todos for display in thinking event."""
        if not todos:
            return "No tasks planned"

        lines = ["Planning tasks:"]
        for todo in todos:
            status = todo.get("status", "pending")
            content = todo.get("content", "")
            active_form = todo.get("activeForm", "")

            # Status emoji
            emoji = {
                "pending": "[ ]",
                "in_progress": "[>]",
                "completed": "[x]",
            }.get(status, "[ ]")

            # Use activeForm if available for more readable display
            display_text = active_form or content
            lines.append(f"{emoji} {display_text}")

        return "\n".join(lines)

    def _get_agent_name(self, event: dict) -> str:
        """
        Determine agent name from event metadata (stateless approach).

        Maps LangGraph node names to UI agent names. This is robust because
        it doesn't rely on tracking start/end states.

        Uses checkpoint_ns (namespace) to detect if we're inside a subagent's
        context, which is important for correctly attributing "tools" nodes.
        """
        metadata = event.get("metadata", {})
        node_name = metadata.get("langgraph_node", "")
        agent_name = metadata.get("lc_agent_name", "")
        checkpoint_ns = metadata.get("checkpoint_ns", "")

        # Use the most specific name available
        name = agent_name or node_name

        # Map Deep Agent node names to UI agent names
        # Nodes that are subagents (will be nested in UI)
        subagent_nodes = {
            "integration_expert": "integration_expert",
            "research_expert": "research_expert",
            "web_researcher": "web_researcher",
            "research-agent": "research_agent",
            "web-researcher": "web_researcher",
            "mentor-matcher": "mentor_matcher",
            "general-purpose": "assistant",
        }

        # Check if we're inside a subagent's namespace
        # checkpoint_ns format is typically "parent_node:child_node" or contains subagent name
        subagent_from_ns = None
        if checkpoint_ns:
            for subagent_key in subagent_nodes:
                if subagent_key in checkpoint_ns:
                    subagent_from_ns = subagent_nodes[subagent_key]
                    break

        # If we're in a subagent namespace and the node is "tools" or "model",
        # attribute to the subagent, not orchestrator
        # IMPORTANT: Only if we actually have active subagents - after subagent completion,
        # the checkpoint_ns might still reference the subagent path, but we should attribute to orchestrator
        if subagent_from_ns and name in ("tools", "model", "agent") and self._active_subagents:
            return subagent_from_ns

        # Direct subagent node match
        if name in subagent_nodes:
            return subagent_nodes[name]

        # Nodes that are the main orchestrator (shown at top level)
        # Only use these if NOT inside a subagent namespace
        orchestrator_nodes = {
            "mentor-hub-agent": "orchestrator",
            "model": "orchestrator",
            "agent": "orchestrator",
            "tools": "orchestrator",
            "__start__": "orchestrator",
            "__end__": "orchestrator",
            "": "orchestrator",
        }

        if name in orchestrator_nodes and not subagent_from_ns:
            return orchestrator_nodes[name]

        # If unknown node, check if it looks like a subagent name
        # (anything that's not explicitly orchestrator is treated as subagent)
        if name and name not in ("orchestrator", "supervisor", "synthesizer"):
            return name  # Return as-is, frontend will treat as subagent

        return "orchestrator"

    def _map_agent_name(self, node_name: str) -> str:
        """Map internal node names to user-friendly names."""
        # First check Deep Agent specific names
        deep_agent_names = {
            "mentor-hub-agent": "orchestrator",
            "model": "orchestrator",  # deepagents library internal model node
            "research-agent": "Research Agent",
            "web-researcher": "Web Researcher",
            "mentor-matcher": "Mentor Matcher",
            "general-purpose": "Assistant",
        }

        if node_name in deep_agent_names:
            return deep_agent_names[node_name]

        # Fall back to parent mapping
        return super()._map_agent_name(node_name)

    def _infer_subagent_name(self, task_description: str, tool_input: dict) -> str:
        """
        Infer subagent name from task description or tool input context.

        Uses keyword matching to determine the type of agent being delegated to.
        Returns a meaningful name or None if cannot be inferred.
        """
        if not task_description:
            return "assistant"

        task_lower = task_description.lower()

        # Check for specific domain keywords
        keyword_to_agent = {
            # Integration/API agents
            ("wrike", "integration", "api", "webhook", "sync"): "integration_expert",
            # Research agents
            ("research", "search", "find", "look up", "investigate"): "research_expert",
            # Web/browsing agents
            ("web", "browse", "scrape", "url", "website"): "web_researcher",
            # Mentor matching
            ("mentor", "match", "pairing", "assign"): "mentor_matcher",
            # Data analysis
            ("analyze", "analysis", "data", "report", "statistics"): "data_analyst",
            # Code/development
            ("code", "program", "develop", "implement", "fix bug"): "developer",
            # Writing/content
            ("write", "draft", "compose", "document", "email"): "writer",
        }

        for keywords, agent_name in keyword_to_agent.items():
            if any(kw in task_lower for kw in keywords):
                return agent_name

        # Check if there's a subagent_type hint in tool_input metadata
        if isinstance(tool_input.get("metadata"), dict):
            agent_hint = tool_input["metadata"].get("agent_type") or tool_input["metadata"].get("subagent")
            if agent_hint:
                return agent_hint

        # Default to assistant for general tasks
        return "assistant"

    def _get_readable_agent_name(self, agent_name: str) -> str:
        """
        Convert internal agent name to a human-readable display name.

        Examples:
        - "web_researcher" -> "Web Researcher"
        - "integration_expert" -> "Integration Expert"
        - "research-agent" -> "Research Agent"
        """
        readable_names = {
            "web_researcher": "Web Researcher",
            "research_expert": "Research Expert",
            "integration_expert": "Integration Expert",
            "mentor_matcher": "Mentor Matcher",
            "data_analyst": "Data Analyst",
            "developer": "Developer",
            "writer": "Writer",
            "assistant": "Assistant",
            "research-agent": "Research Agent",
            "web-researcher": "Web Researcher",
            "mentor-matcher": "Mentor Matcher",
            "integration-agent": "Integration Agent",
            "general-purpose": "General Purpose",
        }

        if agent_name in readable_names:
            return readable_names[agent_name]

        # Convert snake_case or kebab-case to Title Case
        # Replace underscores and hyphens with spaces, then title case
        return agent_name.replace("_", " ").replace("-", " ").title()

    def _summarize_task_description(self, task_description: str, max_length: int = 50) -> str:
        """
        Create a concise summary of a task description for display.

        Extracts the key action/topic from the task description.
        Returns an empty string if the description is empty or unhelpful.
        """
        if not task_description:
            return ""

        # Clean up the description
        desc = task_description.strip()

        # Remove common prefixes that don't add value
        prefixes_to_remove = [
            "please ",
            "can you ",
            "i need you to ",
            "i want you to ",
            "your task is to ",
            "you should ",
        ]
        desc_lower = desc.lower()
        for prefix in prefixes_to_remove:
            if desc_lower.startswith(prefix):
                desc = desc[len(prefix):]
                break

        # Take the first sentence or clause
        for sep in [". ", ".\n", " - ", ": ", "\n"]:
            if sep in desc:
                desc = desc.split(sep)[0]
                break

        # Truncate to max length, breaking at word boundary
        if len(desc) > max_length:
            desc = desc[:max_length]
            # Find last space to avoid cutting words
            last_space = desc.rfind(" ")
            if last_space > max_length // 2:
                desc = desc[:last_space]

        # Capitalize first letter
        if desc:
            desc = desc[0].upper() + desc[1:]

        return desc.strip()
