"""
Deep Agent Event Mapper - Extended SSE event mapping for Deep Agents.

Extends the base LangGraphEventMapper to handle Deep Agent-specific events:
- write_todos: Planning/task decomposition -> thinking events
- task: Subagent delegation -> agent_activity events
- Filesystem operations: Context management (filtered for internal ops)
"""

from __future__ import annotations

import logging
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
        self._active_subagents: set[str] = set()
        self._artifact_writes: dict[str, dict[str, str]] = {}
        self._artifact_write_queue: list[dict[str, str]] = []

    def map_event(self, event: dict[str, Any]) -> list[SSEEvent]:
        """
        Map event with Deep Agent-specific handling.

        Intercepts Deep Agent tool calls to provide better user feedback,
        then delegates to parent for standard events.
        """
        sse_events: list[SSEEvent] = []
        kind = event.get("event")
        timestamp = time.time()

        # Track current agent from metadata
        metadata = event.get("metadata", {})
        node_name = metadata.get("langgraph_node", "")
        agent_name = metadata.get("lc_agent_name", "")

        if agent_name:
            self._current_agent = agent_name
        elif node_name:
            self._current_agent = self._map_agent_name(node_name)

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
                sse_events.append(
                    SSEEvent(
                        event=SSEEventType.TODO,
                        data=TodoData(
                            todos=todo_items,
                            timestamp=timestamp,
                        ),
                    )
                )
                return sse_events

            # -----------------------------------------------------------------
            # task: Emit subagent delegation event
            # -----------------------------------------------------------------
            if tool_name == "task":
                subagent_name = tool_input.get("agent") or tool_input.get("name", "subagent")
                task_description = tool_input.get("task") or tool_input.get("prompt", "")

                self._active_subagents.add(subagent_name)

                # Truncate task description for display
                display_task = task_description[:100]
                if len(task_description) > 100:
                    display_task += "..."

                sse_events.append(
                    SSEEvent(
                        event=SSEEventType.AGENT_ACTIVITY,
                        data=AgentActivityData(
                            agent=self._current_agent,
                            action="delegate",
                            tool_name=subagent_name,
                            details=f"Delegating: {display_task}",
                            timestamp=timestamp,
                        ),
                    )
                )
                return sse_events

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
                sse_events.append(
                    SSEEvent(
                        event=SSEEventType.AGENT_ACTIVITY,
                        data=AgentActivityData(
                            agent=self._current_agent,
                            action=action_type,
                            tool_name=tool_name,
                            details=f"{action_desc}: {path}" if path else action_desc,
                            timestamp=timestamp,
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

                    # Emit artifact event
                    sse_events.append(
                        SSEEvent(
                            event=SSEEventType.ARTIFACT,
                            data=ArtifactData(
                                id=path,
                                artifact_type="file",
                                title=title,
                                summary=path,
                                payload={"path": path, "content": content, "action": action, "saved": is_saved},
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
                                details=action_desc,
                                timestamp=timestamp,
                            ),
                        )
                    )
                    return sse_events

            # task completion: Subagent returned
            if tool_name == "task":
                # Find which subagent completed
                # The output contains the result from the subagent
                output = event.get("data", {}).get("output", "")

                if self._active_subagents:
                    subagent = self._active_subagents.pop()
                    sse_events.append(
                        SSEEvent(
                            event=SSEEventType.AGENT_ACTIVITY,
                            data=AgentActivityData(
                                agent=subagent,
                                action="complete",
                                tool_name="task",
                                details="Subagent completed task",
                                timestamp=timestamp,
                            ),
                        )
                    )
                return sse_events

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
        return super().map_event(event)

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

    def _map_agent_name(self, node_name: str) -> str:
        """Map internal node names to user-friendly names."""
        # First check Deep Agent specific names
        deep_agent_names = {
            "mentor-hub-agent": "orchestrator",
            "research-agent": "Research Agent",
            "web-researcher": "Web Researcher",
            "mentor-matcher": "Mentor Matcher",
            "general-purpose": "Assistant",
        }

        if node_name in deep_agent_names:
            return deep_agent_names[node_name]

        # Fall back to parent mapping
        return super()._map_agent_name(node_name)
