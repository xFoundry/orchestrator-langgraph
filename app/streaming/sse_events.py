"""SSE Event models for streaming orchestrator responses."""

from __future__ import annotations

import json
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel


class SSEEventType(str, Enum):
    """Types of SSE events emitted by the streaming endpoint."""

    AGENT_ACTIVITY = "agent_activity"
    TEXT_CHUNK = "text_chunk"
    CITATION = "citation"
    TOOL_RESULT = "tool_result"
    THINKING = "thinking"  # For planning/reasoning steps
    TODO = "todo"  # For todo/task list updates
    ARTIFACT = "artifact"
    THREAD_TITLE = "thread_title"  # AI-generated thread title
    COMPLETE = "complete"
    ERROR = "error"
    HEARTBEAT = "heartbeat"


class AgentActivityData(BaseModel):
    """Data for agent_activity events."""

    agent: str
    action: str  # "tool_call", "tool_response", "delegate", "thinking"
    tool_name: Optional[str] = None
    tool_args: Optional[dict[str, Any]] = None
    details: Optional[str] = None
    timestamp: Optional[float] = None
    invocation_id: Optional[str] = None  # Unique ID for parallel subagent tracking


class TextChunkData(BaseModel):
    """Data for text_chunk events."""

    chunk: str
    agent: str
    is_partial: bool = True
    invocation_id: Optional[str] = None  # Unique ID for parallel subagent tracking


class CitationData(BaseModel):
    """Data for citation events."""

    source: str
    content: str
    confidence: float = 1.0


class RichCitationData(BaseModel):
    """Enhanced citation with entity information for user-friendly display.

    This model provides structured citation data that enables:
    - Entity-specific display names (e.g., task title, session date + mentor)
    - Grouping by entity type (tasks, sessions, documents)
    - Rich metadata for tooltips (status, due date, excerpt)
    - Inline citation markers via source_number ([source:1], [source:2], etc.)
    """

    source: str  # Unique ID: "task:rec123", "session:rec456", "doc:chunk789"
    entity_type: str  # "task" | "session" | "team" | "mentor" | "document" | "entity"
    display_name: str  # Human-readable name: "Complete project proposal"
    content: str  # Tooltip content / description
    confidence: float = 1.0
    group_key: str  # For frontend grouping: "tasks", "sessions", "documents"
    source_number: Optional[int] = None  # For inline [source:N] markers
    metadata: Optional[dict[str, Any]] = None  # Additional context: status, due_date, etc.


class ToolResultData(BaseModel):
    """Data for tool_result events."""

    agent: str
    tool_name: str
    result_summary: Optional[str] = None
    success: bool = True
    invocation_id: Optional[str] = None  # Unique ID for parallel subagent tracking


class ThinkingData(BaseModel):
    """Data for thinking/reasoning events."""

    phase: str  # "planning", "reasoning", "action", "final_answer"
    content: str
    agent: str
    timestamp: Optional[float] = None
    invocation_id: Optional[str] = None  # Unique ID for parallel subagent tracking


class TodoItemData(BaseModel):
    """Data for a single todo item."""

    id: str
    content: str
    status: str  # "pending", "in_progress", "completed"
    activeForm: Optional[str] = None  # Present continuous form for display


class TodoData(BaseModel):
    """Data for todo events - contains the full list of todos."""

    todos: list[TodoItemData]
    timestamp: Optional[float] = None


class ArtifactData(BaseModel):
    """Data for artifact events."""

    id: str
    artifact_type: str  # "data_table" | "document" | "chat_block"
    title: str
    summary: Optional[str] = None
    payload: Any
    origin: Optional[dict[str, Any]] = None
    created_at: Optional[float] = None


class ThreadTitleData(BaseModel):
    """Data for thread_title events - AI-generated conversation title."""

    title: str
    thread_id: Optional[str] = None


class CompleteData(BaseModel):
    """Data for complete events."""

    status: str
    thread_id: Optional[str] = None
    full_message: str
    citations: list[RichCitationData] = []  # Using RichCitationData for enhanced display
    agent_trace: list[AgentActivityData] = []
    handoff_summary: Optional[str] = None
    handoff_recent_messages: Optional[list[dict[str, Any]]] = None


class ErrorData(BaseModel):
    """Data for error events."""

    message: str
    code: Optional[str] = None
    recoverable: bool = False


class SSEEvent(BaseModel):
    """Generic SSE event wrapper."""

    event: SSEEventType
    data: Any

    def to_sse_string(self) -> str:
        """Format as SSE string for streaming response."""
        if hasattr(self.data, "model_dump"):
            data_dict = self.data.model_dump()
        elif isinstance(self.data, dict):
            data_dict = self.data
        else:
            data_dict = {"value": self.data}

        data_str = json.dumps(data_dict)
        return f"event: {self.event.value}\ndata: {data_str}\n\n"
