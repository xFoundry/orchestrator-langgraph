"""Request and response models for chat endpoints."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class UserContext(BaseModel):
    """User context for personalization."""

    name: Optional[str] = Field(default=None, description="User's display name")
    email: Optional[str] = Field(default=None, description="User's email address")
    role: Optional[str] = Field(
        default=None, description="User's role: Staff, Mentor, or Participant"
    )
    teams: Optional[list[str]] = Field(
        default=None, description="Teams the user is associated with"
    )
    cohort: Optional[str] = Field(default=None, description="User's current cohort")
    auth0_id: Optional[str] = Field(
        default=None, description="User's Auth0 ID for per-user memory isolation"
    )


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""

    message: str = Field(..., description="User's message")
    tenant_id: str = Field(default="default", description="Tenant ID for multi-tenant isolation")
    user_id: Optional[str] = Field(default=None, description="User ID for personalization")
    user_context: Optional[UserContext] = Field(
        default=None, description="User details for personalization"
    )
    session_id: Optional[str] = Field(
        default=None, description="Session ID for conversation context"
    )
    thread_id: Optional[str] = Field(
        default=None, description="Thread ID for conversation continuity"
    )
    group_ids: Optional[list[str]] = Field(
        default=None, description="Group IDs (cohort, team) for scoping"
    )
    use_memory: bool = Field(
        default=False,
        description="Force the memory agent to be called for knowledge retrieval",
    )
    canvas_id: Optional[str] = Field(
        default=None, description="Canvas ID for multi-node chat sessions"
    )
    chat_block_id: Optional[str] = Field(
        default=None, description="Chat block ID within a canvas"
    )
    auto_artifacts: bool = Field(
        default=False, description="Enable automatic artifact creation for this chat"
    )
    context_artifacts: Optional[list[dict[str, Any]]] = Field(
        default=None,
        description="Artifacts to include as context for this chat block",
    )
    use_v3: bool = Field(
        default=True,
        description="Use v3 orchestrator with supervisor pattern (parallel workers, evaluator, retry)",
    )
    use_v2: bool = Field(
        default=False,
        description="Use v2 orchestrator (fallback if use_v3=False)",
    )
    model_config_override: Optional[dict[str, str]] = Field(
        default=None,
        description="Optional model overrides (orchestrator_model, memory_model, etc.)",
    )


class Citation(BaseModel):
    """Citation model for tracking sources."""

    source: str
    content: str
    confidence: float = 1.0


class AgentActivity(BaseModel):
    """Model for tracking agent activity."""

    agent_name: str
    action: str
    details: Optional[str] = None
    timestamp: Optional[float] = None


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""

    message: str = Field(..., description="Response from the orchestrator")
    citations: list[Citation] = Field(
        default_factory=list, description="Sources used in the response"
    )
    agent_trace: list[AgentActivity] = Field(
        default_factory=list, description="Trace of agent activity"
    )
    thread_id: Optional[str] = Field(
        default=None, description="Thread ID for follow-up messages"
    )
    status: str = Field(default="success", description="Response status")
