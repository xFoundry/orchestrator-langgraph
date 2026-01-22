"""Request and response models for chat endpoints."""

from __future__ import annotations

from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator


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


# =============================================================================
# MULTIMODAL CONTENT BLOCKS (LangChain Standard Format)
# =============================================================================

class TextContentBlock(BaseModel):
    """Text content block."""
    type: Literal["text"] = "text"
    text: str = Field(..., description="The text content")


class ImageContentBlock(BaseModel):
    """
    Image content block supporting multiple input methods.
    
    Supports:
    - base64: Base64-encoded image data with mime_type
    - url: Direct URL to the image
    - data: Alias for base64 (frontend compatibility)
    
    Models supported: GPT-4o+, GPT-5+, Claude 3+, Gemini Pro Vision
    """
    type: Literal["image"] = "image"
    # Base64 encoded data (LangChain standard)
    base64: Optional[str] = Field(default=None, description="Base64-encoded image data")
    # Alternative: data field (frontend compatibility with agent-ui)
    data: Optional[str] = Field(default=None, description="Base64-encoded image data (alias)")
    # URL to image
    url: Optional[str] = Field(default=None, description="URL to the image")
    # MIME type (required for base64)
    mime_type: Optional[str] = Field(
        default=None, 
        alias="mimeType",
        description="MIME type (e.g., 'image/jpeg', 'image/png', 'image/gif', 'image/webp')"
    )
    # Optional metadata
    metadata: Optional[dict[str, Any]] = Field(default=None, description="Additional metadata")
    
    model_config = {"populate_by_name": True}  # Allow both mime_type and mimeType
    
    @model_validator(mode="after")
    def normalize_data_field(self):
        """Normalize 'data' field to 'base64' for consistency."""
        if self.data and not self.base64:
            self.base64 = self.data
        return self


class FileContentBlock(BaseModel):
    """
    File content block for documents (PDFs, etc.).
    
    Supports:
    - base64: Base64-encoded file data with mime_type
    - url: Direct URL to the file
    - data: Alias for base64 (frontend compatibility)
    - file_id: Provider-managed file ID
    
    Models supported: Claude 3+, Gemini (PDFs)
    Note: GPT models don't natively support PDF - may need preprocessing
    """
    type: Literal["file"] = "file"
    # Base64 encoded data (LangChain standard)
    base64: Optional[str] = Field(default=None, description="Base64-encoded file data")
    # Alternative: data field (frontend compatibility with agent-ui)
    data: Optional[str] = Field(default=None, description="Base64-encoded file data (alias)")
    # URL to file
    url: Optional[str] = Field(default=None, description="URL to the file")
    # Provider-managed file ID
    file_id: Optional[str] = Field(default=None, description="Provider-managed file ID")
    # MIME type (required for base64)
    mime_type: Optional[str] = Field(
        default=None,
        alias="mimeType", 
        description="MIME type (e.g., 'application/pdf')"
    )
    # Optional metadata
    metadata: Optional[dict[str, Any]] = Field(default=None, description="Additional metadata")
    
    model_config = {"populate_by_name": True}
    
    @model_validator(mode="after")
    def normalize_data_field(self):
        """Normalize 'data' field to 'base64' for consistency."""
        if self.data and not self.base64:
            self.base64 = self.data
        return self


# Union type for all content blocks
ContentBlock = Union[TextContentBlock, ImageContentBlock, FileContentBlock, dict[str, Any]]


class ChatRequest(BaseModel):
    """Request model for chat endpoint with multimodal support."""

    # Legacy: text-only message (still supported for backward compatibility)
    message: Optional[str] = Field(default=None, description="User's text message (legacy)")
    
    # Multimodal: list of content blocks (images, files, text)
    content: Optional[list[ContentBlock]] = Field(
        default=None,
        description="Multimodal content blocks (images, files, text). Takes precedence over 'message'."
    )
    
    @model_validator(mode="after")
    def validate_message_or_content(self):
        """Ensure at least one of message or content is provided."""
        if not self.message and not self.content:
            raise ValueError("Either 'message' or 'content' must be provided")
        return self
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
    use_deep_agent: bool = Field(
        default=False,
        description="Use Deep Agent orchestrator with planning, subagents, and context management (takes precedence over v3/v2)",
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
    model_override: Optional[str] = Field(
        default=None,
        description="Model to use for this request (e.g., 'gpt-5.2', 'claude-opus-4.5')",
    )
    selected_tools: Optional[list[str]] = Field(
        default=None,
        description="User-selected tools to prioritize (e.g., ['firecrawl'])",
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
