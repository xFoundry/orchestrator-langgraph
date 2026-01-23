"""Configuration settings for the orchestrator service."""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"

    # CORS
    cors_origins: str = "http://localhost:3000,http://localhost:8000"

    # LLM Providers
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    openrouter_api_key: Optional[str] = None  # For routing Anthropic models via OpenRouter
    
    # OpenRouter Settings
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    use_openrouter_for_anthropic: bool = True  # Route Anthropic models through OpenRouter

    # Default Models
    default_orchestrator_model: str = "gpt-5.2"
    default_research_model: str = "gpt-5.2-pro"
    model_context_limits: dict[str, int] = Field(
        default_factory=lambda: {
            "gpt-5.2": 20000,
            "gpt-5.2-pro": 50000,
        }
    )

    # External Services
    cognee_api_url: str = "http://localhost:8001"
    cognee_secret_key: Optional[str] = None

    # Mentor Hub API (for live data access)
    mentor_hub_api_url: str = "http://localhost:3001/api"

    # Firecrawl (self-hosted or cloud)
    firecrawl_api_url: str = "https://api.firecrawl.dev"
    firecrawl_api_key: Optional[str] = None

    # Outline MCP (server-to-server)
    outline_mcp_base_url: str = "https://docs.xfoundry.org"
    outline_mcp_api_key: Optional[str] = None
    outline_mcp_protocol_version: str = "2025-06-18"
    outline_mcp_timeout_seconds: float = 30.0

    # Memory (via Cognee integration)
    cognee_memory_enabled: bool = True

    # Redis
    redis_url: str = "redis://localhost:6379"
    # Postgres (LangGraph Store)
    postgres_url: Optional[str] = None

    # Integration encryption
    integration_encryption_key: Optional[str] = None

    # Deep Agent Configuration
    deep_agent_model: Optional[str] = None  # Defaults to default_orchestrator_model
    deep_agent_enable_subagents: bool = True
    deep_agent_enable_filesystem: bool = True
    deep_agent_enable_hitl: bool = False
    deep_agent_memory_ttl_days: int = 30  # How long to retain /memories/ files
    deep_agent_enable_thinking: bool = True  # Enable extended thinking for supported models
    deep_agent_reasoning_effort: str = "high"  # For GPT-5.2: "none", "low", "medium", "high", "xhigh"
    deep_agent_thinking_budget: int = 10000  # Token budget for Claude thinking

    # LangGraph Execution Limits
    recursion_limit: int = 300  # Max graph steps (default 25 is too low for deep agents)

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse CORS origins from comma-separated string."""
        return [origin.strip() for origin in self.cors_origins.split(",")]

    def get_model_context_limit(self, model_name: str) -> int:
        """Return the context window size for a model."""
        return self.model_context_limits.get(model_name, 20000)

    def get_handoff_budgets(self, model_name: str) -> dict[str, int]:
        """Return handoff budget splits for a model."""
        context_limit = self.get_model_context_limit(model_name)
        handoff_budget = max(1, int(context_limit * 0.25))
        summary_budget = max(1, int(handoff_budget * 0.10))
        messages_budget = max(1, handoff_budget - summary_budget)

        return {
            "context_limit": context_limit,
            "handoff_budget": handoff_budget,
            "summary_budget": summary_budget,
            "messages_budget": messages_budget,
        }


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
