"""Integration type definitions and configurations."""

from __future__ import annotations

from typing import Any, TypedDict


class ConfigField(TypedDict):
    """Configuration field definition."""

    name: str
    label: str
    type: str  # text, password, url
    placeholder: str
    required: bool
    help_text: str


class MCPConfig(TypedDict, total=False):
    """MCP server configuration."""

    command: str
    args: list[str]
    env_template: dict[str, str]
    server: str


class IntegrationType(TypedDict):
    """Integration type definition."""

    name: str
    description: str
    icon: str
    category: str
    config_fields: list[ConfigField]
    mcp_config: MCPConfig | None
    coming_soon: bool  # If True, integration is not yet available


# Available integration types
INTEGRATION_TYPES: dict[str, IntegrationType] = {
    "wrike": {
        "name": "Wrike",
        "description": "Project management and collaboration platform",
        "icon": "/wrike.com.svg",
        "category": "project-management",
        "coming_soon": False,
        "config_fields": [
            {
                "name": "token",
                "label": "API Token",
                "type": "password",
                "placeholder": "eyJ0dCI6InAiLCJhbGc...",
                "required": True,
                "help_text": "Generate from Account Settings > Apps & Integrations > API (just the token, no 'Bearer' prefix needed)",
            }
        ],
        "mcp_config": {
            "command": "npx",
            "args": ["-y", "mcp-remote", "https://www.wrike.com/app/mcp/sse"],
            "env_template": {"Authorization": "{token}"},
        },
    },
    "linear": {
        "name": "Linear",
        "description": "Issue tracking and project management",
        "icon": "/linear.svg",
        "category": "project-management",
        "coming_soon": True,
        "config_fields": [
            {
                "name": "apiKey",
                "label": "API Key",
                "type": "password",
                "placeholder": "lin_api_...",
                "required": True,
                "help_text": "From Settings > Account > API",
            }
        ],
        "mcp_config": {
            "server": "linear-server",
            "args": ["PERSONAL_API_KEY={apiKey}"],
        },
    },
    "granola": {
        "name": "Granola",
        "description": "Meeting notes and transcription",
        "icon": "/granola.svg",
        "category": "communication",
        "coming_soon": True,
        "config_fields": [
            {
                "name": "apiKey",
                "label": "API Key",
                "type": "password",
                "placeholder": "gra_...",
                "required": True,
                "help_text": "From Granola Settings > Integrations",
            }
        ],
        "mcp_config": {
            "server": "granola",
            "args": ["API_KEY={apiKey}"],
        },
    },
    "github": {
        "name": "GitHub",
        "description": "Code hosting and version control",
        "icon": "/github.svg",
        "category": "project-management",
        "coming_soon": True,
        "config_fields": [
            {
                "name": "token",
                "label": "Personal Access Token",
                "type": "password",
                "placeholder": "ghp_...",
                "required": True,
                "help_text": "Generate from Settings > Developer settings > Personal access tokens",
            }
        ],
        "mcp_config": None,  # Will use gh CLI with token
    },
    "notion": {
        "name": "Notion",
        "description": "Collaborative workspace and notes",
        "icon": "/notion.svg",
        "category": "communication",
        "coming_soon": True,
        "config_fields": [
            {
                "name": "apiKey",
                "label": "Integration Token",
                "type": "password",
                "placeholder": "secret_...",
                "required": True,
                "help_text": "Create an integration at https://www.notion.so/my-integrations",
            }
        ],
        "mcp_config": {
            "server": "notion",
            "args": ["API_KEY={apiKey}"],
        },
    },
    "slack": {
        "name": "Slack",
        "description": "Team communication and collaboration",
        "icon": "/slack.svg",
        "category": "communication",
        "coming_soon": True,
        "config_fields": [
            {
                "name": "token",
                "label": "Bot Token",
                "type": "password",
                "placeholder": "xoxb-...",
                "required": True,
                "help_text": "From Your Apps > OAuth & Permissions > Bot User OAuth Token",
            }
        ],
        "mcp_config": {
            "server": "slack",
            "args": ["TOKEN={token}"],
        },
    },
}


def get_integration_types() -> list[dict[str, Any]]:
    """Get all integration types with their IDs."""
    return [{"id": type_id, **type_def} for type_id, type_def in INTEGRATION_TYPES.items()]


def get_integration_type(type_id: str) -> IntegrationType | None:
    """Get a specific integration type by ID."""
    return INTEGRATION_TYPES.get(type_id)
