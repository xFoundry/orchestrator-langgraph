"""
User Integration MCP Tools - Dynamic tool loading using langchain-mcp-adapters.

Loads tools from MCP servers based on user's configured integrations.
Supports both HTTP/SSE and stdio transports with per-user authentication.

Documentation:
- https://docs.langchain.com/oss/python/langchain/mcp
- https://reference.langchain.com/python/langchain_mcp_adapters/
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from app.tools.sanitize import SanitizationInterceptor

logger = logging.getLogger(__name__)


# Integration type specifications matching frontend config
INTEGRATION_SPECS = {
    # SSE Integrations
    "wrike": {
        "transport": "sse",
        "url_template": "https://{host}/app/mcp/sse",
        "hosts": {
            "us": "www.wrike.com",
            "eu": "app-eu.wrike.com",
        },
        "auth_header": "Authorization",
        "credential_key": "token",
        "auth_format": "Bearer {token}",
    },

    # stdio Integrations
    "linear": {
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@linear/mcp-server"],
        "env_mapping": {
            "apiKey": "LINEAR_API_KEY",
        },
    },
    "github": {
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-github"],
        "env_mapping": {
            "token": "GITHUB_PERSONAL_ACCESS_TOKEN",
        },
    },
    "slack": {
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-slack"],
        "env_mapping": {
            "botToken": "SLACK_BOT_TOKEN",
            "teamId": "SLACK_TEAM_ID",
        },
    },
    "notion": {
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@notionhq/client"],
        "env_mapping": {
            "token": "NOTION_TOKEN",
        },
    },
    # Granola uses existing Claude Code MCP integration - no server needed
}


def _build_http_connection_config(integration_type: str, credentials: dict, config: dict) -> dict:
    """Build HTTP/SSE connection configuration for an integration."""
    spec = INTEGRATION_SPECS.get(integration_type)
    transport = spec.get("transport") if spec else None
    if not spec or transport not in ("http", "sse", "streamable_http"):
        return {}

    # Build URL
    if "url_template" in spec:
        region = config.get("region", "us")
        host = spec.get("hosts", {}).get(region, list(spec.get("hosts", {}).values())[0])
        url = spec["url_template"].format(host=host)
    else:
        url = spec.get("url", "")

    # Build headers with auth
    headers = {}
    if "auth_header" in spec and "credential_key" in spec:
        cred_key = spec["credential_key"]
        if cred_key in credentials:
            # Use auth_format if provided, otherwise use token as-is
            if "auth_format" in spec:
                auth_value = spec["auth_format"].format(**{cred_key: credentials[cred_key]})
            else:
                auth_value = credentials[cred_key]
            headers[spec["auth_header"]] = auth_value

    return {
        "transport": transport,
        "url": url,
        "headers": headers,
        "timeout": config.get("timeout", 30.0),
    }


def _build_stdio_connection_config(integration_type: str, credentials: dict, config: dict) -> dict:
    """Build stdio connection configuration for an integration."""
    spec = INTEGRATION_SPECS.get(integration_type)
    if not spec or spec.get("transport") != "stdio":
        return {}

    # Build environment variables
    env = {}
    env_mapping = spec.get("env_mapping", {})
    for cred_key, env_key in env_mapping.items():
        if cred_key in credentials:
            env[env_key] = credentials[cred_key]

    # Build args (handle templates if needed)
    if "args_template" in spec:
        args = [
            arg.format(**credentials, **config)
            for arg in spec["args_template"]
        ]
    else:
        args = spec.get("args", [])

    return {
        "transport": "stdio",
        "command": spec.get("command", ""),
        "args": args,
        "env": env,
    }


async def _load_http_tools(integrations: list[dict]) -> list[BaseTool]:
    """
    Load tools from HTTP/SSE MCP servers.

    Uses a single MultiServerMCPClient for all HTTP integrations.
    """
    servers = {}

    for integration in integrations:
        integration_type = integration["type"]
        credentials = integration["credentials"]
        config = integration.get("config", {})

        connection_config = _build_http_connection_config(
            integration_type, credentials, config
        )

        if connection_config:
            servers[integration_type] = connection_config

    if not servers:
        return []

    try:
        # Use interceptor to sanitize outputs at the source for Redis JSON compatibility
        client = MultiServerMCPClient(
            servers,
            tool_name_prefix=True,  # Prefix tool names with server name
            tool_interceptors=[SanitizationInterceptor()],
        )

        tools = await client.get_tools()

        logger.info(f"Loaded {len(tools)} tools from {len(servers)} HTTP MCP server(s)")
        return tools

    except Exception as e:
        logger.error(f"Failed to load HTTP MCP tools: {e}", exc_info=True)
        return []


async def _load_stdio_tools(user_id: str, integrations: list[dict]) -> list[BaseTool]:
    """
    Load tools from stdio MCP servers.

    Creates separate client instances per integration type since stdio
    requires different environment variables for each server.
    """
    all_tools = []

    for integration in integrations:
        integration_type = integration["type"]
        credentials = integration["credentials"]
        config = integration.get("config", {})

        connection_config = _build_stdio_connection_config(
            integration_type, credentials, config
        )

        if not connection_config:
            continue

        try:
            # Create separate client for this integration
            # Use interceptor to sanitize outputs at the source for Redis JSON compatibility
            client = MultiServerMCPClient(
                {integration_type: connection_config},
                tool_name_prefix=True,
                tool_interceptors=[SanitizationInterceptor()],
            )

            tools = await client.get_tools()

            logger.info(
                f"Loaded {len(tools)} tools from {integration_type} stdio MCP server"
            )
            all_tools.extend(tools)

        except Exception as e:
            logger.error(
                f"Failed to load {integration_type} stdio MCP tools: {e}",
                exc_info=True,
            )
            continue

    return all_tools


async def get_user_integration_tools(user_id: str) -> list[BaseTool]:
    """
    Get all MCP tools from user's enabled integrations.

    Queries the database for enabled integrations, decrypts credentials,
    and loads tools from both HTTP and stdio MCP servers.

    Args:
        user_id: Auth0 user ID

    Returns:
        List of LangChain tools from all enabled integrations
    """
    try:
        # Import here to avoid circular dependencies
        from app.db.session import get_db
        from app.services.integration_service import (
            decrypt_credentials,
            get_user_integrations,
        )

        # Get database session
        async for db in get_db():
            try:
                # Fetch user's enabled integrations
                integrations = await get_user_integrations(db, user_id, enabled_only=True)

                if not integrations:
                    logger.info(f"No enabled integrations for user {user_id}")
                    return []

                logger.info(
                    f"Found {len(integrations)} enabled integration(s) for user {user_id}"
                )

                # Prepare integration data with decrypted credentials
                http_integrations = []
                stdio_integrations = []

                for integration in integrations:
                    try:
                        # Decrypt credentials
                        credentials = decrypt_credentials(
                            integration.credentials_encrypted
                        )

                        integration_data = {
                            "type": integration.integration_type,
                            "credentials": credentials,
                            "config": integration.config or {},
                        }

                        # Separate by transport type
                        spec = INTEGRATION_SPECS.get(integration.integration_type)
                        if not spec:
                            logger.warning(
                                f"No spec found for integration type: {integration.integration_type}"
                            )
                            continue

                        if spec.get("transport") in ("http", "sse", "streamable_http"):
                            http_integrations.append(integration_data)
                        elif spec.get("transport") == "stdio":
                            stdio_integrations.append(integration_data)
                        else:
                            logger.debug(
                                f"Skipping {integration.integration_type} - no MCP server needed"
                            )

                    except Exception as e:
                        logger.error(
                            f"Failed to process {integration.integration_type} integration: {e}"
                        )
                        continue

                # Load tools from both transport types
                all_tools = []

                if http_integrations:
                    http_tools = await _load_http_tools(http_integrations)
                    all_tools.extend(http_tools)

                if stdio_integrations:
                    stdio_tools = await _load_stdio_tools(user_id, stdio_integrations)
                    all_tools.extend(stdio_tools)

                logger.info(
                    f"Total MCP tools loaded for user {user_id}: {len(all_tools)}"
                )
                return all_tools

            finally:
                await db.close()

    except Exception as e:
        logger.error(f"Failed to load user integration tools: {e}", exc_info=True)
        return []


async def validate_integration(
    integration_type: str,
    credentials: dict,
    config: dict | None = None,
) -> tuple[bool, str, list[dict]]:
    """
    Validate an integration by attempting to connect and load tools.

    Args:
        integration_type: Type of integration (e.g., "wrike", "linear")
        credentials: Integration credentials
        config: Optional configuration

    Returns:
        Tuple of (success: bool, message: str, tools: list[dict])
        Each tool dict has 'name' and 'description' keys.
    """
    if integration_type not in INTEGRATION_SPECS:
        return False, f"Unknown integration type: {integration_type}", []

    spec = INTEGRATION_SPECS[integration_type]
    config = config or {}

    try:
        # Build connection config
        if spec.get("transport") in ("http", "sse", "streamable_http"):
            connection_config = _build_http_connection_config(
                integration_type, credentials, config
            )
        elif spec.get("transport") == "stdio":
            connection_config = _build_stdio_connection_config(
                integration_type, credentials, config
            )
        else:
            return True, "Integration configured (no MCP server validation needed)", []

        if not connection_config:
            return False, "Failed to build connection configuration", []

        # Create temporary client
        client = MultiServerMCPClient({
            "test": connection_config
        })

        # Try to load tools (validates connection)
        tools = await client.get_tools()

        if not tools:
            return False, "Connected but no tools found", []

        # Extract tool info for response
        tool_info = []
        for tool in tools:
            tool_info.append({
                "name": getattr(tool, "name", str(tool)),
                "description": getattr(tool, "description", None),
            })

        return True, f"Connected successfully. Found {len(tools)} tools.", tool_info

    except BaseException as e:
        # Extract actual error from ExceptionGroup if present
        actual_error = e
        if isinstance(e, BaseExceptionGroup):
            # Get the first nested exception
            for exc in e.exceptions:
                actual_error = exc
                break

        error_str = str(actual_error).lower()

        if "401" in error_str or "unauthorized" in error_str:
            return False, "Invalid credentials. Please check your API token.", []
        elif "403" in error_str or "forbidden" in error_str:
            return False, "Access denied. Check your API token permissions.", []
        elif "404" in error_str or "not found" in error_str:
            return False, "MCP server endpoint not found.", []
        elif "timeout" in error_str or "timed out" in error_str:
            return False, "Connection timed out. Please try again.", []
        elif "connection" in error_str:
            return False, "Could not connect to the server. Check your network.", []
        else:
            logger.error(f"Integration validation failed: {actual_error}", exc_info=True)
            return False, f"Validation failed: {str(actual_error)[:100]}", []
