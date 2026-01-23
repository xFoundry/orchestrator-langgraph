"""
Outline MCP client (Streamable HTTP JSON-RPC).

Handles initialize handshake and session header reuse for tools/list and tools/call.
Includes retry logic for transient server errors (502, 503, 504).
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Optional

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAYS = [1.0, 2.0, 4.0]  # Exponential backoff delays in seconds
RETRYABLE_STATUS_CODES = {502, 503, 504}  # Gateway errors that may be transient


class OutlineMcpClient:
    """Client for Outline MCP (Streamable HTTP JSON-RPC)."""

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str],
        protocol_version: str,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.endpoint = f"{self.base_url}/api/mcp"
        self.api_key = api_key
        self.protocol_version = protocol_version
        self.timeout_seconds = timeout_seconds
        self._session_id: Optional[str] = None
        self._lock = asyncio.Lock()
        self._request_id = 1

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _auth_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _session_headers(self) -> dict[str, str]:
        headers = self._auth_headers()
        if self._session_id:
            headers["MCP-Protocol-Version"] = self.protocol_version
            headers["Mcp-Session-Id"] = self._session_id
        return headers

    async def _initialize(self, client: httpx.AsyncClient) -> None:
        payload = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "initialize",
            "params": {
                "protocolVersion": self.protocol_version,
                "capabilities": {},
                "clientInfo": {"name": "mentor-hub-orchestrator", "version": "1.0.0"},
            },
        }
        resp = await client.post(
            self.endpoint,
            content=json.dumps(payload),
            headers=self._auth_headers(),
        )
        resp.raise_for_status()

        session_id = resp.headers.get("Mcp-Session-Id")
        protocol_version = resp.headers.get("MCP-Protocol-Version") or self.protocol_version
        if not session_id:
            raise RuntimeError("Outline MCP initialize did not return Mcp-Session-Id")

        self._session_id = session_id
        self.protocol_version = protocol_version

    async def _ensure_session(self, client: httpx.AsyncClient) -> None:
        if self._session_id:
            return
        async with self._lock:
            if not self._session_id:
                await self._initialize(client)

    async def _post(self, method: str, params: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            await self._ensure_session(client)
            payload = {
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": method,
            }
            if params is not None:
                payload["params"] = params

            last_error: Optional[Exception] = None

            for attempt in range(MAX_RETRIES):
                try:
                    resp = await client.post(
                        self.endpoint,
                        content=json.dumps(payload),
                        headers=self._session_headers(),
                    )

                    # Handle expired session
                    if resp.status_code in (400, 404):
                        logger.info("Outline MCP session invalid; reinitializing")
                        self._session_id = None
                        await self._initialize(client)
                        resp = await client.post(
                            self.endpoint,
                            content=json.dumps(payload),
                            headers=self._session_headers(),
                        )

                    # Retry on transient server errors
                    if resp.status_code in RETRYABLE_STATUS_CODES:
                        delay = RETRY_DELAYS[attempt] if attempt < len(RETRY_DELAYS) else RETRY_DELAYS[-1]
                        logger.warning(
                            f"Outline MCP returned {resp.status_code}, retrying in {delay}s "
                            f"(attempt {attempt + 1}/{MAX_RETRIES})"
                        )
                        await asyncio.sleep(delay)
                        continue

                    resp.raise_for_status()
                    data = resp.json()
                    if isinstance(data, dict) and data.get("error"):
                        return {"error": data["error"]}
                    return data.get("result", data)

                except httpx.HTTPStatusError as e:
                    if e.response.status_code in RETRYABLE_STATUS_CODES and attempt < MAX_RETRIES - 1:
                        delay = RETRY_DELAYS[attempt] if attempt < len(RETRY_DELAYS) else RETRY_DELAYS[-1]
                        logger.warning(
                            f"Outline MCP error {e.response.status_code}, retrying in {delay}s "
                            f"(attempt {attempt + 1}/{MAX_RETRIES})"
                        )
                        await asyncio.sleep(delay)
                        last_error = e
                        continue
                    raise
                except (httpx.ConnectError, httpx.TimeoutException) as e:
                    if attempt < MAX_RETRIES - 1:
                        delay = RETRY_DELAYS[attempt] if attempt < len(RETRY_DELAYS) else RETRY_DELAYS[-1]
                        logger.warning(
                            f"Outline MCP connection error, retrying in {delay}s "
                            f"(attempt {attempt + 1}/{MAX_RETRIES}): {e}"
                        )
                        await asyncio.sleep(delay)
                        last_error = e
                        continue
                    raise

            # If we exhausted retries, raise the last error
            if last_error:
                raise last_error
            raise RuntimeError("Outline MCP request failed after max retries")

    async def list_tools(self) -> dict[str, Any]:
        return await self._post("tools/list")

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return await self._post(
            "tools/call",
            {"name": name, "arguments": arguments},
        )


_client: Optional[OutlineMcpClient] = None


def get_outline_mcp_client() -> OutlineMcpClient:
    global _client
    if _client is None:
        settings = get_settings()
        _client = OutlineMcpClient(
            base_url=settings.outline_mcp_base_url,
            api_key=settings.outline_mcp_api_key,
            protocol_version=settings.outline_mcp_protocol_version,
            timeout_seconds=settings.outline_mcp_timeout_seconds,
        )
    return _client

