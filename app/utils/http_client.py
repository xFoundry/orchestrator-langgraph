"""
HTTP Client Factory - Stable HTTP/1.1 client for LLM API calls.

This module provides a custom httpx client configured for reliability with
Railway's proxy infrastructure. It addresses the HTTP/2 connection desync
issue that causes "terminated" errors during streaming.

Root Cause:
    The OpenAI SDK uses httpx with HTTP/2 by default. Railway's proxy (Nginx/Envoy)
    aggressively closes idle connections (~300s timeout). When httpx attempts to
    reuse a "zombie" connection, it receives a ConnectionTerminated error.

Solution:
    Disable HTTP/2 to use simpler HTTP/1.1 connections that handle disconnects
    gracefully. Also configure proper connection pool limits for high concurrency.

Reference:
    - https://github.com/encode/httpx/discussions/2112
    - https://github.com/encode/httpx/discussions/2056

Usage:
    from app.utils.http_client import get_http_client

    # For LangChain ChatOpenAI
    client = ChatOpenAI(
        model="...",
        http_client=get_http_client(),
        ...
    )
"""

from __future__ import annotations

import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# Module-level client instance (lazy initialization)
_http_client: Optional[httpx.AsyncClient] = None


def get_connection_limits() -> httpx.Limits:
    """
    Get connection pool limits for high-concurrency LLM workloads.

    Returns:
        httpx.Limits configured for multiple parallel agent streams
    """
    return httpx.Limits(
        max_keepalive_connections=20,  # Small pool of reusable connections
        max_connections=100,           # Support 100+ parallel requests (agents * users)
        keepalive_expiry=30.0,         # Force-close idle connections to prevent zombies
    )


def get_http_client() -> httpx.AsyncClient:
    """
    Get the shared async HTTP client with HTTP/2 disabled.

    This client is configured for stability with Railway's proxy:
    - HTTP/2 disabled to prevent connection desync issues
    - Connection pool sized for high concurrency
    - Short keepalive to prevent zombie connections
    - Long timeout for streaming LLM responses

    Returns:
        Shared httpx.AsyncClient instance
    """
    global _http_client

    if _http_client is None:
        logger.info("Creating HTTP/1.1 client for LLM API calls (HTTP/2 disabled)")
        _http_client = httpx.AsyncClient(
            http2=False,                    # CRITICAL: Disable HTTP/2 to prevent desync
            limits=get_connection_limits(),
            timeout=httpx.Timeout(
                connect=30.0,               # Connection timeout
                read=600.0,                 # 10 min read timeout for long streams
                write=30.0,                 # Write timeout
                pool=30.0,                  # Pool acquisition timeout
            ),
        )
        logger.info(
            f"HTTP client configured: http2=False, max_connections=100, "
            f"keepalive_expiry=30s, read_timeout=600s"
        )

    return _http_client


def get_sync_http_client() -> httpx.Client:
    """
    Get a synchronous HTTP client with the same configuration.

    For use cases that don't support async (e.g., some LangChain sync methods).

    Returns:
        New httpx.Client instance (not shared - sync clients aren't thread-safe)
    """
    return httpx.Client(
        http2=False,
        limits=get_connection_limits(),
        timeout=httpx.Timeout(
            connect=30.0,
            read=600.0,
            write=30.0,
            pool=30.0,
        ),
    )


async def close_http_client() -> None:
    """
    Close the shared HTTP client.

    Call this during application shutdown to cleanly release connections.
    """
    global _http_client

    if _http_client is not None:
        logger.info("Closing shared HTTP client")
        await _http_client.aclose()
        _http_client = None
