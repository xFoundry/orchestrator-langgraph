"""API dependencies for authentication and authorization."""

from __future__ import annotations

import secrets
from typing import Annotated

from fastapi import Depends, Header, HTTPException, status

from app.config import get_settings


async def verify_api_key(
    x_api_key: Annotated[str | None, Header()] = None,
) -> None:
    """Verify the API key from the X-API-Key header.

    If api_key is not configured in settings, authentication is disabled.
    This allows easy development while requiring keys in production.
    """
    settings = get_settings()

    # If no API key is configured, skip authentication (development mode)
    if not settings.api_key:
        return

    # API key is required but not provided
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # Use secrets.compare_digest to prevent timing attacks
    if not secrets.compare_digest(x_api_key, settings.api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )


# Dependency to use in routes
RequireApiKey = Annotated[None, Depends(verify_api_key)]
