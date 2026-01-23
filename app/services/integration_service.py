"""Integration service with encryption for credentials."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from cryptography.fernet import Fernet
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_integration_type, get_settings
from app.db.models import Integration, IntegrationLog

logger = logging.getLogger(__name__)

# Get encryption key from settings (loaded from .env via Pydantic)
settings = get_settings()
ENCRYPTION_KEY = settings.integration_encryption_key
if not ENCRYPTION_KEY:
    logger.warning("INTEGRATION_ENCRYPTION_KEY not set - generating temporary key")
    ENCRYPTION_KEY = Fernet.generate_key().decode()

cipher_suite = Fernet(ENCRYPTION_KEY.encode() if isinstance(ENCRYPTION_KEY, str) else ENCRYPTION_KEY)


def encrypt_credentials(credentials: dict[str, Any]) -> str:
    """Encrypt credentials dictionary."""
    json_str = json.dumps(credentials)
    encrypted_bytes = cipher_suite.encrypt(json_str.encode())
    return encrypted_bytes.decode()


def decrypt_credentials(encrypted_str: str) -> dict[str, Any]:
    """Decrypt credentials string."""
    try:
        decrypted_bytes = cipher_suite.decrypt(encrypted_str.encode())
        return json.loads(decrypted_bytes.decode())
    except Exception as e:
        logger.error(f"Failed to decrypt credentials: {e}")
        return {}


async def create_integration(
    db: AsyncSession,
    user_id: str,
    integration_type: str,
    credentials: dict[str, Any],
    enabled: bool = True,
) -> Integration:
    """Create a new integration for a user."""
    # Check if integration type exists
    type_def = get_integration_type(integration_type)
    if not type_def:
        raise ValueError(f"Unknown integration type: {integration_type}")

    # Check if user already has this integration
    stmt = select(Integration).where(
        Integration.user_id == user_id,
        Integration.integration_type == integration_type,
    )
    result = await db.execute(stmt)
    existing = result.scalar_one_or_none()

    if existing:
        raise ValueError(f"Integration {integration_type} already exists for this user")

    # Encrypt credentials
    encrypted_creds = encrypt_credentials(credentials)

    # Create integration
    integration = Integration(
        user_id=user_id,
        integration_type=integration_type,
        name=type_def["name"],
        status="connected",
        enabled=enabled,
        credentials_encrypted=encrypted_creds,
        last_sync_at=datetime.utcnow(),
    )

    db.add(integration)
    await db.flush()

    # Log the action
    log = IntegrationLog(
        integration_id=integration.id,
        action="connected",
        details={"integration_type": integration_type},
    )
    db.add(log)

    await db.commit()
    await db.refresh(integration)

    logger.info(f"Created integration {integration_type} for user {user_id}")
    return integration


async def get_user_integrations(
    db: AsyncSession,
    user_id: str,
    enabled_only: bool = False,
) -> list[Integration]:
    """Get all integrations for a user."""
    stmt = select(Integration).where(Integration.user_id == user_id)

    if enabled_only:
        stmt = stmt.where(Integration.enabled == True)

    result = await db.execute(stmt)
    return list(result.scalars().all())


async def get_integration(
    db: AsyncSession,
    integration_id: str,
    user_id: str,
) -> Integration | None:
    """Get a specific integration by ID (with user ownership check)."""
    import uuid as uuid_module

    # Validate UUID format
    try:
        uuid_obj = uuid_module.UUID(integration_id)
    except (ValueError, TypeError) as e:
        logger.warning(f"Invalid UUID format for integration_id: {integration_id}")
        return None

    stmt = select(Integration).where(
        Integration.id == uuid_obj,
        Integration.user_id == user_id,
    )
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def update_integration(
    db: AsyncSession,
    integration_id: str,
    user_id: str,
    credentials: dict[str, Any] | None = None,
    enabled: bool | None = None,
) -> Integration:
    """Update an integration."""
    integration = await get_integration(db, integration_id, user_id)
    if not integration:
        raise ValueError("Integration not found")

    credentials_updated = False
    # Only update credentials if non-empty dict is provided
    if credentials is not None and len(credentials) > 0:
        integration.credentials_encrypted = encrypt_credentials(credentials)
        integration.status = "connected"
        integration.error_message = None
        credentials_updated = True

    if enabled is not None:
        integration.enabled = enabled

    integration.updated_at = datetime.utcnow()
    if credentials_updated:
        integration.last_sync_at = datetime.utcnow()

    # Log the action
    log = IntegrationLog(
        integration_id=integration.id,
        action="updated",
        details={"enabled": enabled, "credentials_updated": credentials_updated},
    )
    db.add(log)

    await db.commit()
    await db.refresh(integration)

    logger.info(f"Updated integration {integration_id} for user {user_id}")
    return integration


async def delete_integration(
    db: AsyncSession,
    integration_id: str,
    user_id: str,
) -> None:
    """Delete an integration."""
    integration = await get_integration(db, integration_id, user_id)
    if not integration:
        raise ValueError("Integration not found")

    # Log the action before deleting
    log = IntegrationLog(
        integration_id=integration.id,
        action="disconnected",
        details={"integration_type": integration.integration_type},
    )
    db.add(log)
    await db.flush()

    await db.delete(integration)
    await db.commit()

    logger.info(f"Deleted integration {integration_id} for user {user_id}")


async def test_integration_connection(
    db: AsyncSession,
    integration_id: str,
    user_id: str,
) -> dict[str, Any]:
    """Test an integration connection using MCP validation."""
    integration = await get_integration(db, integration_id, user_id)
    if not integration:
        raise ValueError("Integration not found")

    # Decrypt credentials
    credentials = decrypt_credentials(integration.credentials_encrypted)

    if not credentials:
        integration.status = "error"
        integration.error_message = "Invalid credentials"
        await db.commit()
        raise ValueError("Invalid credentials")

    # Use MCP validation to actually test the connection
    try:
        from app.tools.integration_mcp_tools import validate_integration

        success, message, tools = await validate_integration(
            integration.integration_type,
            credentials,
            integration.config,
        )

        if success:
            integration.status = "connected"
            integration.error_message = None
            integration.last_sync_at = datetime.utcnow()

            # Log the successful test
            log = IntegrationLog(
                integration_id=integration.id,
                action="test_connection",
                details={"status": "success", "message": message, "tool_count": len(tools)},
            )
            db.add(log)

            await db.commit()

            return {
                "status": "success",
                "message": message,
                "integration_type": integration.integration_type,
                "tools": tools,
            }
        else:
            integration.status = "error"
            integration.error_message = message

            # Log the failed test
            log = IntegrationLog(
                integration_id=integration.id,
                action="test_connection",
                details={"status": "error", "message": message},
            )
            db.add(log)

            await db.commit()

            raise ValueError(message)

    except ImportError as e:
        logger.error(f"Could not import MCP validation: {e}")
        # Fallback to basic validation
        integration.status = "connected"
        integration.error_message = None
        integration.last_sync_at = datetime.utcnow()

        log = IntegrationLog(
            integration_id=integration.id,
            action="test_connection",
            details={"status": "success", "note": "Basic validation only"},
        )
        db.add(log)

        await db.commit()

        return {
            "status": "success",
            "message": "Integration configured (MCP validation not available)",
            "integration_type": integration.integration_type,
        }


async def get_integration_credentials(
    db: AsyncSession,
    integration_id: str,
    user_id: str,
) -> dict[str, Any]:
    """Get decrypted credentials for an integration."""
    integration = await get_integration(db, integration_id, user_id)
    if not integration:
        raise ValueError("Integration not found")

    if not integration.enabled:
        raise ValueError("Integration is disabled")

    return decrypt_credentials(integration.credentials_encrypted)
