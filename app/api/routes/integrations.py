"""API routes for integrations management."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_integration_types
from app.db.session import get_db
from app.models.integration import (
    IntegrationCreate,
    IntegrationResponse,
    IntegrationTestResponse,
    IntegrationTypeField,
    IntegrationTypeResponse,
    IntegrationUpdate,
)
from app.services import integration_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/integrations", tags=["integrations"])


@router.get("/types", response_model=list[IntegrationTypeResponse])
async def list_integration_types() -> list[dict[str, Any]]:
    """List all available integration types."""
    types = get_integration_types()

    # Transform to match response model
    return [
        {
            "id": t["id"],
            "name": t["name"],
            "description": t["description"],
            "icon": t["icon"],
            "category": t["category"],
            "comingSoon": t.get("coming_soon", False),
            "configFields": [
                {
                    "name": f["name"],
                    "label": f["label"],
                    "type": f["type"],
                    "placeholder": f["placeholder"],
                    "required": f["required"],
                    "helpText": f["help_text"],
                }
                for f in t["config_fields"]
            ],
        }
        for t in types
    ]


@router.get("", response_model=list[IntegrationResponse])
async def list_integrations(
    user_id: str = Query(..., description="User ID from Auth0"),
    enabled_only: bool = Query(False, description="Only return enabled integrations"),
    db: AsyncSession = Depends(get_db),
) -> list[IntegrationResponse]:
    """List all integrations for a user."""
    try:
        integrations = await integration_service.get_user_integrations(
            db, user_id, enabled_only=enabled_only
        )
        return [IntegrationResponse.model_validate(i) for i in integrations]
    except Exception as e:
        logger.error(f"Error listing integrations for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to list integrations")


@router.get("/{integration_id}", response_model=IntegrationResponse)
async def get_integration(
    integration_id: str,
    user_id: str = Query(..., description="User ID from Auth0"),
    db: AsyncSession = Depends(get_db),
) -> IntegrationResponse:
    """Get a specific integration by ID."""
    try:
        integration = await integration_service.get_integration(db, integration_id, user_id)
        if not integration:
            raise HTTPException(status_code=404, detail="Integration not found")
        return IntegrationResponse.model_validate(integration)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting integration {integration_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get integration")


@router.post("", response_model=IntegrationResponse, status_code=201)
async def create_integration(
    request: IntegrationCreate,
    user_id: str = Query(..., description="User ID from Auth0"),
    db: AsyncSession = Depends(get_db),
) -> IntegrationResponse:
    """Create a new integration for a user."""
    try:
        integration = await integration_service.create_integration(
            db,
            user_id=user_id,
            integration_type=request.integration_type,
            credentials=request.credentials,
            enabled=request.enabled,
        )
        return IntegrationResponse.model_validate(integration)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating integration: {e}")
        raise HTTPException(status_code=500, detail="Failed to create integration")


@router.put("/{integration_id}", response_model=IntegrationResponse)
async def update_integration(
    integration_id: str,
    request: IntegrationUpdate,
    user_id: str = Query(..., description="User ID from Auth0"),
    db: AsyncSession = Depends(get_db),
) -> IntegrationResponse:
    """Update an existing integration."""
    try:
        integration = await integration_service.update_integration(
            db,
            integration_id=integration_id,
            user_id=user_id,
            credentials=request.credentials,
            enabled=request.enabled,
        )
        return IntegrationResponse.model_validate(integration)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error updating integration {integration_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to update integration: {str(e)}")


@router.delete("/{integration_id}", status_code=204)
async def delete_integration(
    integration_id: str,
    user_id: str = Query(..., description="User ID from Auth0"),
    db: AsyncSession = Depends(get_db),
) -> None:
    """Delete an integration."""
    try:
        await integration_service.delete_integration(db, integration_id, user_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error deleting integration {integration_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete integration")


@router.post("/{integration_id}/test", response_model=IntegrationTestResponse)
async def test_integration(
    integration_id: str,
    user_id: str = Query(..., description="User ID from Auth0"),
    db: AsyncSession = Depends(get_db),
) -> IntegrationTestResponse:
    """Test an integration connection."""
    try:
        result = await integration_service.test_integration_connection(db, integration_id, user_id)
        return IntegrationTestResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error testing integration {integration_id}: {e}")
        raise HTTPException(status_code=500, detail="Connection test failed")
