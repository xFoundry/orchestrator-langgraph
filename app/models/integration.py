"""Pydantic models for integration API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_serializer


class IntegrationCreate(BaseModel):
    """Request model for creating an integration."""

    integration_type: str = Field(..., alias="integrationType", description="Type of integration (e.g., 'wrike', 'linear')")
    credentials: dict[str, str] = Field(..., description="Integration credentials")
    enabled: bool = Field(default=True, description="Whether the integration is enabled")

    class Config:
        populate_by_name = True


class IntegrationUpdate(BaseModel):
    """Request model for updating an integration."""

    credentials: Optional[dict[str, str]] = Field(None, description="Updated credentials")
    enabled: Optional[bool] = Field(None, description="Whether the integration is enabled")


class IntegrationResponse(BaseModel):
    """Response model for integration."""

    id: str | UUID
    user_id: str = Field(alias="userId")
    integration_type: str = Field(alias="integrationType")
    name: Optional[str] = None
    status: str
    enabled: bool
    last_sync_at: Optional[datetime] = Field(None, alias="lastSyncAt")
    error_message: Optional[str] = Field(None, alias="errorMessage")
    created_at: datetime = Field(alias="createdAt")
    updated_at: datetime = Field(alias="updatedAt")

    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
        serialize_by_alias=True,
    )

    @field_serializer("id")
    def serialize_id(self, v: str | UUID) -> str:
        return str(v)


class IntegrationToolInfo(BaseModel):
    """Information about an available tool from an integration."""

    name: str
    description: Optional[str] = None

    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)


class IntegrationTestResponse(BaseModel):
    """Response model for integration connection test."""

    status: str
    message: str
    integration_type: str = Field(alias="integrationType")
    tools: list[IntegrationToolInfo] = Field(default_factory=list)

    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)


class IntegrationTypeField(BaseModel):
    """Configuration field definition."""

    name: str
    label: str
    type: str
    placeholder: str
    required: bool
    help_text: str = Field(alias="helpText")

    class Config:
        populate_by_name = True


class IntegrationTypeResponse(BaseModel):
    """Response model for integration type."""

    id: str
    name: str
    description: str
    icon: str
    category: str
    coming_soon: bool = Field(default=False, alias="comingSoon")
    config_fields: list[IntegrationTypeField] = Field(alias="configFields")

    class Config:
        populate_by_name = True
