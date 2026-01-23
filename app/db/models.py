"""SQLAlchemy models for integrations."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import JSON, Boolean, Column, DateTime, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.db.session import Base


class Integration(Base):
    """User integration settings with encrypted credentials."""

    __tablename__ = "integrations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String(255), nullable=False, index=True)
    integration_type = Column(String(50), nullable=False)
    name = Column(String(100))
    status = Column(String(20), default="disconnected")  # connected, disconnected, error
    enabled = Column(Boolean, default=False)
    credentials_encrypted = Column(Text)  # AES-256 encrypted JSON
    config = Column(JSON)  # Additional configuration
    last_sync_at = Column(DateTime(timezone=True))
    error_message = Column(Text)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationship to logs
    logs = relationship("IntegrationLog", back_populates="integration", cascade="all, delete-orphan")

    # Unique constraint on user_id + integration_type
    __table_args__ = (
        {"schema": None},
    )


class IntegrationLog(Base):
    """Integration activity audit log."""

    __tablename__ = "integration_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    integration_id = Column(UUID(as_uuid=True), ForeignKey("integrations.id"), nullable=False)
    action = Column(String(50), nullable=False)  # connected, disconnected, tool_called, error
    details = Column(JSON)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    # Relationship to integration
    integration = relationship("Integration", back_populates="logs")
