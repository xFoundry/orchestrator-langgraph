"""
Outline MCP Tools - Server-to-server access via Outline MCP endpoint.

Wraps high-value MCP tools as LangGraph StructuredTool functions.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.tools.outline_mcp_client import get_outline_mcp_client
from app.tools.sanitize import sanitize_for_json

logger = logging.getLogger(__name__)


def _parse_mcp_result(result: dict[str, Any]) -> dict[str, Any]:
    """Extract JSON payload from MCP content blocks when possible."""
    content = result.get("content")
    if not isinstance(content, list):
        return sanitize_for_json(result)
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text")
            if isinstance(text, str):
                try:
                    return sanitize_for_json(json.loads(text))
                except Exception:
                    return sanitize_for_json({"text": text})
    return sanitize_for_json(result)


def _truncate(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"


class CollectionsListInput(BaseModel):
    include_archived: bool = Field(
        default=False,
        description="Include archived/deleted collections.",
    )


@tool(args_schema=CollectionsListInput)
async def outline_collections_list(include_archived: bool = False) -> dict[str, Any]:
    """List collections (folders/categories) in Outline."""
    client = get_outline_mcp_client()
    result = await client.call_tool(
        "collections_list",
        {"includeArchived": include_archived},
    )
    return _parse_mcp_result(result)


class CollectionsInfoInput(BaseModel):
    id: str = Field(..., description="Collection ID (UUID).")


@tool(args_schema=CollectionsInfoInput)
async def outline_collections_info(id: str) -> dict[str, Any]:
    """Get collection details including document structure."""
    client = get_outline_mcp_client()
    result = await client.call_tool("collections_info", {"id": id})
    return _parse_mcp_result(result)


class CollectionsDocumentsInput(BaseModel):
    id: str = Field(..., description="Collection ID (UUID).")


@tool(args_schema=CollectionsDocumentsInput)
async def outline_collections_documents(id: str) -> dict[str, Any]:
    """Get a flat list of documents for a collection (with hierarchy paths)."""
    client = get_outline_mcp_client()
    result = await client.call_tool("collections_documents", {"id": id})
    return _parse_mcp_result(result)


class DocumentsListInput(BaseModel):
    collection_id: Optional[str] = Field(
        default=None,
        description="Optional collection ID to filter documents.",
    )
    user_id: Optional[str] = Field(
        default=None,
        description="Optional user ID to filter documents.",
    )
    limit: int = Field(default=25, ge=1, le=100, description="Max documents.")
    offset: int = Field(default=0, ge=0, description="Pagination offset.")
    sort: str = Field(
        default="updatedAt",
        description="Sort by updatedAt, createdAt, title, or index.",
    )
    direction: str = Field(default="DESC", description="ASC or DESC.")


@tool(args_schema=DocumentsListInput)
async def outline_documents_list(
    collection_id: Optional[str] = None,
    user_id: Optional[str] = None,
    limit: int = 25,
    offset: int = 0,
    sort: str = "updatedAt",
    direction: str = "DESC",
) -> dict[str, Any]:
    """List documents with metadata (no full content)."""
    client = get_outline_mcp_client()
    # Build args, excluding None values (MCP server rejects null)
    args: dict[str, Any] = {
        "limit": limit,
        "offset": offset,
        "sort": sort,
        "direction": direction,
    }
    if collection_id is not None:
        args["collectionId"] = collection_id
    if user_id is not None:
        args["userId"] = user_id

    result = await client.call_tool("documents_list", args)
    return _parse_mcp_result(result)


class DocumentsSearchInput(BaseModel):
    query: str = Field(..., description="Search terms for Outline documents.")
    collection_id: Optional[str] = Field(
        default=None, description="Optional collection ID filter."
    )
    limit: int = Field(default=25, ge=1, le=100, description="Max results.")
    include_archived: bool = Field(default=False, description="Search archived docs.")
    include_drafts: bool = Field(default=False, description="Search draft docs.")


@tool(args_schema=DocumentsSearchInput)
async def outline_documents_search(
    query: str,
    collection_id: Optional[str] = None,
    limit: int = 25,
    include_archived: bool = False,
    include_drafts: bool = False,
) -> dict[str, Any]:
    """Search Outline documents by keyword/phrase."""
    client = get_outline_mcp_client()
    # Build args, excluding None values (MCP server rejects null)
    args: dict[str, Any] = {
        "query": query,
        "limit": limit,
        "includeArchived": include_archived,
        "includeDrafts": include_drafts,
    }
    if collection_id is not None:
        args["collectionId"] = collection_id

    result = await client.call_tool("documents_search", args)
    parsed = _parse_mcp_result(result)
    if isinstance(parsed, dict) and "results" in parsed:
        parsed["results"] = (parsed["results"] or [])[:limit]
    return parsed


class DocumentsInfoInput(BaseModel):
    id: str = Field(..., description="Document ID (UUID) or URL slug.")
    max_chars: int = Field(
        default=4000,
        ge=500,
        description="Max chars of document text to return.",
    )


@tool(args_schema=DocumentsInfoInput)
async def outline_documents_info(id: str, max_chars: int = 4000) -> dict[str, Any]:
    """Get full document content (truncated)."""
    client = get_outline_mcp_client()
    result = await client.call_tool("documents_info", {"id": id})
    parsed = _parse_mcp_result(result)
    if isinstance(parsed, dict) and isinstance(parsed.get("text"), str):
        parsed["text"] = _truncate(parsed["text"], max_chars)
    return parsed


def get_outline_mcp_tools() -> list:
    return [
        outline_collections_list,
        outline_collections_info,
        outline_collections_documents,
        outline_documents_list,
        outline_documents_search,
        outline_documents_info,
    ]

