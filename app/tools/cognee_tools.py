"""
Cognee Search Tools - HTTP client wrapper for cognee-service.

Provides multiple LangGraph tools for parallel knowledge retrieval:
- Chunks: Raw text segments with embeddings
- Graph: Knowledge graph traversal and synthesis
- RAG: Traditional semantic search
- Summaries: Document-level overviews
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
from typing import Any, Optional

import httpx
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.config import get_settings
from app.tools.sanitize import sanitize_for_json

logger = logging.getLogger(__name__)


def _compute_hmac_signature(timestamp: str, body: str, secret: str) -> str:
    """Compute HMAC-SHA256 signature for request verification."""
    message = f"{timestamp}.{body}"
    return hmac.new(
        secret.encode("utf-8"),
        message.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _get_auth_headers(body_str: str) -> dict[str, str]:
    """Generate HMAC authentication headers for the request."""
    settings = get_settings()
    if not settings.cognee_secret_key:
        return {"Content-Type": "application/json"}

    timestamp = str(int(time.time()))
    signature = _compute_hmac_signature(timestamp, body_str, settings.cognee_secret_key)
    return {
        "X-Timestamp": timestamp,
        "X-Signature": signature,
        "Content-Type": "application/json",
    }


async def _cognee_search(
    query: str,
    search_type: str,
    top_k: int = 10,
) -> dict[str, Any]:
    """Internal search function that calls Cognee API."""
    settings = get_settings()

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            payload = {
                "query": query,
                "search_type": search_type,
                "top_k": top_k,
            }
            body_str = json.dumps(payload)

            response = await client.post(
                f"{settings.cognee_api_url}/search",
                content=body_str,
                headers=_get_auth_headers(body_str),
            )
            response.raise_for_status()
            result = response.json()

            # Add source metadata for citations
            for r in result.get("results", []):
                if isinstance(r, dict):
                    r["_source"] = f"cognee:{search_type.lower()}"
                    r["_search_type"] = search_type

            logger.info(f"Cognee {search_type} search: {len(result.get('results', []))} results")
            return sanitize_for_json(result)

    except httpx.HTTPError as e:
        logger.error(f"Cognee {search_type} search failed: {e}")
        return {"error": str(e), "results": [], "search_type": search_type}


# Tool input schemas
class SearchInput(BaseModel):
    """Input schema for search tools."""

    query: str = Field(..., description="Natural language search query")
    top_k: int = Field(default=20, description="Number of results to return")


class AddContentInput(BaseModel):
    """Input schema for cognee_add tool."""

    content: str = Field(..., description="Text content to add to the knowledge base")
    dataset: str = Field(default="default", description="Dataset name")


@tool(args_schema=SearchInput)
async def search_chunks(query: str, top_k: int = 20) -> dict[str, Any]:
    """
    Search for raw text chunks from documents.

    Returns the actual text segments that were indexed, with their
    embedding similarity scores. Best for finding specific passages
    or when you need the original source text.

    Args:
        query: Natural language search query
        top_k: Number of chunks to retrieve (default 20)

    Returns:
        Dict with 'results' containing text chunks with scores and metadata.
    """
    return await _cognee_search(query, "CHUNKS", top_k)


@tool(args_schema=SearchInput)
async def search_graph(query: str, top_k: int = 15) -> dict[str, Any]:
    """
    Search using knowledge graph traversal and LLM synthesis.

    Traverses the knowledge graph to find relevant entities and
    relationships, then synthesizes a comprehensive answer.
    Best for complex questions requiring reasoning across multiple facts.

    Args:
        query: Natural language search query
        top_k: Number of graph paths to consider (default 15)

    Returns:
        Dict with 'results' containing synthesized answer from graph.
    """
    return await _cognee_search(query, "GRAPH_COMPLETION", top_k)


@tool(args_schema=SearchInput)
async def search_rag(query: str, top_k: int = 25) -> dict[str, Any]:
    """
    Traditional RAG-style semantic search with LLM completion.

    Retrieves semantically similar content and generates an answer.
    Best for straightforward questions where embedding similarity
    is sufficient to find relevant information.

    Args:
        query: Natural language search query
        top_k: Number of documents to retrieve (default 25)

    Returns:
        Dict with 'results' containing RAG-generated answer.
    """
    return await _cognee_search(query, "RAG_COMPLETION", top_k)


@tool(args_schema=SearchInput)
async def search_summaries(query: str, top_k: int = 5) -> dict[str, Any]:
    """
    Search document summaries for high-level overview.

    Retrieves pre-computed document summaries rather than raw chunks.
    Best for getting a quick overview of topics or documents
    without detailed specifics.

    Args:
        query: Natural language search query
        top_k: Number of summaries to retrieve (default 5)

    Returns:
        Dict with 'results' containing document summaries.
    """
    return await _cognee_search(query, "SUMMARIES", top_k)


@tool(args_schema=SearchInput)
async def search_natural_language(query: str, top_k: int = 10) -> dict[str, Any]:
    """
    Search using natural language query translation.

    Translates natural language queries into graph queries
    to find relevant entities and relationships. Best for
    complex questions about connections between concepts.

    Args:
        query: Natural language query
        top_k: Maximum results to return (default 10)

    Returns:
        Dict with 'results' containing matched entities/relationships.
    """
    return await _cognee_search(query, "NATURAL_LANGUAGE", top_k)


@tool(args_schema=AddContentInput)
async def cognee_add(
    content: str,
    dataset: str = "default",
) -> dict[str, Any]:
    """
    Add knowledge to Cognee for processing and graph construction.

    Args:
        content: Text content to add to the knowledge base
        dataset: Dataset name to add content to

    Returns:
        Status of the add operation
    """
    settings = get_settings()

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            payload = {
                "content": content,
                "dataset": dataset,
                "metadata": {},
            }
            body_str = json.dumps(payload)

            response = await client.post(
                f"{settings.cognee_api_url}/add",
                content=body_str,
                headers=_get_auth_headers(body_str),
            )
            response.raise_for_status()
            result = response.json()

            logger.info(f"Cognee add completed: {result.get('status', 'unknown')}")
            return sanitize_for_json(result)

    except httpx.HTTPError as e:
        logger.error(f"Cognee add failed: {e}")
        return {"error": str(e), "status": "failed"}
