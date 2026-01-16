"""
Graph Query Tools - Direct graph queries without LLM synthesis.

Provides LangGraph tools for the agent to query the Cognee knowledge graph
directly using Cypher queries and text search.
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

logger = logging.getLogger(__name__)

# Graph schema for agent context
GRAPH_SCHEMA = """
## Knowledge Graph Schema

### Node Types
- **Entity**: Named entities (people, teams, organizations, events, etc.)
- **EntityType**: Categories of entities (person, team, organization, event, etc.)
- **TextSummary**: Document summaries
- **DocumentChunk**: Raw text passages from documents
- **TextDocument**: Source documents

### Key Entity Types
- person, team, organization, event, session, project, program, school

### Relationship Types
- `is_a`: Entity type classification (e.g., defenx -[is_a]-> team)
- `mentored_by`: Mentor relationships (e.g., defenx -[mentored_by]-> alex onufrak)
- `has_member`: Team membership (e.g., defenx -[has_member]-> arturalsina)
- `participates_in`: Event participation
- `held_on`: Date relationships (e.g., session -[held_on]-> 2025-12-15)
- `associated_with`: General associations
- `works_at`: Employment
- `is_part_of`: Hierarchical containment
- `contains`: Document structure

### Example Queries
- Find team mentors: `MATCH (t:Entity)-[r:mentored_by]->(m:Entity) WHERE t.name = 'defenx' RETURN m.name`
- Find session dates: `MATCH (e:Entity)-[r:held_on]->(d) WHERE e.name CONTAINS 'session' RETURN e.name, d`
- Find team members: `MATCH (t:Entity)-[:has_member]->(p:Entity) WHERE t.name = 'defenx' RETURN p.name`
- Search by type: `MATCH (e:Entity)-[:is_a]->(t:EntityType) WHERE t.name = 'team' RETURN e.name`
"""


def _get_auth_headers(body_str: str) -> dict[str, str]:
    """Generate HMAC authentication headers for Cognee API."""
    settings = get_settings()
    if not settings.cognee_secret_key:
        return {"Content-Type": "application/json"}

    timestamp = str(int(time.time()))
    message = f"{timestamp}.{body_str}"
    signature = hmac.new(
        settings.cognee_secret_key.encode("utf-8"),
        message.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return {
        "X-Timestamp": timestamp,
        "X-Signature": signature,
        "Content-Type": "application/json",
    }


async def _cognee_search(query: str, search_type: str, top_k: int = 20) -> dict[str, Any]:
    """Execute a Cognee search request."""
    settings = get_settings()

    payload = {
        "query": query,
        "search_type": search_type,
        "top_k": top_k,
    }
    body_str = json.dumps(payload)

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{settings.cognee_api_url}/search",
                content=body_str,
                headers=_get_auth_headers(body_str),
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as e:
        logger.error(f"Cognee {search_type} request failed: {e}")
        return {"error": str(e), "results": []}


# Tool input schemas
class QueryGraphInput(BaseModel):
    """Input schema for query_graph tool."""

    cypher_query: str = Field(..., description="A valid Cypher query string")


class SearchTextInput(BaseModel):
    """Input schema for search_text tool."""

    query: str = Field(..., description="Natural language search query")
    top_k: int = Field(default=20, description="Number of results to return")


class FindEntityInput(BaseModel):
    """Input schema for find_entity tool."""

    name: str = Field(..., description="Entity name to search for (case-insensitive)")


@tool(args_schema=QueryGraphInput)
async def query_graph(cypher_query: str) -> dict[str, Any]:
    """
    Execute a Cypher query against the knowledge graph.

    Use this to query entities, relationships, and patterns in the graph.
    Returns raw results without LLM synthesis - you must interpret the results.

    **Schema Reference:**
    - Node types: Entity, EntityType, TextSummary, DocumentChunk
    - Entity types: person, team, organization, event, session, project
    - Key relationships: is_a, mentored_by, has_member, participates_in, held_on

    **Example Queries:**
    - Find all teams: `MATCH (e:Entity)-[:is_a]->(:EntityType {name: 'team'}) RETURN e.name`
    - Find team's mentors: `MATCH (t:Entity {name: 'defenx'})-[:mentored_by]->(m) RETURN m.name`
    - Find sessions on a date: `MATCH (s:Entity)-[:held_on]->(d) WHERE d CONTAINS '2025-12' RETURN s`

    Args:
        cypher_query: A valid Cypher query string

    Returns:
        Dict with 'results' containing query results as list of records
    """
    logger.info(f"Executing Cypher query: {cypher_query[:100]}...")
    result = await _cognee_search(cypher_query, "CYPHER", top_k=50)

    results = result.get("results", [])
    logger.info(f"Cypher query returned {len(results)} results")

    return {
        "query": cypher_query,
        "results": results,
        "count": len(results),
    }


@tool(args_schema=SearchTextInput)
async def search_text(query: str, top_k: int = 20) -> dict[str, Any]:
    """
    Search text chunks using semantic similarity.

    Returns raw text passages from documents that match the query.
    No LLM synthesis - you must read and interpret the text yourself.

    Use this for:
    - Finding specific mentions or quotes
    - Getting detailed context about topics
    - Searching when you don't know the exact entity names

    Args:
        query: Natural language search query
        top_k: Number of results to return (default 20)

    Returns:
        Dict with 'results' containing text chunks with their content
    """
    logger.info(f"Searching text chunks: {query}")
    result = await _cognee_search(query, "CHUNKS", top_k)

    # Extract just the text content for easier processing
    chunks = []
    for r in result.get("results", []):
        if isinstance(r, dict) and "text" in r:
            chunks.append({
                "text": r["text"],
                "id": r.get("id"),
            })

    logger.info(f"Text search returned {len(chunks)} chunks")

    return {
        "query": query,
        "results": chunks,
        "count": len(chunks),
    }


@tool(args_schema=FindEntityInput)
async def find_entity(name: str) -> dict[str, Any]:
    """
    Find an entity by name and return its properties and relationships.

    This is a convenience function that runs multiple Cypher queries to get
    comprehensive information about an entity.

    Args:
        name: The entity name to search for (case-insensitive partial match)

    Returns:
        Dict with entity info, type, and relationships
    """
    logger.info(f"Finding entity: {name}")

    # Find the entity
    entity_query = f"MATCH (e:Entity) WHERE toLower(e.name) CONTAINS toLower('{name}') RETURN e.name, e.description, labels(e) LIMIT 5"
    entities = await _cognee_search(entity_query, "CYPHER", top_k=5)

    if not entities.get("results"):
        return {"found": False, "message": f"No entity found matching '{name}'"}

    entity = entities["results"][0]
    entity_name = entity.get("e.name")

    # Get outgoing relationships
    out_query = f"MATCH (e:Entity {{name: '{entity_name}'}})-[r]->(t) RETURN type(r) as rel, t.name as target LIMIT 30"
    outgoing = await _cognee_search(out_query, "CYPHER", top_k=30)

    # Get incoming relationships
    in_query = f"MATCH (s)-[r]->(e:Entity {{name: '{entity_name}'}}) RETURN s.name as source, type(r) as rel LIMIT 30"
    incoming = await _cognee_search(in_query, "CYPHER", top_k=30)

    return {
        "found": True,
        "name": entity_name,
        "description": entity.get("e.description"),
        "labels": entity.get("labels(e)"),
        "outgoing_relationships": outgoing.get("results", []),
        "incoming_relationships": incoming.get("results", []),
    }


@tool
async def get_graph_schema() -> dict[str, Any]:
    """
    Get the knowledge graph schema including node types and relationships.

    Call this first to understand what you can query in the graph.

    Returns:
        Dict with node types, entity types, and relationship types
    """
    logger.info("Fetching graph schema")

    # Get node label counts
    labels_query = "MATCH (n) RETURN labels(n) as labels, count(*) as count ORDER BY count DESC LIMIT 15"
    labels = await _cognee_search(labels_query, "CYPHER", top_k=15)

    # Get relationship type counts
    rels_query = "MATCH ()-[r]->() RETURN type(r) as rel_type, count(*) as count ORDER BY count DESC LIMIT 20"
    rels = await _cognee_search(rels_query, "CYPHER", top_k=20)

    # Get entity types
    types_query = "MATCH (e:EntityType) RETURN e.name as name LIMIT 25"
    types = await _cognee_search(types_query, "CYPHER", top_k=25)

    return {
        "node_labels": labels.get("results", []),
        "relationship_types": rels.get("results", []),
        "entity_types": [t.get("name") for t in types.get("results", [])],
        "schema_guide": GRAPH_SCHEMA,
    }
