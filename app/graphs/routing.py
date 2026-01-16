"""
Query Classification and Routing for Orchestrator v2.

Analyzes incoming queries to determine the best workflow pattern:
- Simple: Quick parallel search
- Complex: Broad parallel search with multiple strategies
- Analytical: Deep reasoning with evaluator-optimizer
- Action: Live Mentor Hub API access
"""

from __future__ import annotations

import logging
import re
from typing import Literal

logger = logging.getLogger(__name__)


# Query type definitions
QueryType = Literal["simple", "complex", "analytical", "action"]


# =============================================================================
# KEYWORD PATTERNS
# =============================================================================

# Simple lookup queries - factual questions with straightforward answers
SIMPLE_PATTERNS = [
    r"\bwhat is\b",
    r"\bwho is\b",
    r"\btell me about\b",
    r"\blist\b",
    r"\bshow me\b",
    r"\bfind\b",
    r"\bget\b",
    r"\bdescribe\b",
]

# Complex queries - need multiple search strategies
COMPLEX_PATTERNS = [
    r"\bexplain\b",
    r"\bwhat are\b",
    r"\bhow many\b",
    r"\bwhat.*challenges\b",
    r"\bwhat.*problems\b",
    r"\bwhat.*feedback\b",
    r"\bwhat.*recommendations\b",
]

# Analytical queries - require reasoning, comparison, synthesis
ANALYTICAL_PATTERNS = [
    r"\bcompare\b",
    r"\banalyze\b",
    r"\bwhy\b",
    r"\bhow does\b",
    r"\bwhat.*best\b",
    r"\bwhat.*should\b",
    r"\brecommend\b",
    r"\bsuggest\b",
    r"\bevaluate\b",
    r"\bassess\b",
    r"\bwhat patterns\b",
    r"\bwhat insights\b",
]

# Action queries - need live data from Mentor Hub
ACTION_PATTERNS = [
    r"\bmy\s+sessions?\b",
    r"\bmy\s+team\b",
    r"\bmy\s+tasks?\b",
    r"\bupcoming\b",
    r"\bscheduled\b",
    r"\bthis week\b",
    r"\bnext week\b",
    r"\btoday\b",
    r"\btomorrow\b",
    r"\bcurrent\b",
    r"\blive\b",
    r"\bcreate\b",
    r"\bschedule\b",
    r"\bbook\b",
    r"\bavailable\b",
    r"\bavailability\b",
]

# Mentor matching queries - specialized workflow
MENTOR_MATCHING_PATTERNS = [
    r"\bmentor.*match\b",
    r"\bfind.*mentor\b",
    r"\bwho.*mentor\b",
    r"\brecommend.*mentor\b",
    r"\bbest.*mentor\b",
    r"\bmentor.*for\b",
]


def _matches_any_pattern(text: str, patterns: list[str]) -> bool:
    """Check if text matches any of the given patterns."""
    text_lower = text.lower()
    return any(re.search(pattern, text_lower) for pattern in patterns)


def _count_pattern_matches(text: str, patterns: list[str]) -> int:
    """Count how many patterns match the text."""
    text_lower = text.lower()
    return sum(1 for pattern in patterns if re.search(pattern, text_lower))


# =============================================================================
# CLASSIFICATION FUNCTIONS
# =============================================================================

def classify_query(query: str) -> QueryType:
    """
    Classify a query to determine the best workflow.

    Uses pattern matching to identify query intent and complexity.

    Args:
        query: The user's query string

    Returns:
        Query type: "simple", "complex", "analytical", or "action"
    """
    # Check for action queries first (live data needs)
    if _matches_any_pattern(query, ACTION_PATTERNS):
        logger.debug(f"Query classified as 'action': {query[:50]}...")
        return "action"

    # Check for analytical queries (need reasoning)
    if _matches_any_pattern(query, ANALYTICAL_PATTERNS):
        logger.debug(f"Query classified as 'analytical': {query[:50]}...")
        return "analytical"

    # Check for mentor matching (specialized)
    if _matches_any_pattern(query, MENTOR_MATCHING_PATTERNS):
        logger.debug(f"Query classified as 'analytical' (mentor matching): {query[:50]}...")
        return "analytical"

    # Check complexity based on pattern counts
    complex_matches = _count_pattern_matches(query, COMPLEX_PATTERNS)
    simple_matches = _count_pattern_matches(query, SIMPLE_PATTERNS)

    # Multi-part questions are complex
    if query.count("?") > 1 or " and " in query.lower():
        logger.debug(f"Query classified as 'complex' (multi-part): {query[:50]}...")
        return "complex"

    # Long queries tend to be complex
    word_count = len(query.split())
    if word_count > 15:
        logger.debug(f"Query classified as 'complex' (long): {query[:50]}...")
        return "complex"

    # Use pattern matching
    if complex_matches > simple_matches:
        logger.debug(f"Query classified as 'complex': {query[:50]}...")
        return "complex"

    if simple_matches > 0:
        logger.debug(f"Query classified as 'simple': {query[:50]}...")
        return "simple"

    # Default to complex for safety
    logger.debug(f"Query classified as 'complex' (default): {query[:50]}...")
    return "complex"


def get_workflow_for_query_type(query_type: QueryType) -> str:
    """
    Map query type to workflow name.

    Args:
        query_type: The classified query type

    Returns:
        Workflow name to execute
    """
    workflow_map = {
        "simple": "parallel_researcher",      # Quick parallel search
        "complex": "parallel_researcher",     # Broad parallel search
        "analytical": "evaluator_optimizer",  # Reasoning with quality gates
        "action": "mentor_hub_workflow",      # Live API access
    }

    return workflow_map.get(query_type, "parallel_researcher")


def should_use_parallel_research(query_type: QueryType) -> bool:
    """Check if query should use parallel research pattern."""
    return query_type in ("simple", "complex")


def should_use_quality_gates(query_type: QueryType) -> bool:
    """Check if query should use evaluator-optimizer pattern."""
    return query_type == "analytical"


def should_use_live_data(query_type: QueryType) -> bool:
    """Check if query needs live Mentor Hub data."""
    return query_type == "action"


# =============================================================================
# ENTITY EXTRACTION
# =============================================================================

def extract_entities_from_query(query: str) -> list[str]:
    """
    Extract potential entity names from a query.

    Looks for capitalized words and known entity patterns.

    Args:
        query: The user's query string

    Returns:
        List of potential entity names
    """
    entities = []

    # Find capitalized words (potential names/teams)
    # Skip common words that are often capitalized at sentence start
    skip_words = {"what", "who", "how", "when", "where", "why", "the", "a", "an", "i"}
    words = query.split()

    for i, word in enumerate(words):
        # Skip first word (often capitalized regardless)
        if i == 0:
            continue

        # Check if word is capitalized
        clean_word = word.strip(".,?!\"'")
        if clean_word and clean_word[0].isupper() and clean_word.lower() not in skip_words:
            entities.append(clean_word)

    # Look for quoted strings
    quoted = re.findall(r'"([^"]+)"', query)
    entities.extend(quoted)

    quoted_single = re.findall(r"'([^']+)'", query)
    entities.extend(quoted_single)

    return list(set(entities))


# =============================================================================
# QUERY ANALYSIS
# =============================================================================

def analyze_query(query: str) -> dict:
    """
    Perform comprehensive query analysis.

    Args:
        query: The user's query string

    Returns:
        Analysis dict with query_type, workflow, entities, etc.
    """
    query_type = classify_query(query)
    entities = extract_entities_from_query(query)

    return {
        "query_type": query_type,
        "workflow": get_workflow_for_query_type(query_type),
        "entities": entities,
        "use_parallel": should_use_parallel_research(query_type),
        "use_quality_gates": should_use_quality_gates(query_type),
        "use_live_data": should_use_live_data(query_type),
        "word_count": len(query.split()),
        "is_multi_part": query.count("?") > 1 or " and " in query.lower(),
    }
