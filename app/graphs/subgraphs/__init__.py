"""Specialized sub-agent subgraphs for the orchestrator."""

from app.graphs.subgraphs.researchers import (
    create_entity_researcher,
    create_text_researcher,
    create_summary_researcher,
    create_deep_reasoning,
    create_mentor_matcher,
    ENTITY_RESEARCHER_DESCRIPTION,
    TEXT_RESEARCHER_DESCRIPTION,
    SUMMARY_RESEARCHER_DESCRIPTION,
    DEEP_REASONING_DESCRIPTION,
    MENTOR_MATCHER_DESCRIPTION,
)

__all__ = [
    "create_entity_researcher",
    "create_text_researcher",
    "create_summary_researcher",
    "create_deep_reasoning",
    "create_mentor_matcher",
    "ENTITY_RESEARCHER_DESCRIPTION",
    "TEXT_RESEARCHER_DESCRIPTION",
    "SUMMARY_RESEARCHER_DESCRIPTION",
    "DEEP_REASONING_DESCRIPTION",
    "MENTOR_MATCHER_DESCRIPTION",
]
