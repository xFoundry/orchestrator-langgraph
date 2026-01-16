"""LangGraph CLI entrypoint for Agent Server deployments."""

from __future__ import annotations

from typing import Optional

from langchain_core.runnables import RunnableConfig

from app.graphs.orchestrator_v3 import create_orchestrator_v3


def make_graph(config: Optional[RunnableConfig] = None):
    """Return the compiled orchestrator graph for LangGraph Agent Server."""
    _ = config
    return create_orchestrator_v3()
