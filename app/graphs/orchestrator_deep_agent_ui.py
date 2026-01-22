"""Deep Agent graph with Agent Chat UI-focused event mapping."""

from __future__ import annotations

import logging
from typing import Optional

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from app.graphs.orchestrator_deep_agent import get_orchestrator_deep_agent
from app.middleware.deep_agent_ui import DeepAgentUIMiddleware

logger = logging.getLogger(__name__)


async def make_graph(config: Optional[RunnableConfig] = None) -> CompiledStateGraph:
    """Return a Deep Agent graph configured for Agent Chat UI rendering.
    
    The DeepAgentUIMiddleware handles:
    - Tool call/result events
    - Artifact streaming
    - Thinking/reasoning extraction via after_model hook
    
    Supports configurable model selection via config["configurable"]["model"].
    """
    # Extract model from config if provided
    model_override = None
    if config and "configurable" in config:
        model_override = config["configurable"].get("model")
        if model_override:
            logger.info(f"Using user-selected model: {model_override}")
    
    return await get_orchestrator_deep_agent(
        checkpointer=None,
        store=None,
        middleware=[DeepAgentUIMiddleware()],
        model_override=model_override,
    )

