"""
Deep Agent Orchestrator - Built on LangGraph's Deep Agents library.

This orchestrator leverages the deepagents library for:
- TodoListMiddleware: Planning/task decomposition (write_todos tool)
- FilesystemMiddleware: Context management (ls, read_file, write_file, edit_file, glob, grep)
- SubAgentMiddleware: Task delegation with context isolation (task tool)
- CompositeBackend: Hybrid ephemeral + persistent storage

The agent automatically:
1. Plans complex tasks using write_todos
2. Offloads large context to the filesystem
3. Delegates deep research to specialized subagents
4. Summarizes old conversation history when context gets full
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from langchain.chat_models import init_chat_model
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore

from app.config import get_settings
from app.tools.tool_registry import create_default_registry

logger = logging.getLogger(__name__)

# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """You are an AI assistant for Mentor Hub, a mentorship platform that connects students with mentors.

## Your Capabilities

1. **Planning**: Use `write_todos` to break complex tasks into steps and track progress.
   - If you expect to call 2+ tools, create a todo list before the first tool call.
   - Mark items in progress/completed as you go.
2. **Context Management**: Use filesystem tools to store/retrieve large context.
3. **Delegation**: Use the `task` tool to spawn subagents for specialized deep work.
4. **Tools**: Knowledge graph search, live Mentor Hub data, and web research.
5. **Clarifications**: Use `request_clarifications` when key details are missing.

## Available Tool Categories

### Knowledge Graph (Cognee)
For historical context, relationships, and program information:
- `find_entity`: Find entities and their relationships by name
- `search_text`: Semantic search over document chunks
- `query_graph`: Run Cypher queries for complex relationships
- `search_chunks`: Raw text passage retrieval
- `search_summaries`: Document-level overviews
- `search_graph`: Graph-enhanced search with synthesis
- `search_rag`: Full RAG pipeline

### Mentor Hub (Live Data)
For current/real-time information:
- `get_mentor_hub_sessions`: Session schedules, agendas, summaries
- `get_mentor_hub_team`: Team details and members
- `search_mentor_hub_mentors`: Find mentors by expertise
- `get_mentor_hub_tasks`: Team tasks and blockers

### Web Research (Firecrawl)
For external information:
- `firecrawl_search`: Web search
- `firecrawl_scrape`: Scrape a specific URL
- `firecrawl_map`: List URLs on a website
- `firecrawl_extract`: Extract structured data

### Clarifications (User Input)
For missing or ambiguous details:
- `request_clarifications`: Ask multiple-choice questions with an "other" option

## Guidelines

1. **Choose the right tools**:
   - For current/live data (schedules, tasks, members): Use Mentor Hub tools
   - For historical context, relationships, program info: Use Cognee knowledge graph tools
   - For external research: Use Firecrawl web tools

2. **Plan before acting**: For any task that requires multiple tool calls or multi-step work,
   use `write_todos` first. Update the todo list as you complete steps.

3. **Verify entity names**: When querying Mentor Hub, use `find_entity` first to get exact names.

4. **Delegate deep research**: For in-depth analysis requiring many searches, spawn a subagent using `task`.

5. **Cite your sources**: Reference where information came from in your responses.

6. **Keep context clean**: For tasks with large intermediate results, delegate to subagents.
7. **Ask before guessing**: When required details are missing, call `request_clarifications`.
   - Ask 1-3 questions at once (bundle them).
   - Use multiple choice or multi-select options.
   - Include an "other" option for custom input.
   - Do not ask follow-up clarifying questions in plain text; use the tool again.
   - Pause and wait for the user's response before proceeding.

## Artifacts

When creating substantial output (documents, reports, plans, notes), write them as artifacts:

- **Use `/artifacts/filename.md`** for:
  - Documents longer than ~20 lines
  - Content the user will want to edit or reference later
  - Plans, reports, analyses, drafts, summaries

- **Output directly to chat** for:
  - Brief answers and explanations
  - Quick summaries under 15-20 lines
  - Conversational responses

When updating an artifact, use `edit_file` with targeted string replacement (find old text, replace with new) rather than rewriting the entire file.

To save an artifact permanently (across conversations), copy or move it to `/artifacts/saved/filename.md`.
"""

# =============================================================================
# SUBAGENT CONFIGURATIONS
# =============================================================================

def get_subagent_configs(
    settings,
    registry,
) -> list[dict]:
    """
    Define specialized subagents for delegation.

    Each subagent has:
    - name: Unique identifier for the task tool
    - description: When to use this subagent (helps main agent decide)
    - system_prompt: Detailed instructions for the subagent
    - tools: Explicit tool list (keep minimal and task-focused)
    - model: Optional model override
    """
    subagents: list[dict[str, Any]] = []

    research_tools = registry.get_tools("cognee")
    if research_tools:
        subagents.append(
            {
                "name": "research-agent",
                "description": (
                    "Use for in-depth research requiring multiple searches and synthesis. "
                    "Good for complex questions about program history, relationships between "
                    "entities, or topics requiring cross-referencing multiple sources."
                ),
                "system_prompt": """You are a research specialist for Mentor Hub. Your job is to:

1. Break down the research question into searchable queries
2. Search multiple sources (knowledge graph, documents, summaries)
3. Cross-reference findings to verify accuracy
4. Synthesize a comprehensive but concise answer

Use these strategies:
- Start with find_entity to verify entity names
- Use search_text for detailed passages
- Use search_graph for synthesized overviews
- Query relationships with query_graph when needed

Output format:
- Summary (2-3 paragraphs max)
- Key findings (bullet points)
- Sources used

Keep your response under 500 words to maintain clean context for the main agent.""",
                "tools": research_tools,
            }
        )

    web_research_tools = registry.get_tools("firecrawl")
    if web_research_tools:
        subagents.append(
            {
                "name": "web-researcher",
                "description": (
                    "Use for questions requiring external web research. "
                    "Good for competitive analysis, industry trends, news, "
                    "or information not in the knowledge base."
                ),
                "system_prompt": """You are a web research specialist. Your job is to:

1. Search the web for relevant information
2. Scrape and analyze relevant pages
3. Extract key facts and cite sources
4. Provide a summary with links

Strategies:
- Use firecrawl_search to find relevant pages
- Use firecrawl_scrape to get page content
- Use firecrawl_extract for structured data extraction

Output format:
- Summary of findings
- Key facts (bullet points)
- Source URLs

Keep your response focused and cite all sources.""",
                "tools": web_research_tools,
            }
        )

    mentor_tools = registry.get_tools("mentor_hub", "cognee")
    if mentor_tools:
        subagents.append(
            {
                "name": "mentor-matcher",
                "description": (
                    "Use for finding and recommending mentors based on expertise, "
                    "availability, or team needs. Specialized in mentor-student matching."
                ),
                "system_prompt": """You are a mentor matching specialist. Your job is to:

1. Understand the required expertise or needs
2. Search for mentors with relevant backgrounds
3. Analyze mentor bios and experience
4. Provide ranked recommendations with rationale

Use these tools:
- search_mentor_hub_mentors: Find mentors by expertise
- find_entity: Get detailed mentor info from knowledge graph
- search_text: Find additional context about mentors

Output format:
- Recommended mentors (ranked)
- For each: Name, expertise match, rationale
- Any caveats or considerations

Be specific about why each mentor is a good match.""",
                "tools": mentor_tools,
            }
        )

    return subagents


# =============================================================================
# DEEP AGENT FACTORY
# =============================================================================

async def create_orchestrator_deep_agent(
    checkpointer: Optional[BaseCheckpointSaver] = None,
    store: Optional[BaseStore] = None,
    user_context: Optional[dict[str, Any]] = None,
    auth0_id: Optional[str] = None,
    selected_tools: Optional[list[str]] = None,
) -> CompiledStateGraph:
    """
    Create a Deep Agent orchestrator with all capabilities.

    This function creates a LangGraph agent using the deepagents library,
    which provides built-in planning, context management, and subagent capabilities.

    Args:
        checkpointer: LangGraph checkpointer for session persistence
        user_context: User details for personalization (name, role, teams)
        auth0_id: User ID for memory isolation
        selected_tools: Optional list of tool groups to enable (e.g., ["cognee", "firecrawl"])

    Returns:
        Compiled Deep Agent graph ready for execution
    """
    settings = get_settings()
    user_id = auth0_id or "anonymous"

    logger.info(f"Creating Deep Agent for user: {user_id[:20]}...")

    # =========================================================================
    # Initialize Model
    # =========================================================================
    model_name = settings.deep_agent_model or settings.default_orchestrator_model
    model_name_lower = model_name.lower()

    # Support different model providers
    if ":" in model_name:
        provider = model_name.split(":", 1)[0].lower()
        if provider == "anthropic":
            model = init_chat_model(
                model_name,
                api_key=settings.anthropic_api_key,
            )
        elif provider == "openai":
            model = init_chat_model(
                model_name,
                api_key=settings.openai_api_key,
            )
        else:
            model = init_chat_model(model_name)
    elif "claude" in model_name_lower or "anthropic" in model_name_lower:
        model = init_chat_model(
            f"anthropic:{model_name}",
            api_key=settings.anthropic_api_key,
        )
    else:
        # Default to OpenAI
        model = init_chat_model(
            f"openai:{model_name}",
            api_key=settings.openai_api_key,
        )

    # =========================================================================
    # Get Tools from Registry
    # =========================================================================
    registry = create_default_registry(user_id=auth0_id)

    # Use all available tools; selected_tools only influences priority guidance.
    tools = registry.get_tools()
    if selected_tools:
        logger.info(f"Preferred tool groups: {selected_tools}")

    logger.info(f"Loaded {len(tools)} tools")

    # =========================================================================
    # Configure Backend (Ephemeral + Persistent)
    # =========================================================================
    filesystem_enabled = settings.deep_agent_enable_filesystem
    store = store if filesystem_enabled else None

    def make_backend(runtime):
        """
        Create a CompositeBackend that routes:
        - /memories/* -> Persistent Redis store (cross-conversation)
        - /artifacts/saved/* -> Persistent Redis store (user-saved artifacts)
        - /artifacts/* -> Ephemeral StateBackend (working artifacts)
        - /* -> Ephemeral StateBackend (current session only)
        """
        try:
            from deepagents.backends import CompositeBackend, StateBackend, StoreBackend

            # Default ephemeral backend for working files
            ephemeral = StateBackend(runtime)

            if not filesystem_enabled:
                return ephemeral

            # Try to set up persistent backend for memories and saved artifacts
            if store:
                logger.debug("Using store backend for /memories/ and /artifacts/saved/ persistence")
                return CompositeBackend(
                    default=ephemeral,
                    routes={
                        "/memories/": StoreBackend(runtime),
                        "/artifacts/saved/": StoreBackend(runtime),
                        # /artifacts/ without /saved/ remains ephemeral (default)
                    },
                )

            # Fall back to all ephemeral
            return ephemeral

        except ImportError as e:
            logger.warning(f"Could not import deepagents backends: {e}")
            return None

    # =========================================================================
    # Build Context-Aware System Prompt
    # =========================================================================
    system_prompt = SYSTEM_PROMPT

    if user_context:
        context_lines = []
        if user_context.get("name"):
            context_lines.append(f"- **User**: {user_context['name']}")
        if user_context.get("role"):
            context_lines.append(f"- **Role**: {user_context['role']}")
        if user_context.get("teams"):
            teams = user_context["teams"]
            if isinstance(teams, list):
                context_lines.append(f"- **Teams**: {', '.join(teams)}")
            else:
                context_lines.append(f"- **Teams**: {teams}")
        if user_context.get("cohort"):
            context_lines.append(f"- **Cohort**: {user_context['cohort']}")

        if context_lines:
            system_prompt += "\n\n## Current User Context\n" + "\n".join(context_lines)

    if selected_tools:
        preferred = ", ".join(selected_tools)
        system_prompt += (
            "\n\n## Tool Preference\n"
            f"The user requested prioritizing these tool groups when relevant: {preferred}.\n"
            "You may use other tools if needed to answer correctly."
        )

    # =========================================================================
    # Create the Deep Agent
    # =========================================================================
    try:
        from deepagents import create_deep_agent

        subagents = []
        if settings.deep_agent_enable_subagents:
            subagents = get_subagent_configs(settings, registry)
        else:
            logger.info("Deep Agent subagents disabled by configuration")

        agent_kwargs = {
            "model": model,
            "system_prompt": system_prompt,
            "tools": tools,
            "subagents": subagents,
            "checkpointer": checkpointer,
            "name": "mentor-hub-agent",
        }

        if filesystem_enabled:
            agent_kwargs["backend"] = make_backend
        if store:
            agent_kwargs["store"] = store

        agent = create_deep_agent(**agent_kwargs)

        logger.info(
            f"Created Deep Agent with {len(tools)} tools and {len(subagents)} subagents"
        )

        return agent

    except ImportError as e:
        logger.error(f"deepagents library not installed: {e}")
        raise ImportError(
            "The deepagents library is required. Install it with: pip install deepagents"
        ) from e


# =============================================================================
# CONVENIENCE FUNCTION FOR ROUTE HANDLER
# =============================================================================

async def get_orchestrator_deep_agent(
    checkpointer: Optional[BaseCheckpointSaver] = None,
    store: Optional[BaseStore] = None,
    user_context: Optional[dict[str, Any]] = None,
    auth0_id: Optional[str] = None,
    selected_tools: Optional[list[str]] = None,
) -> CompiledStateGraph:
    """
    Get a Deep Agent orchestrator instance.

    This is the main entry point for the chat stream route.
    Alias for create_orchestrator_deep_agent for consistency with other orchestrators.
    """
    return await create_orchestrator_deep_agent(
        checkpointer=checkpointer,
        store=store,
        user_context=user_context,
        auth0_id=auth0_id,
        selected_tools=selected_tools,
    )
