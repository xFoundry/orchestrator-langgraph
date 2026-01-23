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
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore

from app.config import get_settings, Settings
from app.tools.tool_registry import create_default_registry

logger = logging.getLogger(__name__)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _create_anthropic_model(
    settings: Settings,
    model_name: str,
    model_kwargs: dict[str, Any],
    log: logging.Logger,
):
    """Create an Anthropic model, routing through OpenRouter if configured.
    
    OpenRouter provides access to Anthropic models via an OpenAI-compatible API.
    This allows using a single OpenRouter API key instead of a direct Anthropic key.
    
    OpenRouter supports reasoning tokens for Anthropic models via the `reasoning` parameter
    in extra_body, which is different from Anthropic's native `thinking` parameter.
    
    Args:
        settings: Application settings
        model_name: The Anthropic model name (e.g., "claude-sonnet-4.5")
        model_kwargs: Additional kwargs to pass to the model
        log: Logger instance
    
    Returns:
        Configured chat model instance
    """
    # Check if we should use OpenRouter for Anthropic models
    if settings.use_openrouter_for_anthropic and settings.openrouter_api_key:
        # OpenRouter uses OpenAI-compatible API with model names like "anthropic/claude-..."
        openrouter_model_name = f"anthropic/{model_name}"
        
        log.info(f"Routing Anthropic model through OpenRouter: {openrouter_model_name}")
        
        # Filter out Anthropic-specific kwargs that need special handling for OpenRouter
        anthropic_only_params = {"thinking", "anthropic_metadata", "anthropic_headers"}
        openrouter_kwargs = {k: v for k, v in model_kwargs.items() if k not in anthropic_only_params}
        
        # Convert Anthropic's "thinking" parameter to OpenRouter's "reasoning" via extra_body
        # OpenRouter supports reasoning tokens for Anthropic models with max_tokens or effort
        extra_body = {}
        if "thinking" in model_kwargs and settings.deep_agent_enable_thinking:
            thinking_config = model_kwargs["thinking"]
            # OpenRouter uses reasoning.max_tokens for Anthropic models (min 1024)
            budget_tokens = thinking_config.get("budget_tokens", settings.deep_agent_thinking_budget)
            extra_body["reasoning"] = {
                "max_tokens": max(budget_tokens, 1024),  # OpenRouter minimum is 1024
            }
            log.info(f"Enabled OpenRouter reasoning for Anthropic with max_tokens={budget_tokens}")
        elif settings.deep_agent_enable_thinking:
            # Enable reasoning with default budget if thinking is enabled but no config provided
            extra_body["reasoning"] = {
                "max_tokens": max(settings.deep_agent_thinking_budget, 1024),
            }
            log.info(f"Enabled OpenRouter reasoning with default budget: {settings.deep_agent_thinking_budget}")
        
        # Use ChatOpenAI with OpenRouter's base URL
        return ChatOpenAI(
            model=openrouter_model_name,
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
            streaming=True,
            # OpenRouter-specific headers
            default_headers={
                "HTTP-Referer": "https://xfoundry.org",
                "X-Title": "XFoundry Agent",
            },
            # Pass reasoning config via extra_body for OpenRouter
            model_kwargs={"extra_body": extra_body} if extra_body else {},
            **openrouter_kwargs,
        )
    elif settings.anthropic_api_key:
        # Use direct Anthropic API - supports all Anthropic-specific features like extended thinking
        log.info(f"Using direct Anthropic API for: {model_name}")
        return init_chat_model(
            f"anthropic:{model_name}",
            api_key=settings.anthropic_api_key,
            **model_kwargs,
        )
    else:
        raise ValueError(
            "No API key configured for Anthropic models. "
            "Set either OPENROUTER_API_KEY (recommended) or ANTHROPIC_API_KEY."
        )


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
4. **Tools**: Knowledge graph search, live Mentor Hub data, Outline documents, and web research.
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

### Outline (Docs Knowledge Base)
For internal documentation and policy references:
- `outline_collections_list`: List collections (folders/categories)
- `outline_documents_search`: Search documents by keywords/phrases
- `outline_documents_info`: Read full document content (use IDs from search/list)

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
   - For internal documentation/policies: Use Outline tools
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

## Plan + Approval Workflow

For any multi-step or structured deliverable (tables, calendars, multi-week schedules, reports):

1. **Plan first**:
   - Use `write_todos` to outline tasks.
   - Write a user-facing plan to `/artifacts/<slug>-plan.md`.
2. **Request approval**:
   - Call `request_clarifications` with options like "approve", "request changes", "cancel".
   - Wait for the user's response before continuing.
3. **Execute after approval**:
   - Create the final deliverable as `/artifacts/<slug>.md`.
   - Summarize briefly in chat and reference the artifact.

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
    middleware: Optional[list[Any]] = None,
    model_override: Optional[str] = None,
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
        model_override: Optional model name to use instead of default (from UI selection)

    Returns:
        Compiled Deep Agent graph ready for execution
    """
    settings = get_settings()
    user_id = auth0_id or "anonymous"

    logger.info(f"Creating Deep Agent for user: {user_id[:20]}...")

    # =========================================================================
    # Initialize Model with Thinking Support
    # =========================================================================
    # Use model_override if provided (from UI selection), otherwise fall back to settings
    model_name = model_override or settings.deep_agent_model or settings.default_orchestrator_model
    logger.info(f"Using model: {model_name}")
    model_name_lower = model_name.lower()
    enable_thinking = settings.deep_agent_enable_thinking

    # Determine provider and configure thinking
    is_claude = "claude" in model_name_lower or "anthropic" in model_name_lower
    is_gpt5 = "gpt-5" in model_name_lower or "gpt5" in model_name_lower
    is_o_series = model_name_lower.startswith("o1") or model_name_lower.startswith("o3")

    model_kwargs: dict[str, Any] = {}
    # Note: Thinking extraction is handled by DeepAgentUIMiddleware.aafter_model()

    # Configure thinking based on model type
    if enable_thinking:
        if is_claude:
            # Claude extended thinking
            model_kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": settings.deep_agent_thinking_budget,
            }
            logger.info(f"Enabled Claude thinking with {settings.deep_agent_thinking_budget} token budget")
        elif is_gpt5 or is_o_series:
            # GPT-5.x and o-series: use reasoning dict with effort AND summary
            # This enables the Responses API which returns reasoning summaries
            model_kwargs["reasoning"] = {
                "effort": settings.deep_agent_reasoning_effort,
                "summary": "auto",  # Request reasoning summary in responses
            }
            logger.info(f"Enabled {model_name} reasoning with effort={settings.deep_agent_reasoning_effort}, summary=auto")

    # Support different model providers
    # Use ChatOpenAI directly for GPT-5.x to ensure reasoning parameter works
    # Route Anthropic models through OpenRouter if configured
    if ":" in model_name:
        provider = model_name.split(":", 1)[0].lower()
        actual_model_name = model_name.split(":", 1)[1]
        
        if provider == "anthropic":
            model = _create_anthropic_model(settings, actual_model_name, model_kwargs, logger)
        elif provider == "openai":
            # Use ChatOpenAI directly for better reasoning support
            if is_gpt5 or is_o_series:
                model = ChatOpenAI(
                    model=actual_model_name,
                    api_key=settings.openai_api_key,
                    **model_kwargs,
                )
                logger.info(f"Using ChatOpenAI directly for {actual_model_name} with reasoning support")
            else:
                model = init_chat_model(
                    model_name,
                    api_key=settings.openai_api_key,
                    **model_kwargs,
                )
        else:
            model = init_chat_model(model_name, **model_kwargs)
    elif is_claude:
        model = _create_anthropic_model(settings, model_name, model_kwargs, logger)
    elif is_gpt5 or is_o_series:
        # Use ChatOpenAI directly for GPT-5.x/o-series models
        model = ChatOpenAI(
            model=model_name,
            api_key=settings.openai_api_key,
            **model_kwargs,
        )
        logger.info(f"Using ChatOpenAI directly for {model_name} with reasoning support")
    else:
        # Default to OpenAI via init_chat_model
        model = init_chat_model(
            f"openai:{model_name}",
            api_key=settings.openai_api_key,
            **model_kwargs,
        )

    # =========================================================================
    # Get Tools from Registry
    # =========================================================================
    registry = await create_default_registry(user_id=auth0_id)

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
        Create a SimpleScopedStoreBackend that handles all path routing internally.

        Path -> Namespace mapping (handled by SimpleScopedStoreBackend):
        - /shared/* -> tenant-scoped (all staff can access)
        - /artifacts/saved/* -> user-scoped (persists across threads)
        - /memories/* -> user-scoped (persists across threads)
        - /artifacts/* -> thread-scoped
        - /context/* -> thread-scoped

        Namespace hierarchy:
        - tenant/shared/ for /shared/*
        - tenant/users/{user_id}/saved/ for /artifacts/saved/*
        - tenant/users/{user_id}/memories/ for /memories/*
        - tenant/users/{user_id}/{thread_id}/artifacts/ for /artifacts/*
        - tenant/users/{user_id}/{thread_id}/context/ for /context/*
        """
        try:
            from deepagents.backends import StateBackend
            from app.backends.simple_scoped_store_backend import SimpleScopedStoreBackend

            if not filesystem_enabled:
                return StateBackend(runtime)

            # Use SimpleScopedStoreBackend directly - it handles all path routing internally
            # Don't use CompositeBackend as it strips prefixes before passing to the backend,
            # which breaks SimpleScopedStoreBackend's path-to-scope mapping
            if store:
                logger.debug("Using SimpleScopedStoreBackend for persistent filesystem")
                scoped_backend = SimpleScopedStoreBackend(runtime)
                scoped_backend._store = store
                return scoped_backend

            # Fall back to ephemeral if no store
            return StateBackend(runtime)

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

        interrupt_on = None
        if settings.deep_agent_enable_hitl and checkpointer:
            interrupt_on = {
                "write_file": {"allowed_decisions": ["approve", "edit", "reject"]},
                "edit_file": {"allowed_decisions": ["approve", "edit", "reject"]},
            }
        elif settings.deep_agent_enable_hitl and not checkpointer:
            logger.warning("Deep Agent HITL enabled but no checkpointer provided; disabling HITL.")

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
        if interrupt_on:
            agent_kwargs["interrupt_on"] = interrupt_on

        if middleware:
            agent_kwargs["middleware"] = middleware

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
    middleware: Optional[list[Any]] = None,
    model_override: Optional[str] = None,
) -> CompiledStateGraph:
    """
    Get a Deep Agent orchestrator instance.

    This is the main entry point for the chat stream route.
    Alias for create_orchestrator_deep_agent for consistency with other orchestrators.
    
    Args:
        model_override: Optional model name to use instead of default (from UI selection)
    """
    return await create_orchestrator_deep_agent(
        checkpointer=checkpointer,
        store=store,
        user_context=user_context,
        auth0_id=auth0_id,
        selected_tools=selected_tools,
        middleware=middleware,
        model_override=model_override,
    )
