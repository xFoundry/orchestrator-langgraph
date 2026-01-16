# Orchestrator v2 Architecture

## Overview

This document outlines a comprehensive redesign of the LangGraph orchestrator to leverage:

1. **LangGraph v1** with `create_agent` and middleware
2. **Parallelization** with Send API for concurrent research
3. **Evaluator-Optimizer Cycles** for quality gates and self-correction
4. **Mentor-Hub Tools** for direct API access to application data
5. **Long-Term Memory** with semantic search for cross-session persistence
6. **Swarm Patterns** for flexible agent handoffs

---

## Current State vs Target State

| Aspect | Current (v0.6.11) | Target (v1.x) |
|--------|-------------------|---------------|
| Agent Creation | `create_react_agent` (deprecated) | `create_agent` with middleware |
| Execution | Sequential subgraph calls | Parallel with `Send` API |
| Quality | No validation | Evaluator-optimizer cycles |
| Data Access | Cognee only | Cognee + Direct Mentor-Hub API |
| Memory | Session only (checkpointer) | Session + Long-term (store) |
| Agent Coordination | Tool wrapping | Swarm handoffs |

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Frontend (Next.js)                             │
│                      Vercel AI SDK useChat hook                          │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │ SSE Stream
┌───────────────────────────────▼─────────────────────────────────────────┐
│                        orchestrator-langgraph v2                         │
│                          FastAPI + LangGraph v1                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                    SUPERVISOR ORCHESTRATOR                         │ │
│  │  • Analyzes query complexity                                       │ │
│  │  • Routes to appropriate workflow pattern                          │ │
│  │  • Aggregates and evaluates results                                │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                    │                                     │
│          ┌─────────────────────────┼─────────────────────────┐          │
│          │                         │                         │          │
│          ▼                         ▼                         ▼          │
│  ┌──────────────┐         ┌──────────────┐         ┌──────────────┐    │
│  │   PARALLEL   │         │  SEQUENTIAL  │         │    SWARM     │    │
│  │  RESEARCHER  │         │    CHAIN     │         │   HANDOFF    │    │
│  │   (Send API) │         │  (Deep Dive) │         │  (Specialists)│    │
│  └──────────────┘         └──────────────┘         └──────────────┘    │
│          │                         │                         │          │
│          └─────────────────────────┼─────────────────────────┘          │
│                                    │                                     │
│                                    ▼                                     │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                    EVALUATOR-OPTIMIZER                             │ │
│  │  • Validates completeness                                          │ │
│  │  • Checks confidence thresholds                                    │ │
│  │  • Triggers retry cycles if needed                                 │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
├─────────────────────────────────────────────────────────────────────────┤
│                              TOOL LAYERS                                 │
│                                                                          │
│  ┌─────────────────────────────────┐  ┌──────────────┐  ┌─────────────┐ │
│  │    COGNEE TOOLS (All RAG)      │  │ MENTOR-HUB   │  │   MEMORY    │ │
│  │                                │  │    TOOLS     │  │   STORE     │ │
│  │ Graph: query_graph, find_entity│  │              │  │             │ │
│  │ Text: search_text, search_chunk│  │ get_sessions │  │ remember    │ │
│  │ RAG: search_rag, search_graph  │  │ get_mentors  │  │ recall      │ │
│  │ Summary: search_summaries      │  │ get_teams    │  │ search      │ │
│  │ Schema: get_graph_schema       │  │ get_tasks    │  │             │ │
│  └─────────────────────────────────┘  └──────────────┘  └─────────────┘ │
│                                                                          │
├─────────────────────────────────────────────────────────────────────────┤
│                           PERSISTENCE LAYER                              │
│                                                                          │
│  ┌─────────────────────────┐    ┌─────────────────────────────────────┐ │
│  │   Redis Checkpointer    │    │      InMemoryStore + Embeddings     │ │
│  │   (Short-term memory)   │    │        (Long-term semantic)         │ │
│  │                         │    │                                     │ │
│  │  • Session state        │    │  • User preferences                 │ │
│  │  • Message history      │    │  • Cross-session facts              │ │
│  │  • Tool call history    │    │  • Semantic search                  │ │
│  └─────────────────────────┘    └─────────────────────────────────────┘ │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼                               ▼
             ┌──────────────┐               ┌──────────────┐
             │   Cognee     │               │  Mentor-Hub  │
             │    API       │               │    API       │
             │ (RAG + Graph)│               │ (Live Data)  │
             └──────────────┘               └──────────────┘

Note: All RAG/knowledge graph operations go through Cognee API.
The "graph tools" (query_graph, find_entity, etc.) use Cognee's Cypher
interface - there is no separate Graphiti service.
```

---

## Core Workflow Patterns

### 1. Parallel Research Pattern (Send API)

For queries that need broad data gathering, spawn multiple researcher workers in parallel:

```python
from langgraph.types import Send
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
import operator


class ResearchState(TypedDict):
    query: str
    research_tasks: list[dict]
    research_results: Annotated[list[dict], operator.add]  # Parallel writes
    final_answer: str
    confidence: float


class WorkerState(TypedDict):
    task: dict
    research_results: Annotated[list[dict], operator.add]


def planner(state: ResearchState) -> dict:
    """Analyze query and create parallel research tasks."""
    query = state["query"]

    # Decompose into parallel tasks based on query analysis
    tasks = [
        {"type": "entity", "query": query, "focus": "entities and relationships"},
        {"type": "text", "query": query, "focus": "relevant passages"},
        {"type": "summary", "query": query, "focus": "high-level overview"},
        {"type": "mentor_hub", "query": query, "focus": "live application data"},
    ]

    return {"research_tasks": tasks}


def researcher_worker(state: WorkerState) -> dict:
    """Execute a single research task."""
    task = state["task"]
    task_type = task["type"]

    # Route to appropriate research strategy
    if task_type == "entity":
        results = entity_research(task["query"])
    elif task_type == "text":
        results = text_research(task["query"])
    elif task_type == "summary":
        results = summary_research(task["query"])
    elif task_type == "mentor_hub":
        results = mentor_hub_research(task["query"])

    return {"research_results": [{"type": task_type, "data": results}]}


def spawn_researchers(state: ResearchState) -> list[Send]:
    """Spawn parallel research workers using Send API."""
    return [
        Send("researcher_worker", {"task": task})
        for task in state["research_tasks"]
    ]


def synthesizer(state: ResearchState) -> dict:
    """Synthesize all research results into final answer."""
    results = state["research_results"]
    # Combine and synthesize results
    final_answer = synthesize_results(results)
    confidence = calculate_confidence(results)

    return {"final_answer": final_answer, "confidence": confidence}


# Build graph
builder = StateGraph(ResearchState)
builder.add_node("planner", planner)
builder.add_node("researcher_worker", researcher_worker)
builder.add_node("synthesizer", synthesizer)

builder.add_edge(START, "planner")
builder.add_conditional_edges("planner", spawn_researchers, ["researcher_worker"])
builder.add_edge("researcher_worker", "synthesizer")
builder.add_edge("synthesizer", END)

parallel_researcher = builder.compile()
```

### 2. Evaluator-Optimizer Pattern (Cycles)

Add quality gates that can trigger retry cycles:

```python
from typing import Literal


class EvaluatedState(TypedDict):
    query: str
    research_results: list[dict]
    final_answer: str
    confidence: float
    evaluation: dict
    retry_count: int


def evaluator(state: EvaluatedState) -> dict:
    """Evaluate the quality and completeness of research results."""
    results = state["research_results"]
    answer = state["final_answer"]
    query = state["query"]

    # Evaluation criteria
    evaluation = {
        "has_sufficient_data": len(results) >= 3,
        "confidence_threshold": state["confidence"] >= 0.7,
        "answers_query": check_query_coverage(query, answer),
        "has_citations": check_citations(answer, results),
    }

    evaluation["passed"] = all(evaluation.values())

    return {"evaluation": evaluation}


def should_retry(state: EvaluatedState) -> Literal["retry", "finalize"]:
    """Decide whether to retry research or finalize."""
    if state["evaluation"]["passed"]:
        return "finalize"

    if state["retry_count"] >= 2:  # Max 2 retries
        return "finalize"

    return "retry"


def retry_planner(state: EvaluatedState) -> dict:
    """Generate refined search strategy based on evaluation gaps."""
    evaluation = state["evaluation"]

    # Identify what's missing
    gaps = []
    if not evaluation["has_sufficient_data"]:
        gaps.append("Expand search with more query variations")
    if not evaluation["confidence_threshold"]:
        gaps.append("Search for more authoritative sources")
    if not evaluation["answers_query"]:
        gaps.append("Focus search on specific query aspects")

    # Generate refined tasks
    refined_tasks = generate_refined_tasks(state["query"], gaps)

    return {
        "research_tasks": refined_tasks,
        "retry_count": state["retry_count"] + 1,
    }


# Build graph with cycle
builder = StateGraph(EvaluatedState)
builder.add_node("planner", planner)
builder.add_node("researcher_worker", researcher_worker)
builder.add_node("synthesizer", synthesizer)
builder.add_node("evaluator", evaluator)
builder.add_node("retry_planner", retry_planner)
builder.add_node("finalize", finalize_response)

builder.add_edge(START, "planner")
builder.add_conditional_edges("planner", spawn_researchers, ["researcher_worker"])
builder.add_edge("researcher_worker", "synthesizer")
builder.add_edge("synthesizer", "evaluator")
builder.add_conditional_edges(
    "evaluator",
    should_retry,
    {"retry": "retry_planner", "finalize": "finalize"}
)
builder.add_conditional_edges("retry_planner", spawn_researchers, ["researcher_worker"])
builder.add_edge("finalize", END)

evaluator_optimizer = builder.compile()
```

### 3. Supervisor Router Pattern

Route queries to the most appropriate workflow based on complexity:

```python
from typing import Literal


def classify_query(state: dict) -> Literal["simple", "complex", "analytical", "action"]:
    """Classify query to determine best workflow."""
    query = state["query"].lower()

    # Simple lookups
    if any(kw in query for kw in ["what is", "who is", "tell me about", "list"]):
        return "simple"

    # Complex analysis
    if any(kw in query for kw in ["compare", "analyze", "why", "how does"]):
        return "analytical"

    # Actionable requests (needs live data)
    if any(kw in query for kw in ["schedule", "create", "my sessions", "upcoming"]):
        return "action"

    return "complex"


def route_to_workflow(state: dict) -> str:
    """Route to appropriate workflow based on classification."""
    classification = classify_query(state)

    routing = {
        "simple": "parallel_researcher",      # Quick parallel search
        "complex": "parallel_researcher",     # Broad parallel search
        "analytical": "deep_reasoning",       # Sequential analysis
        "action": "mentor_hub_workflow",      # Live API access
    }

    return routing[classification]


# Supervisor graph
supervisor_builder = StateGraph(SupervisorState)
supervisor_builder.add_node("classifier", classify_query)
supervisor_builder.add_node("parallel_researcher", parallel_researcher)
supervisor_builder.add_node("deep_reasoning", deep_reasoning_graph)
supervisor_builder.add_node("mentor_hub_workflow", mentor_hub_graph)

supervisor_builder.add_edge(START, "classifier")
supervisor_builder.add_conditional_edges(
    "classifier",
    route_to_workflow,
    {
        "parallel_researcher": "parallel_researcher",
        "deep_reasoning": "deep_reasoning",
        "mentor_hub_workflow": "mentor_hub_workflow",
    }
)
```

---

## Mentor-Hub Tools

Direct API access to application data for real-time information:

```python
# app/tools/mentor_hub_tools.py
from langchain_core.tools import tool
from typing import Optional
import httpx


MENTOR_HUB_API_URL = "http://localhost:3000/api"


@tool
async def get_upcoming_sessions(
    team_id: Optional[str] = None,
    mentor_id: Optional[str] = None,
    limit: int = 10
) -> dict:
    """
    Get upcoming mentorship sessions from the Mentor Hub application.

    Use this to answer questions about scheduled sessions, upcoming meetings,
    or to check a team's/mentor's calendar.

    Args:
        team_id: Filter by specific team
        mentor_id: Filter by specific mentor
        limit: Maximum sessions to return

    Returns:
        List of upcoming sessions with details
    """
    async with httpx.AsyncClient() as client:
        params = {"status": "Scheduled", "limit": limit}
        if team_id:
            params["teamId"] = team_id
        if mentor_id:
            params["mentorId"] = mentor_id

        response = await client.get(f"{MENTOR_HUB_API_URL}/sessions", params=params)
        return response.json()


@tool
async def get_team_details(team_id: str) -> dict:
    """
    Get detailed information about a specific team from Mentor Hub.

    Includes team members, active sessions, tasks, and project information.

    Args:
        team_id: The team's unique identifier

    Returns:
        Team details including members, sessions, tasks
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{MENTOR_HUB_API_URL}/teams/{team_id}")
        return response.json()


@tool
async def get_mentor_availability(mentor_id: str) -> dict:
    """
    Check a mentor's availability and upcoming commitments.

    Args:
        mentor_id: The mentor's contact ID

    Returns:
        Mentor's scheduled sessions and availability windows
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{MENTOR_HUB_API_URL}/mentors/{mentor_id}")
        return response.json()


@tool
async def get_team_tasks(
    team_id: str,
    status: Optional[str] = None
) -> dict:
    """
    Get action items/tasks for a team from Mentor Hub.

    Args:
        team_id: The team's unique identifier
        status: Filter by status (Open, In Progress, Completed, Blocked)

    Returns:
        List of tasks with assignees and due dates
    """
    async with httpx.AsyncClient() as client:
        params = {"teamId": team_id}
        if status:
            params["status"] = status

        response = await client.get(f"{MENTOR_HUB_API_URL}/tasks", params=params)
        return response.json()


@tool
async def get_session_feedback(session_id: str) -> dict:
    """
    Get feedback submitted for a specific session.

    Args:
        session_id: The session's unique identifier

    Returns:
        Feedback records with ratings and comments
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{MENTOR_HUB_API_URL}/sessions/{session_id}/feedback")
        return response.json()


@tool
async def search_mentors(
    expertise: Optional[list[str]] = None,
    cohort_id: Optional[str] = None,
    available_only: bool = True
) -> dict:
    """
    Search for mentors by expertise or availability.

    Use this for mentor matching recommendations.

    Args:
        expertise: List of expertise areas to match
        cohort_id: Filter by cohort participation
        available_only: Only return available mentors

    Returns:
        List of matching mentors with their profiles
    """
    async with httpx.AsyncClient() as client:
        params = {}
        if expertise:
            params["expertise"] = ",".join(expertise)
        if cohort_id:
            params["cohortId"] = cohort_id
        if available_only:
            params["status"] = "Active"

        response = await client.get(f"{MENTOR_HUB_API_URL}/mentors", params=params)
        return response.json()


@tool
async def get_cohort_overview(cohort_id: str) -> dict:
    """
    Get overview of a cohort including teams, mentors, and progress.

    Args:
        cohort_id: The cohort's unique identifier

    Returns:
        Cohort details with teams, mentors, timeline
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{MENTOR_HUB_API_URL}/cohorts/{cohort_id}")
        return response.json()
```

---

## Long-Term Memory with Semantic Search

```python
# app/persistence/memory_store.py
from langgraph.store.memory import InMemoryStore
from langchain.embeddings import init_embeddings
from langgraph.graph import StateGraph
from langgraph.store.base import BaseStore
from langgraph.config import RunnableConfig


def create_memory_store() -> InMemoryStore:
    """Create a memory store with semantic search capabilities."""
    embeddings = init_embeddings("openai:text-embedding-3-small")

    store = InMemoryStore(
        index={
            "embed": embeddings,
            "dims": 1536,
            "fields": ["content", "summary"],  # Fields to embed
        }
    )

    return store


# Usage in nodes
def call_model_with_memory(
    state: dict,
    config: RunnableConfig,
    *,
    store: BaseStore
) -> dict:
    """Model call that incorporates long-term memory."""
    user_id = config["configurable"]["user_id"]
    namespace = (user_id, "memories")

    # Semantic search for relevant memories
    query = state["messages"][-1].content
    memories = store.search(namespace, query=query, limit=5)

    # Format memories for context
    memory_context = "\n".join([
        f"- {m.value['content']}" for m in memories
    ])

    # Include memories in prompt
    enhanced_messages = state["messages"].copy()
    if memory_context:
        enhanced_messages.insert(0, {
            "role": "system",
            "content": f"Relevant user context:\n{memory_context}"
        })

    # Call model
    response = model.invoke(enhanced_messages)

    return {"messages": [response]}


def save_to_memory(
    state: dict,
    config: RunnableConfig,
    *,
    store: BaseStore
) -> dict:
    """Save important facts to long-term memory."""
    user_id = config["configurable"]["user_id"]
    namespace = (user_id, "memories")

    # Extract facts worth remembering
    facts = extract_memorable_facts(state)

    for fact in facts:
        store.put(
            namespace,
            key=fact["id"],
            value={
                "content": fact["content"],
                "summary": fact["summary"],
                "source": fact["source"],
                "timestamp": fact["timestamp"],
            }
        )

    return state


# Compile graph with store
builder = StateGraph(...)
graph = builder.compile(
    checkpointer=redis_checkpointer,  # Short-term
    store=memory_store,                # Long-term
)
```

---

## Migration to create_agent with Middleware

```python
# app/graphs/orchestrator_v2.py
from langchain.agents import create_agent
from langchain.agents.middleware import (
    SummarizationMiddleware,
    HumanInTheLoopMiddleware,
)


def create_orchestrator_v2(
    checkpointer,
    store,
    user_context: dict = None,
) -> CompiledGraph:
    """Create orchestrator using LangChain v1 create_agent."""

    # All tools
    tools = [
        # Cognee tools
        search_graph,
        search_rag,
        search_chunks,
        search_summaries,
        # Graphiti tools
        query_graph,
        find_entity,
        search_text,
        get_graph_schema,
        # Mentor-Hub tools
        get_upcoming_sessions,
        get_team_details,
        get_mentor_availability,
        get_team_tasks,
        search_mentors,
        # Memory tools
        remember,
        recall,
    ]

    # Middleware for enhanced functionality
    middleware = [
        # Summarize long conversations
        SummarizationMiddleware(
            model="gpt-4o-mini",
            trigger={"tokens": 4000}
        ),
        # Tool selection based on query type
        ToolSelectorMiddleware(),
    ]

    # Create agent with middleware
    agent = create_agent(
        model="gpt-4o",
        tools=tools,
        system_prompt=ORCHESTRATOR_PROMPT,
        middleware=middleware,
        name="orchestrator",
    )

    # Compile with persistence
    return agent.compile(
        checkpointer=checkpointer,
        store=store,
    )
```

---

## Updated Project Structure

```
orchestrator-langgraph/
├── app/
│   ├── graphs/
│   │   ├── orchestrator_v2.py      # Main supervisor with create_agent
│   │   ├── workflows/
│   │   │   ├── parallel_researcher.py   # Send API parallelization
│   │   │   ├── evaluator_optimizer.py   # Quality cycles
│   │   │   ├── deep_reasoning.py        # Sequential analysis
│   │   │   └── mentor_hub_workflow.py   # Live API workflow
│   │   ├── state.py                     # Unified state schemas
│   │   └── routing.py                   # Query classification
│   │
│   ├── tools/
│   │   ├── cognee_tools.py         # Cognee RAG tools
│   │   ├── graphiti_tools.py       # Graphiti graph tools
│   │   ├── mentor_hub_tools.py     # NEW: Direct API tools
│   │   ├── memory_tools.py         # Remember/recall tools
│   │   └── evaluation_tools.py     # Quality assessment tools
│   │
│   ├── persistence/
│   │   ├── redis.py                # Checkpointer (short-term)
│   │   └── memory_store.py         # NEW: Semantic store (long-term)
│   │
│   ├── middleware/
│   │   ├── tool_selector.py        # Dynamic tool selection
│   │   ├── confidence_gate.py      # Quality threshold middleware
│   │   └── citation_tracker.py     # Source attribution
│   │
│   └── api/
│       └── routes/
│           ├── chat_stream.py      # SSE streaming endpoint
│           └── internal/
│               └── mentor_hub.py   # Internal API proxy
```

---

## Implementation Priority

### Phase 1: Foundation (Week 1)
1. Upgrade to LangGraph v1 / LangChain v1
2. Migrate from `create_react_agent` to `create_agent`
3. Add basic middleware support

### Phase 2: Parallelization (Week 2)
4. Implement parallel researcher with Send API
5. Add supervisor routing logic
6. Test parallel execution performance

### Phase 3: Quality Gates (Week 3)
7. Implement evaluator-optimizer cycles
8. Add confidence scoring
9. Add retry logic with refined queries

### Phase 4: Mentor-Hub Integration (Week 4)
10. Create mentor-hub API tools
11. Add authentication passthrough
12. Implement live data workflows

### Phase 5: Long-Term Memory (Week 5)
13. Set up InMemoryStore with embeddings
14. Implement remember/recall tools
15. Add cross-session fact persistence

---

## Dependencies Update

```toml
[project]
dependencies = [
    # LangGraph v1
    "langgraph>=1.0.0",
    "langgraph-checkpoint-redis>=1.0.0",

    # LangChain v1
    "langchain>=1.0.0",
    "langchain-core>=1.0.0",
    "langchain-openai>=1.0.0",

    # Swarm (optional)
    "langgraph-swarm>=1.0.0",

    # Embeddings for memory
    "openai>=1.0.0",

    # Existing
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.32.0",
    "httpx>=0.27.0",
    "redis>=5.0.0",
]
```

---

## Performance Expectations

| Metric | Current | With Parallelization | With Caching |
|--------|---------|---------------------|--------------|
| Simple Query | 3-5s | 1-2s | <1s |
| Complex Query | 8-15s | 3-5s | 2-3s |
| Analytical Query | 15-30s | 5-10s | 3-5s |
| Quality Score | ~70% | ~85% | ~90% |

---

## Next Steps

1. **Create upgrade branch**: `feature/langgraph-v2`
2. **Test LangGraph v1 compatibility** with current tools
3. **Implement parallel researcher** as proof-of-concept
4. **Add mentor-hub tool** for `get_upcoming_sessions`
5. **Benchmark** before/after parallelization
