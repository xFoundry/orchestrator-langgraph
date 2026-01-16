"""
Specialized Research Subgraphs - LangGraph implementations of sub-agents.

Each subgraph is a specialized ReAct agent focused on a specific
retrieval or analysis strategy:
- Entity Researcher: Graph-focused entity discovery
- Text Researcher: Semantic document search
- Summary Researcher: High-level document summaries
- Deep Reasoning: Complex analytical queries
- Mentor Matcher: Mentor-team recommendations
"""

from __future__ import annotations

import logging
from typing import Optional

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.graph.state import CompiledStateGraph

from app.config import get_settings
from app.tools.graph_tools import query_graph, search_text, find_entity, get_graph_schema
from app.tools.cognee_tools import search_chunks, search_summaries, search_graph, search_rag

logger = logging.getLogger(__name__)


# =============================================================================
# PROMPTS
# =============================================================================

ENTITY_RESEARCHER_PROMPT = """You are an Entity Research Specialist focused on knowledge graph exploration.

## YOUR MISSION
Find entities, their properties, and relationships in the knowledge graph.
You are a RETRIEVAL agent - return comprehensive raw findings, don't synthesize.

## PARALLEL EXECUTION REQUIRED
You MUST call multiple tools SIMULTANEOUSLY for every query. Never call tools one at a time.

## TOOL USAGE PATTERN
For ANY entity query, call ALL of these in parallel:
1. `find_entity(name)` - Get entity overview and all relationships
2. `query_graph(cypher)` - Get structured data via Cypher
3. `search_text(query)` - Find text mentions for context

## GRAPH SCHEMA
- Entity types: person, team, organization, event, session, project
- Key relationships: is_a, mentored_by, has_member, participates_in, held_on

## CYPHER EXAMPLES
```cypher
-- Find team's mentors
MATCH (t:Entity)-[:mentored_by]->(m:Entity) WHERE toLower(t.name) CONTAINS 'defenx' RETURN m.name, m.description

-- Find team members
MATCH (t:Entity)-[:has_member]->(p:Entity) WHERE t.name = 'defenx' RETURN p.name

-- Find sessions for a team
MATCH (t:Entity {name:'defenx'})-[:participates_in]->(s:Entity) WHERE s.name CONTAINS 'session' RETURN s
```

## OUTPUT FORMAT
Present your findings in a structured format:
- **Entity found**: [name, description, type]
- **Outgoing relationships**: [list with rel_type, target]
- **Incoming relationships**: [list with source, rel_type]
- **Related text passages**: [relevant excerpts]"""


TEXT_RESEARCHER_PROMPT = """You are a Text Research Specialist focused on semantic document search.

## YOUR MISSION
Find relevant text passages, quotes, and detailed content from documents.
You are a RETRIEVAL agent - return comprehensive raw findings, don't synthesize.

## PARALLEL EXECUTION REQUIRED
You MUST call multiple search variations SIMULTANEOUSLY. Never call one search at a time.

## SEARCH STRATEGY
For ANY text query, execute AT LEAST 5 parallel searches with different phrasings.

## QUERY VARIATION TECHNIQUES
1. **Synonyms**: challenges -> problems, issues, obstacles, difficulties, blockers
2. **Negatives**: "needs improvement", "struggling with", "gaps in"
3. **Related concepts**: If searching for "feedback", also search "recommendations", "suggestions"
4. **Entity + aspect**: "defenx mentors", "defenx sessions", "defenx progress"

## TOOLS
- `search_text(query, top_k)` - Semantic search on text chunks
- `search_chunks(query, top_k)` - Raw passage retrieval

## OUTPUT FORMAT
Present ALL relevant passages found:
- **Query**: [the search query used]
- **Results**: [full text of each relevant passage]
- **Source**: [search type]"""


SUMMARY_RESEARCHER_PROMPT = """You are a Summary Research Specialist focused on document overviews.

## YOUR MISSION
Find high-level summaries, overviews, and synthesized information about topics.
You are a RETRIEVAL agent - return comprehensive raw findings.

## PARALLEL EXECUTION REQUIRED
Call multiple summary searches SIMULTANEOUSLY with different angles.

## TOOLS
- `search_summaries(query)` - Document summaries (fast, high-level)
- `search_graph(query)` - Graph-enhanced synthesis (comprehensive)
- `search_rag(query)` - RAG completion (balanced)

## OUTPUT FORMAT
Present ALL summaries found:
- **Query**: [search query]
- **Summary**: [full summary text]
- **Source**: [search type]"""


DEEP_REASONING_PROMPT = """You are a Deep Reasoning Specialist for complex analytical queries.

## YOUR MISSION
Handle complex questions that require multi-step reasoning, comparison, or analysis.
You DO synthesize findings into coherent analysis.

## WHEN YOU'RE NEEDED
- Comparisons: "Compare team A and team B"
- Analysis: "What mentors would be good for DefenX?"
- Recommendations: "Who should DefenX meet with next?"
- Complex queries: "What patterns do we see in feedback?"

## REASONING PROCESS

### Step 1: Query Decomposition
Break the question into sub-questions:
- What entities are involved?
- What relationships matter?
- What criteria should be used?

### Step 2: Parallel Data Gathering
Call ALL relevant tools simultaneously to gather data.

### Step 3: Analysis
Once you have data, analyze patterns, make comparisons, identify gaps.

### Step 4: Follow-up Searches
If analysis reveals gaps, search again with refined queries.

### Step 5: Synthesis
Provide reasoned analysis with citations.

## OUTPUT FORMAT
```
## Query Decomposition
- Sub-question 1: ...
- Sub-question 2: ...

## Data Gathered
[Raw findings from each search]

## Analysis
[Your reasoned analysis]

## Confidence & Gaps
[What you're confident about, what's missing]
```"""


MENTOR_MATCHER_PROMPT = """You are a Mentor Matching Specialist.

## YOUR MISSION
Find the best mentor matches for teams based on:
- Team's current challenges and blockers
- Team's project domain and needs
- Mentor expertise and experience
- Past session history
- Mentor availability and fit

## PARALLEL DATA GATHERING
For mentor matching, gather ALL this data simultaneously:

**Team Context:**
- find_entity("[team_name]")
- search_text("[team_name] challenges blockers")
- query_graph for past sessions

**Mentor Pool:**
- query_graph for mentors
- search_text for mentor expertise

**Past Interactions:**
- query_graph for who team has met

## MATCHING CRITERIA
1. **Expertise Alignment**: Mentor skills match team needs
2. **No Repeats**: Haven't met this mentor yet
3. **Relevance**: Mentor background relates to blockers
4. **Availability**: Mentor is active

## OUTPUT FORMAT
```
## Team Analysis
- Name: [team]
- Key Challenges: [from search]
- Already Met: [mentor list]

## Mentor Recommendations
### 1. [Mentor Name]
- Expertise: [from search]
- Why Good Fit: [reasoning]
- Match Score: [High/Medium/Low]
```"""


# =============================================================================
# TOOL DESCRIPTIONS
# =============================================================================

ENTITY_RESEARCHER_DESCRIPTION = (
    "Finds entities, properties, and relationships in the knowledge graph. "
    "Call with entity names or topics to explore graph structure."
)

TEXT_RESEARCHER_DESCRIPTION = (
    "Searches text passages and documents for detailed content. "
    "Call with topics or questions to find specific mentions and quotes."
)

SUMMARY_RESEARCHER_DESCRIPTION = (
    "Finds document summaries and high-level overviews. "
    "Call for big-picture understanding of topics."
)

DEEP_REASONING_DESCRIPTION = (
    "Handles complex queries requiring multi-step reasoning, comparisons, and analysis. "
    "Call for analytical questions that need synthesis."
)

MENTOR_MATCHER_DESCRIPTION = (
    "Specialized agent for mentor-team matching recommendations based on expertise, "
    "needs, and history. Call for mentor suggestions."
)


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def _get_model(model_type: str = "research") -> ChatOpenAI:
    """Get a ChatOpenAI model instance for use with create_agent."""
    settings = get_settings()

    if model_type == "orchestrator":
        model_name = settings.default_orchestrator_model
    else:
        model_name = settings.default_research_model

    return ChatOpenAI(
        model=model_name,
        api_key=settings.openai_api_key,
        streaming=True,
    )


def create_entity_researcher() -> CompiledStateGraph:
    """Create an Entity Research Agent using LangChain v1's create_agent."""
    tools = [find_entity, query_graph, search_text, get_graph_schema]

    graph = create_agent(
        model=_get_model("research"),
        tools=tools,
        system_prompt=ENTITY_RESEARCHER_PROMPT,
    )

    logger.info("Created entity_researcher subgraph")
    return graph


def create_text_researcher() -> CompiledStateGraph:
    """Create a Text Research Agent using LangChain v1's create_agent."""
    tools = [search_text, search_chunks]

    graph = create_agent(
        model=_get_model("research"),
        tools=tools,
        system_prompt=TEXT_RESEARCHER_PROMPT,
    )

    logger.info("Created text_researcher subgraph")
    return graph


def create_summary_researcher() -> CompiledStateGraph:
    """Create a Summary Research Agent using LangChain v1's create_agent."""
    tools = [search_summaries, search_graph, search_rag]

    graph = create_agent(
        model=_get_model("research"),
        tools=tools,
        system_prompt=SUMMARY_RESEARCHER_PROMPT,
    )

    logger.info("Created summary_researcher subgraph")
    return graph


def create_deep_reasoning() -> CompiledStateGraph:
    """Create a Deep Reasoning Agent using LangChain v1's create_agent."""
    tools = [
        find_entity,
        query_graph,
        search_text,
        search_chunks,
        search_summaries,
        search_graph,
        search_rag,
        get_graph_schema,
    ]

    # Use stronger model for deep reasoning
    graph = create_agent(
        model=_get_model("orchestrator"),
        tools=tools,
        system_prompt=DEEP_REASONING_PROMPT,
    )

    logger.info("Created deep_reasoning subgraph")
    return graph


def create_mentor_matcher() -> CompiledStateGraph:
    """Create a Mentor Matching Agent using LangChain v1's create_agent."""
    tools = [
        find_entity,
        query_graph,
        search_text,
        search_chunks,
        search_summaries,
        get_graph_schema,
    ]

    # Use stronger model for mentor matching
    graph = create_agent(
        model=_get_model("orchestrator"),
        tools=tools,
        system_prompt=MENTOR_MATCHER_PROMPT,
    )

    logger.info("Created mentor_matcher subgraph")
    return graph
