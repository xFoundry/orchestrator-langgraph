# Orchestrator LangGraph

Multi-agent orchestrator for the CognoXent mentorship platform, built with LangGraph.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend (Next.js)                        │
│                  Vercel AI SDK useChat hook                      │
└─────────────────────────────┬───────────────────────────────────┘
                              │ SSE Stream
┌─────────────────────────────▼───────────────────────────────────┐
│                    orchestrator-langgraph                        │
│                      FastAPI + LangGraph                         │
├─────────────────────────────────────────────────────────────────┤
│  Main Orchestrator (create_react_agent)                         │
│  ├── Direct Tools: query_graph, find_entity, search_*, etc.     │
│  └── Subgraph Tools (wrapped as callable tools):                │
│      ├── entity_researcher  (ReAct subgraph)                    │
│      ├── text_researcher    (ReAct subgraph)                    │
│      ├── summary_researcher (ReAct subgraph)                    │
│      ├── deep_reasoning     (ReAct subgraph)                    │
│      └── mentor_matcher     (ReAct subgraph)                    │
├─────────────────────────────────────────────────────────────────┤
│  Redis Checkpointer (AsyncRedisSaver) - Session persistence     │
└─────────────────────────────────────────────────────────────────┘
```

## Local Development

### Prerequisites

- Python 3.11+
- Redis (or Docker)

### Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -e .

# Copy environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Running

```bash
# Start Redis (if not using Docker)
redis-server

# Or use Docker Compose
docker compose up redis -d

# Start the service
uvicorn app.main:app --reload --port 8000
```

### With Docker Compose

```bash
docker compose up
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Service info |
| `/health` | GET | Health check |
| `/live` | GET | Kubernetes liveness probe |
| `/ready` | GET | Kubernetes readiness probe |
| `/chat` | POST | Synchronous chat (waits for full response) |
| `/chat/stream` | POST | SSE streaming chat |
| `/chat/stream/health` | GET | Stream endpoint health |

### Chat Request

```json
{
  "message": "What feedback did mentors give to DefenX?",
  "thread_id": "optional-session-id",
  "user_id": "optional-user-id",
  "tenant_id": "default",
  "user_context": {
    "name": "John Doe",
    "email": "john@example.com",
    "role": "Staff",
    "cohort": "Spring 2025",
    "auth0_id": "auth0|123"
  },
  "use_memory": false
}
```

### SSE Event Types

| Event | Description |
|-------|-------------|
| `agent_activity` | Agent actions (tool calls, delegations) |
| `text_chunk` | Partial response text |
| `citation` | Source citations |
| `tool_result` | Tool execution results |
| `thinking` | Internal reasoning phases |
| `complete` | Final response with full message |
| `error` | Error information |

## Railway Deployment

### 1. Create Railway Project

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login
railway login

# Create new project
railway init
```

### 2. Add Redis Plugin

In the Railway dashboard:
1. Click "New" → "Database" → "Redis"
2. Railway will automatically set `REDIS_URL`

### 3. Configure Environment Variables

Set these in Railway dashboard or via CLI:

```bash
# Required - LLM API Keys
railway variables set OPENAI_API_KEY=sk-...

# OpenRouter - Recommended for Anthropic models (single API key for all providers)
# Get your key at: https://openrouter.ai/keys
railway variables set OPENROUTER_API_KEY=sk-or-v1-...

# OR use direct Anthropic API (not needed if using OpenRouter)
# railway variables set ANTHROPIC_API_KEY=sk-ant-...

# External Services
railway variables set COGNEE_API_URL=https://your-cognee-service.railway.app
railway variables set COGNEE_SECRET_KEY=your-secret
railway variables set CORS_ORIGINS=https://your-frontend.vercel.app

# Outline MCP (docs knowledge base)
railway variables set OUTLINE_MCP_BASE_URL=https://docs.xfoundry.org
railway variables set OUTLINE_MCP_API_KEY=your-outline-api-key
railway variables set OUTLINE_MCP_PROTOCOL_VERSION=2025-06-18

# Redis is automatically linked via ${{Redis.REDIS_URL}}

# Optional - Model Configuration
railway variables set DEFAULT_ORCHESTRATOR_MODEL=gpt-5.2
railway variables set DEFAULT_RESEARCH_MODEL=gpt-5.2-pro
railway variables set LOG_LEVEL=INFO

# Optional - OpenRouter Settings
# railway variables set USE_OPENROUTER_FOR_ANTHROPIC=true  # Default: true
# railway variables set OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

#### Model Provider Setup

The orchestrator supports two ways to access Anthropic models:

1. **OpenRouter (Recommended)**: Single API key for multiple providers
   - Set `OPENROUTER_API_KEY` 
   - Anthropic models are automatically routed through OpenRouter
   - Benefits: Unified billing, fallbacks, access to many providers

2. **Direct Anthropic API**: Use your Anthropic API key directly
   - Set `ANTHROPIC_API_KEY`
   - Set `USE_OPENROUTER_FOR_ANTHROPIC=false`

### 4. Deploy

```bash
railway up
```

Or connect to GitHub for automatic deployments.

### 5. Update Frontend

Add the Railway URL to your frontend environment:

```env
ORCHESTRATOR_LANGGRAPH_URL=https://orchestrator-langgraph-production.up.railway.app
```

## Outline MCP Smoke Test

1) Initialize a session:

```bash
curl -s -X POST "$OUTLINE_MCP_BASE_URL/api/mcp" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OUTLINE_MCP_API_KEY" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
      "protocolVersion": "2025-06-18",
      "capabilities": {},
      "clientInfo": { "name": "mcp-smoke-test", "version": "1.0.0" }
    }
  }' -i
```

2) Copy `Mcp-Session-Id` + `MCP-Protocol-Version` headers and list tools:

```bash
curl -s -X POST "$OUTLINE_MCP_BASE_URL/api/mcp" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OUTLINE_MCP_API_KEY" \
  -H "MCP-Protocol-Version: 2025-06-18" \
  -H "Mcp-Session-Id: <session-id>" \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/list"}'
```

## Project Structure

```
orchestrator-langgraph/
├── app/
│   ├── main.py              # FastAPI with Redis lifespan
│   ├── config.py            # Environment settings
│   ├── graphs/
│   │   ├── orchestrator.py  # Main ReAct agent
│   │   ├── state.py         # State schemas
│   │   └── subgraphs/
│   │       └── researchers.py  # 5 subgraph factories
│   ├── tools/
│   │   ├── graph_tools.py   # Cypher query tools
│   │   ├── cognee_tools.py  # Search tools
│   │   ├── user_memory.py   # Per-user memory
│   │   └── subgraph_wrapper.py
│   ├── streaming/
│   │   ├── sse_events.py    # Event models
│   │   └── event_mapper.py  # LangGraph → SSE
│   ├── api/routes/
│   │   ├── chat.py          # Sync endpoint
│   │   └── chat_stream.py   # SSE endpoint
│   └── persistence/
│       └── redis.py         # Checkpointer
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
├── railway.toml
└── .env.example
```

## Comparison with Google ADK Version

| Feature | Google ADK | LangGraph |
|---------|------------|-----------|
| Agent Framework | `LlmAgent` | `create_react_agent` |
| Sub-agents | `AgentTool` | `StructuredTool` wrapper |
| Persistence | `InMemorySessionService` | `AsyncRedisSaver` |
| Streaming | Custom event mapping | `astream_events()` |
| Planner | `PlanReActPlanner` | Built-in ReAct |

Both versions produce the same SSE event format for frontend compatibility.
