# LLM Rate Limiting

This document describes the rate limiting system implemented to prevent 529 "Overloaded" errors when running multiple sub-agents in parallel.

## Problem

When the orchestrator spawns multiple sub-agents simultaneously (e.g., user asks "research topics A, B, and C"), each sub-agent makes its own LLM API call. These parallel calls can overwhelm the LLM provider (OpenRouter, Anthropic, OpenAI), resulting in:

- **529 "Overloaded"**: Provider is at capacity, request rejected
- **503 "Service Unavailable"**: Temporary service issues
- **Rate limit errors**: Account-level limits exceeded

The `deepagents` library executes sub-agent tool calls via `asyncio.gather`, which fires all requests simultaneously.

## Solution

We implemented a **queue-based rate limiting system** that:

1. **Queues LLM calls** instead of firing them simultaneously
2. **Retries transparently** on transient errors with exponential backoff
3. **Is invisible to users** - sub-agents still "run in parallel", they just wait their turn for API calls

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  User Request: "Research A, B, and C"                       │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Orchestrator spawns 3 sub-agents (parallel via asyncio)    │
│  - research-agent(A)                                        │
│  - research-agent(B)                                        │
│  - research-agent(C)                                        │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  LLM Concurrency Manager (Global Singleton)                 │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Semaphore (max_concurrent=5)                       │    │
│  │  Rate Limiter (requests_per_second=3.0)             │    │
│  │  Jitter (0-100ms random delay)                      │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
│  Call Queue:                                                │
│  [Agent A: waiting] [Agent B: in-flight] [Agent C: waiting] │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  OpenRouter / LLM Provider                                  │
│  (Receives controlled flow of requests)                     │
└─────────────────────────────────────────────────────────────┘
```

### Components

#### 1. `LLMConcurrencyManager` (Singleton)

Located in `app/middleware/llm_rate_limiter.py`

Manages global concurrency across all LLM calls:

- **Semaphore**: Limits concurrent API calls (default: 5)
- **Rate Limiter**: Smooths request rate (default: 3 RPS)
- **Jitter**: Adds random delay (0-100ms) to prevent thundering herd
- **Stats**: Tracks active calls, total calls, retried calls

#### 2. `RateLimitedChatModel` (Wrapper)

Wraps any LangChain `BaseChatModel` with:

- **Automatic queuing**: Waits for semaphore + rate limiter before each call
- **Retry with backoff**: Retries on 529, 503, 502, 504 errors
- **Transparent interface**: Drop-in replacement for any chat model

#### 3. Configuration

Settings in `app/config/settings.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `llm_max_concurrent` | 5 | Max parallel LLM API calls |
| `llm_requests_per_second` | 3.0 | Max requests per second |
| `llm_max_retries` | 3 | Retry attempts for transient errors |
| `llm_min_retry_wait` | 1.0 | Min wait between retries (seconds) |
| `llm_max_retry_wait` | 30.0 | Max wait between retries (seconds) |

## Configuration

### Environment Variables

```bash
# Rate Limiting (prevents 529 "Overloaded" errors)
LLM_MAX_CONCURRENT=5          # Max parallel LLM calls
LLM_REQUESTS_PER_SECOND=3.0   # Max RPS (smooths bursts)
LLM_MAX_RETRIES=3             # Retry attempts
LLM_MIN_RETRY_WAIT=1.0        # Min retry wait (seconds)
LLM_MAX_RETRY_WAIT=30.0       # Max retry wait (seconds)
```

### Tuning for OpenRouter

Adjust based on your OpenRouter credit balance:

| Credit Balance | `LLM_MAX_CONCURRENT` | `LLM_REQUESTS_PER_SECOND` |
|---------------|----------------------|---------------------------|
| $5            | 3                    | 2.0                       |
| $10           | 5                    | 3.0                       |
| $25           | 8                    | 5.0                       |
| $50+          | 10                   | 8.0                       |

**Note**: OpenRouter's actual capacity depends on:
1. Your credit balance (~$1 = 1 RPS capacity)
2. The underlying model provider's load (Anthropic/OpenAI)
3. Time of day (peak hours have more 529s)

### Tuning for High Load

For production with many concurrent users:

```bash
# Conservative settings for reliability
LLM_MAX_CONCURRENT=8
LLM_REQUESTS_PER_SECOND=5.0
LLM_MAX_RETRIES=5
LLM_MAX_RETRY_WAIT=60.0
```

## How It Works

### Request Flow

1. **Sub-agent makes LLM call** via `model.ainvoke()` or `model.astream()`
2. **Rate limiter intercepts** the call
3. **Jitter added** (0-100ms random delay) to stagger requests
4. **Semaphore acquired** (waits if at max concurrent)
5. **Rate limiter acquired** (waits if exceeding RPS)
6. **API call made** to OpenRouter/provider
7. **On success**: Response returned, semaphore released
8. **On 529/503**: Wait with exponential backoff, retry up to N times

### Retry Logic

Retryable errors:
- `529 Overloaded`
- `503 Service Unavailable`
- `502 Bad Gateway`
- `504 Gateway Timeout`
- `Rate limit` / `Too many requests`

Backoff formula:
```
wait_time = min(max_wait, min_wait * 2^attempt + random(0, 1))
```

Example with defaults (min=1s, max=30s):
- Attempt 1: Wait 1-2 seconds
- Attempt 2: Wait 2-3 seconds
- Attempt 3: Wait 4-5 seconds (then fail if still erroring)

## Monitoring

### Logging

The rate limiter logs at DEBUG level:

```
DEBUG - LLM call acquired: active=3/5, total=47
DEBUG - LLM call released: active=2/5
WARNING - Retryable LLM error: 529 Overloaded. Retrying in 2.3s...
```

Enable debug logging to monitor:

```bash
LOG_LEVEL=DEBUG
```

### Stats

Get current stats programmatically:

```python
from app.middleware.llm_rate_limiter import LLMConcurrencyManager

manager = await LLMConcurrencyManager.get_instance()
stats = manager.get_stats()
# {
#     "active_calls": 3,
#     "max_concurrent": 5,
#     "total_calls": 147,
#     "retried_calls": 12,
#     "requests_per_second": 3.0
# }
```

## Troubleshooting

### Still Getting 529 Errors

1. **Lower concurrent limit**: Try `LLM_MAX_CONCURRENT=3`
2. **Lower RPS**: Try `LLM_REQUESTS_PER_SECOND=1.5`
3. **Increase retries**: Try `LLM_MAX_RETRIES=5`
4. **Check OpenRouter balance**: Low balance = lower capacity

### Requests Taking Too Long

1. **Increase concurrent limit**: Try `LLM_MAX_CONCURRENT=8`
2. **Increase RPS**: Try `LLM_REQUESTS_PER_SECOND=5.0`
3. **Check if hitting retries**: Enable DEBUG logging

### Sub-agents Timing Out

The rate limiter adds latency when queuing. If sub-agents have tight timeouts:

1. Increase sub-agent timeouts
2. Or increase `LLM_MAX_CONCURRENT` to reduce queue wait time

## Implementation Details

### Why Not Per-User Limits?

We use a **global** rate limiter instead of per-user because:

1. **OpenRouter has global capacity**: Your account shares limits, not per-user
2. **Simpler**: One semaphore to manage
3. **Fairer**: Prevents one user's burst from starving others

If you need per-user limits (e.g., for billing), you can add a second layer in `LLMConcurrencyManager`.

### Why Wrap the Model?

We wrap the model (not the tool executor) because:

1. **Catches all LLM calls**: Including from sub-agents with their own models
2. **Works with streaming**: Both `ainvoke` and `astream` are rate-limited
3. **Transparent**: No changes needed to deepagents or LangGraph internals

### Thread Safety

The `LLMConcurrencyManager` singleton is thread-safe:

- Uses `asyncio.Lock` for initialization
- `asyncio.Semaphore` is async-safe
- `aiolimiter.AsyncLimiter` is async-safe

## Dependencies

Added to `pyproject.toml`:

```toml
# Rate limiting and retry
"aiolimiter>=1.1.0",
"tenacity>=8.2.0",
```

Install:

```bash
pip install aiolimiter tenacity
# or
uv sync
```
