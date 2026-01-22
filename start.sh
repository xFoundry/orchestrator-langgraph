#!/usr/bin/env sh
set -e

if [ -f "./scripts/ensure-env.sh" ]; then
  . "./scripts/ensure-env.sh"
fi

MODE="${USE_LANGGRAPH_SERVER:-false}"

if [ "$MODE" = "true" ] || [ "$MODE" = "1" ]; then
  echo "Starting LangGraph Agent Server..."
  CONFIG_PATH="${LANGGRAPH_CONFIG:-/app/langgraph.json}"
  if [ ! -f "$CONFIG_PATH" ]; then
    echo "ERROR: langgraph.json not found at $CONFIG_PATH"
    exit 1
  fi
  exec langgraph dev --config "$CONFIG_PATH" --host 0.0.0.0 --port "${PORT:-2024}" --no-reload --no-browser
fi

echo "Starting FastAPI orchestrator..."
# Use multiple workers for concurrent request handling (default: 4)
# For 80-90 users with 100 concurrent chats, 4-8 workers is recommended
WORKERS="${UVICORN_WORKERS:-4}"
echo "Using ${WORKERS} uvicorn workers for concurrent streaming..."

if command -v python >/dev/null 2>&1; then
  exec python -m uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}" --workers "$WORKERS"
fi
exec python3 -m uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}" --workers "$WORKERS"
