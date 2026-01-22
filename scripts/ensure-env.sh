#!/usr/bin/env sh
set -e

# Load local env files when present
if [ -f ".env" ]; then
  set -a
  . ".env"
  set +a
fi

if [ -f ".env.local" ]; then
  set -a
  . ".env.local"
  set +a
fi

# Ensure defaults for persistence-related settings
if [ -z "${REDIS_URL:-}" ]; then
  export REDIS_URL="redis://localhost:6379"
fi

if [ -z "${USE_MEMORY_STORE:-}" ]; then
  export USE_MEMORY_STORE="false"
fi

if [ -z "${USE_MEMORY_CHECKPOINTER:-}" ]; then
  export USE_MEMORY_CHECKPOINTER="false"
fi

