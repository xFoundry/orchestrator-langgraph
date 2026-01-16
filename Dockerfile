# Multi-stage build for orchestrator-langgraph
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml README.md langgraph.json start.sh ./
COPY app/ app/
RUN pip install --no-cache-dir build "langgraph-cli[inmem]" && \
    pip install --no-cache-dir . && \
    chmod +x /app/start.sh

# Final stage
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app/start.sh /app/start.sh
COPY --from=builder /app/langgraph.json /app/langgraph.json

# Copy application code
COPY app/ app/

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

EXPOSE 8000 2024

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD sh -c "python -c 'import os, httpx; httpx.get(\"http://localhost:%s/ok\" % os.getenv(\"PORT\",\"8000\"))'" || exit 1

# Run the application
CMD ["sh", "/app/start.sh"]
