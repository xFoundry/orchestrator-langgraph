"""FastAPI application entry point with Redis lifespan management."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pythonjsonlogger import jsonlogger

from app.config import get_settings
from app.persistence.redis import get_checkpointer, close_checkpointer
from app.api.routes import chat, chat_stream

# Configure logging
settings = get_settings()
logger = logging.getLogger(__name__)


def setup_logging() -> None:
    """Configure JSON logging for production."""
    log_handler = logging.StreamHandler()
    formatter = jsonlogger.JsonFormatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers = [log_handler]
    root_logger.setLevel(settings.log_level.upper())

    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan with Redis initialization and cleanup."""
    setup_logging()
    logger.info("Starting orchestrator-langgraph service")

    # Initialize Redis checkpointer
    try:
        checkpointer = await get_checkpointer()
        app.state.checkpointer = checkpointer
        logger.info(f"Redis checkpointer initialized at {settings.redis_url}")
    except Exception as e:
        logger.error(f"Failed to initialize Redis checkpointer: {e}")
        raise

    yield

    # Cleanup
    logger.info("Shutting down orchestrator-langgraph service")
    await close_checkpointer()


# Create FastAPI app
app = FastAPI(
    title="CognoXent Orchestrator (LangGraph)",
    description="Multi-agent orchestrator for the mentorship platform using LangGraph",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat.router)
app.include_router(chat_stream.router)


@app.get("/")
async def root() -> dict:
    """Root endpoint with service info."""
    return {
        "service": "orchestrator-langgraph",
        "version": "0.1.0",
        "status": "running",
    }


@app.get("/health")
async def health() -> dict:
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/live")
async def liveness() -> dict:
    """Kubernetes liveness probe."""
    return {"status": "alive"}


@app.get("/ready")
async def readiness() -> dict:
    """Kubernetes readiness probe."""
    # Check if Redis is connected
    try:
        checkpointer = app.state.checkpointer
        if checkpointer is None:
            return {"status": "not_ready", "reason": "checkpointer_not_initialized"}
        return {"status": "ready"}
    except Exception as e:
        return {"status": "not_ready", "reason": str(e)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
    )
