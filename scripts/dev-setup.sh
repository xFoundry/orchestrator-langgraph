#!/bin/bash
# Development setup script for orchestrator-langgraph

set -e

echo "🚀 Setting up orchestrator-langgraph for local development..."

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
if [[ "$PYTHON_VERSION" < "3.11" ]]; then
    echo "❌ Python 3.11+ required. Found: $PYTHON_VERSION"
    exit 1
fi
echo "✅ Python version: $PYTHON_VERSION"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
pip install -e ".[dev]"

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "⚠️  No .env file found. Creating from .env.example..."
    cp .env.example .env
    echo "📝 Please edit .env and add your OPENAI_API_KEY"
fi

# Check for Redis
if command -v redis-cli &> /dev/null; then
    if redis-cli ping &> /dev/null; then
        echo "✅ Redis is running"
    else
        echo "⚠️  Redis is installed but not running"
        echo "   Start with: redis-server"
        echo "   Or use Docker: docker run -d -p 6379:6379 redis:7-alpine"
    fi
else
    echo "⚠️  Redis not found. Install it or use Docker:"
    echo "   brew install redis"
    echo "   Or: docker run -d -p 6379:6379 redis:7-alpine"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Make sure .env has your OPENAI_API_KEY"
echo "2. Start Redis: redis-server (or docker run -d -p 6379:6379 redis:7-alpine)"
echo "3. Run the server: source .venv/bin/activate && uvicorn app.main:app --reload"
echo ""
echo "The service will be available at http://localhost:8000"
