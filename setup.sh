#!/bin/bash
# RAG PDF Parser Setup Script for Linux/Mac

set -e

echo "================================"
echo "RAG PDF Parser Setup"
echo "================================"

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "Python version: $PYTHON_VERSION"

# Check if uvx is available
if command -v uvx &> /dev/null; then
    echo "Using uvx for installation..."
    USE_UVX=true
else
    echo "uvx not found. Using pip instead..."
    USE_UVX=false
fi

# Create virtual environment if not using uvx
if [ "$USE_UVX" = false ]; then
    if [ ! -d "venv" ]; then
        echo "Creating virtual environment..."
        python3 -m venv venv
    fi

    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Install dependencies
echo "Installing dependencies..."
if [ "$USE_UVX" = true ]; then
    uvx --from . rag-pdf-parser --help > /dev/null 2>&1 || pip install -e .
else
    pip install -e .
fi

# Create necessary directories
echo "Creating directories..."
mkdir -p output data templates tests

# Check for GPU support
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "NVIDIA GPU detected. Install GPU support? (y/n)"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        pip install -e ".[gpu]"
    fi
fi

# Install Ollama
if ! command -v ollama &> /dev/null; then
    echo ""
    echo "Ollama not found. Install Ollama? (y/n)"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        curl -fsSL https://ollama.com/install.sh | sh
    fi
fi

# Pull required Ollama models
if command -v ollama &> /dev/null; then
    echo ""
    echo "Pulling Ollama models..."
    ollama pull qwen3-vl:8b || echo "Failed to pull qwen3-vl:8b"
    ollama pull gpt-oss:20b || echo "Failed to pull gpt-oss:20b"
    ollama pull qwen2.5-coder:7b || echo "Failed to pull qwen2.5-coder:7b"
    ollama pull qwen3:8b || echo "Failed to pull qwen3:8b"
    ollama pull mistral:7b || echo "Failed to pull mistral:7b"
fi

echo ""
echo "================================"
echo "Setup complete!"
echo "================================"
echo ""
echo "Usage:"
echo "  Process PDF:    python main.py input.pdf"
echo "  Run viewer:     streamlit run streamlit_viewer.py"
echo "  Run tests:      python tests/test_pipeline.py"
echo ""
echo "With uvx:"
echo "  uvx rag-pdf-parser input.pdf"
echo ""
echo "With Docker:"
echo "  docker-compose up"
echo ""
