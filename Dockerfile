# RAG PDF Parser Docker Image
# Multi-stage build for smaller image size

# Stage 1: Base image with system dependencies
FROM python:3.11-slim AS base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Stage 2: GPU support (optional)
FROM base AS gpu

# Install CUDA support (for GPU acceleration)
RUN pip install --no-cache-dir paddlepaddle-gpu

# Stage 3: CPU-only (default)
FROM base AS cpu

# Install CPU-only paddlepaddle
RUN pip install --no-cache-dir paddlepaddle

# Stage 4: Application
FROM cpu AS final

# Copy project files
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY templates/ ./templates/
COPY main.py streamlit_viewer.py ./

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Install Ollama (for VLM and translation)
RUN curl -fsSL https://ollama.com/install.sh | sh || true

# Create output directory
RUN mkdir -p /app/output /app/data

# Expose Streamlit port
EXPOSE 8501

# Default command: run Streamlit viewer
CMD ["streamlit", "run", "streamlit_viewer.py", "--server.address", "0.0.0.0"]
