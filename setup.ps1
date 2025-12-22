# RAG PDF Parser Setup Script for Windows PowerShell

Write-Host "================================" -ForegroundColor Cyan
Write-Host "RAG PDF Parser Setup" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan

# Check Python version
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Python version: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "Error: Python is not installed" -ForegroundColor Red
    exit 1
}

# Check if uvx is available
$useUvx = $false
try {
    uvx --version 2>&1 | Out-Null
    Write-Host "Using uvx for installation..." -ForegroundColor Green
    $useUvx = $true
} catch {
    Write-Host "uvx not found. Using pip instead..." -ForegroundColor Yellow
}

# Create virtual environment if not using uvx
if (-not $useUvx) {
    if (-not (Test-Path "venv")) {
        Write-Host "Creating virtual environment..." -ForegroundColor Yellow
        python -m venv venv
    }

    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    & .\venv\Scripts\Activate.ps1
}

# Install dependencies
Write-Host "Installing dependencies..." -ForegroundColor Yellow
if ($useUvx) {
    try {
        uvx --from . rag-pdf-parser --help 2>&1 | Out-Null
    } catch {
        pip install -e .
    }
} else {
    pip install -e .
}

# Create necessary directories
Write-Host "Creating directories..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path output, data, templates, tests | Out-Null

# Check for GPU support
try {
    nvidia-smi 2>&1 | Out-Null
    Write-Host ""
    $response = Read-Host "NVIDIA GPU detected. Install GPU support? (y/n)"
    if ($response -match "^[yY]") {
        pip install -e ".[gpu]"
    }
} catch {
    Write-Host "No NVIDIA GPU detected. Using CPU version." -ForegroundColor Yellow
}

# Install Ollama
try {
    ollama --version 2>&1 | Out-Null
    Write-Host "Ollama is already installed." -ForegroundColor Green
} catch {
    Write-Host ""
    $response = Read-Host "Ollama not found. Install Ollama? (y/n)"
    if ($response -match "^[yY]") {
        Write-Host "Downloading Ollama installer..." -ForegroundColor Yellow
        Invoke-WebRequest -Uri "https://ollama.com/download/OllamaSetup.exe" -OutFile "$env:TEMP\OllamaSetup.exe"
        Start-Process -FilePath "$env:TEMP\OllamaSetup.exe" -Wait
        Remove-Item "$env:TEMP\OllamaSetup.exe"
    }
}

# Pull required Ollama models
try {
    ollama --version 2>&1 | Out-Null
    Write-Host ""
    Write-Host "Pulling Ollama models..." -ForegroundColor Yellow

    $models = @("qwen3-vl:8b", "gpt-oss:20b", "qwen2.5-coder:7b", "qwen3:8b", "mistral:7b")
    foreach ($model in $models) {
        try {
            ollama pull $model
        } catch {
            Write-Host "Failed to pull $model" -ForegroundColor Red
        }
    }
} catch {
    Write-Host "Skipping Ollama model download (Ollama not installed)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "================================" -ForegroundColor Cyan
Write-Host "Setup complete!" -ForegroundColor Green
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Usage:" -ForegroundColor Yellow
Write-Host "  Process PDF:    python main.py input.pdf"
Write-Host "  Run viewer:     python -m streamlit run streamlit_viewer.py"
Write-Host "  Run tests:      python tests/test_pipeline.py"
Write-Host ""
Write-Host "With uvx:" -ForegroundColor Yellow
Write-Host "  uvx rag-pdf-parser input.pdf"
Write-Host ""
Write-Host "With Docker:" -ForegroundColor Yellow
Write-Host "  docker-compose up"
Write-Host ""
