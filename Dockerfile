# Modomo Backend - Railway Deployment
# Multi-stage build for optimized image size

FROM python:3.11.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements (for layer caching)
COPY requirements-railway-minimal.txt .

# Upgrade pip and setuptools to avoid build issues
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-railway-minimal.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/models /app/checkpoints /app/uploads /app/logs

# Expose port (Railway will set PORT env variable)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Start command (Railway will override with Procfile if present)
CMD ["uvicorn", "api.main_minimal:app", "--host", "0.0.0.0", "--port", "8000"]
