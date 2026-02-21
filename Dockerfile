# ==============================================================================
# Legal RAG System - Production Dockerfile
# Multi-stage build for FastAPI backend
# ==============================================================================

# Stage 1: Build dependencies
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build dependencies for psycopg2
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements-prod.txt requirements.txt
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# Stage 2: Production image
FROM python:3.12-slim

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash appuser

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY execution/ execution/

# Create writable directories for non-root user
RUN mkdir -p document_files && chown appuser:appuser document_files

# Set cache directories to writable locations
ENV HF_HOME=/tmp/huggingface
ENV XDG_CACHE_HOME=/tmp/cache
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Switch to non-root user
USER appuser

# Railway provides $PORT at runtime; default to 8000 for local Docker
ENV PORT=8000

# Health check (uses shell form to expand $PORT)
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT}/api/v1/health || exit 1

EXPOSE ${PORT}

# Start uvicorn (shell form to expand $PORT)
CMD python -m uvicorn execution.legal_rag.api:app \
    --host 0.0.0.0 --port ${PORT} \
    --workers 2 --timeout-keep-alive 30
