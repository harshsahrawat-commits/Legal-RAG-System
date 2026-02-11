#!/usr/bin/env bash
# ==============================================================================
# Legal RAG System - Production Startup Script
# Usage: ./deploy/start.sh
# ==============================================================================
set -euo pipefail

echo "=== Legal RAG System - Starting ==="

# Load environment variables
if [ -f .env ]; then
    set -a
    source .env
    set +a
else
    echo "ERROR: .env file not found. Copy .env.template to .env and fill in values."
    exit 1
fi

# Verify required env vars
REQUIRED_VARS=(POSTGRES_URL VOYAGE_API_KEY COHERE_API_KEY NVIDIA_API_KEY)
for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var:-}" ]; then
        echo "ERROR: $var is not set in .env"
        exit 1
    fi
done

# Database initialization
echo "--- Initializing database ---"
python -c "
from execution.legal_rag.vector_store import VectorStore
import os
store = VectorStore(os.environ['POSTGRES_URL'])
store.initialize()
print('Database initialized successfully')
"

# Enable RLS if not already enabled
echo "--- Ensuring RLS is active ---"
python -c "
import psycopg2, os
conn = psycopg2.connect(os.environ['POSTGRES_URL'])
cur = conn.cursor()
cur.execute(\"SELECT relrowsecurity FROM pg_class WHERE relname = 'document_chunks'\")
row = cur.fetchone()
if row and not row[0]:
    cur.execute('ALTER TABLE document_chunks ENABLE ROW LEVEL SECURITY')
    conn.commit()
    print('RLS enabled on document_chunks')
else:
    print('RLS already active')
cur.close()
conn.close()
"

# Start server
WORKERS="${WORKERS:-2}"
LOG_LEVEL="${LOG_LEVEL:-info}"

echo "--- Starting uvicorn (workers=$WORKERS, log_level=$LOG_LEVEL) ---"
exec python -m uvicorn execution.legal_rag.api:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers "$WORKERS" \
    --log-level "$LOG_LEVEL" \
    --timeout-keep-alive 30
