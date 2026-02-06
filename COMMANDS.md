# Legal RAG - Quick Commands Reference

## Prerequisites

```bash
# Start PostgreSQL (required for vector store)
brew services start postgresql@17

# Verify PostgreSQL is running
brew services list | grep postgresql
```

## Run

```bash
# Run the Streamlit demo app
streamlit run execution/legal_rag/demo_app.py

# Run on a specific port
streamlit run execution/legal_rag/demo_app.py --server.port 8502

# Run the test pipeline
python -m execution.legal_rag.test_pipeline
```

**Demo URL:** http://localhost:8501

## Stop

```bash
# Stop Streamlit (find and kill process)
pkill -f streamlit

# Or use lsof to find process on port 8501
lsof -i :8501
kill -9 <PID>

# Stop PostgreSQL
brew services stop postgresql@17
```

## Refresh / Restart

```bash
# Quick restart (stop + start)
pkill -f streamlit && streamlit run execution/legal_rag/demo_app.py

# Clear Streamlit cache and restart
streamlit cache clear && streamlit run execution/legal_rag/demo_app.py

# Restart PostgreSQL
brew services restart postgresql@17
```

## Database Commands

```bash
# Connect to database
/opt/homebrew/opt/postgresql@17/bin/psql -d legal_rag

# Reset database (delete all data)
/opt/homebrew/opt/postgresql@17/bin/psql -d legal_rag -c "DROP TABLE IF EXISTS document_chunks, legal_documents CASCADE;"

# Verify pgvector extension
/opt/homebrew/opt/postgresql@17/bin/psql -d legal_rag -c "SELECT * FROM pg_extension WHERE extname = 'vector';"
```

## Environment Setup

```bash
# Install dependencies
pip install -r requirements_legal_rag.txt

# Check required env vars
cat .env | grep -E "COHERE_API_KEY|POSTGRES_URL|ANTHROPIC_API_KEY"
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Port 8501 in use | `pkill -f streamlit` or use `--server.port 8502` |
| PostgreSQL not running | `brew services start postgresql@17` |
| Import errors | `pip install -r requirements_legal_rag.txt` |
| Missing API keys | Check `.env` file has `COHERE_API_KEY`, `ANTHROPIC_API_KEY` |
