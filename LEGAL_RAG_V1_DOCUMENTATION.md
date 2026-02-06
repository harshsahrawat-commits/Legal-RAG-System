# Legal RAG System - Version 1.0 Documentation

> Complete reference for the Legal RAG (Retrieval Augmented Generation) system.
> **Version:** 1.0
> **Built:** February 1-3, 2026
> **Status:** Fully functional with Streamlit demo app

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture](#2-architecture)
3. [Component Details](#3-component-details)
4. [Database Schema](#4-database-schema)
5. [Environment Setup](#5-environment-setup)
6. [How to Start](#6-how-to-start)
7. [How to Stop](#7-how-to-stop)
8. [Usage Guide](#8-usage-guide)
9. [Cost Estimates](#9-cost-estimates)
10. [Known Limitations](#10-known-limitations)
11. [Troubleshooting](#11-troubleshooting)
12. [Future Roadmap](#12-future-roadmap)

---

## 1. System Overview

### Purpose
A **legal-grade Agentic RAG system** designed for querying legal documents (contracts, statutes, case law) with citation-level precision. Built for enterprise legal AI SaaS applications requiring accurate document retrieval and answer generation.

### Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Language** | Python 3.14 | Core implementation |
| **Vector DB** | PostgreSQL 17 + pgvector | Document & embedding storage |
| **Embeddings** | Cohere embed-v3 (1024 dims) | Semantic search vectors |
| **Reranking** | Cohere rerank-v3 | 40% precision improvement |
| **LLM** | NVIDIA NIM (Llama 3.1 70B) | Answer generation |
| **Frontend** | Streamlit | Demo web interface |
| **Framework** | LlamaIndex | RAG orchestration |

### Key Features
- **Hybrid Search**: Combines vector (semantic) + keyword (BM25) search
- **Reciprocal Rank Fusion**: Merges search results optimally
- **Hierarchical Chunking**: Preserves legal document structure
- **Citation Extraction**: Exact section and page references
- **Multi-tenant Ready**: Row-Level Security prepared
- **Document Persistence**: Survives app restarts

---

## 2. Architecture

### System Flow Diagram

```
                    ┌─────────────────────────────────────────────────────────┐
                    │                    QUERY PIPELINE                        │
                    └─────────────────────────────────────────────────────────┘

User Query: "What are the termination clauses?"
     │
     ▼
┌─────────────────┐
│ Cohere embed-v3 │ ──► 1024-dimensional query vector
└─────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────┐
│              HYBRID RETRIEVAL PIPELINE               │
│  ┌─────────────────┐    ┌─────────────────┐        │
│  │  Vector Search  │    │  Keyword Search │        │
│  │   (Semantic)    │    │     (BM25)      │        │
│  │     60%         │    │      40%        │        │
│  └────────┬────────┘    └────────┬────────┘        │
│           │                      │                  │
│           └──────────┬───────────┘                  │
│                      ▼                              │
│         ┌─────────────────────────┐                │
│         │ Reciprocal Rank Fusion  │                │
│         └───────────┬─────────────┘                │
│                     ▼                              │
│         ┌─────────────────────────┐                │
│         │   Cohere Reranking      │                │
│         │  (Top 20 → Top 5)       │                │
│         └───────────┬─────────────┘                │
└─────────────────────┼───────────────────────────────┘
                      ▼
           ┌─────────────────────────┐
           │   Citation Extractor    │
           │ [Doc, Section, Page N]  │
           └───────────┬─────────────┘
                      ▼
           ┌─────────────────────────┐
           │      NVIDIA NIM         │
           │ (Llama 3.1 70B Instruct)│
           └───────────┬─────────────┘
                      ▼
           ┌─────────────────────────┐
           │  Answer with Citations  │
           │  "According to [1]..."  │
           └─────────────────────────┘
```

### Ingestion Pipeline

```
PDF Document
     │
     ▼
┌─────────────────┐
│ Document Parser │ ──► Extracts text, structure, metadata
│ (Docling/PyMuPDF)│     Auto-detects document type
└────────┬────────┘
         ▼
┌─────────────────┐
│ Legal Chunker   │ ──► Hierarchical chunks (L0-L3)
│                 │     100-token overlap
└────────┬────────┘
         ▼
┌─────────────────┐
│ Cohere embed-v3 │ ──► 1024-dim embeddings
│ (batch process) │     Cached to file
└────────┬────────┘
         ▼
┌─────────────────┐
│ PostgreSQL +    │ ──► Stores documents & chunks
│ pgvector        │     IVFFlat + GIN indexes
└─────────────────┘
```

### 3-Layer Architecture (from CLAUDE.md)

| Layer | Purpose | Components |
|-------|---------|------------|
| **Layer 1: Directive** | What to do | `directives/legal_rag/*.md` - SOPs |
| **Layer 2: Orchestration** | Decision making | AI agent (Claude/LLM) |
| **Layer 3: Execution** | Doing the work | `execution/legal_rag/*.py` scripts |

---

## 3. Component Details

### File Structure

```
Legal_RAG_System_V1/
├── CLAUDE.md                     # Agent operating instructions
├── UPDATE.md                     # Session changelog
├── COMMANDS.md                   # Quick command reference
├── requirements_legal_rag.txt    # Python dependencies
├── .env.template                 # Environment variable template
├── LEGAL_RAG_V1_DOCUMENTATION.md # This file
│
├── directives/legal_rag/
│   ├── ingest_document.md        # Document ingestion SOP
│   └── query_documents.md        # Query handling SOP
│
└── execution/legal_rag/
    ├── __init__.py               # Module exports
    ├── document_parser.py        # PDF extraction
    ├── chunker.py                # Hierarchical chunking
    ├── embeddings.py             # Cohere embeddings
    ├── vector_store.py           # PostgreSQL + pgvector
    ├── retriever.py              # Hybrid search
    ├── citation.py               # Citation formatting
    ├── demo_app.py               # Streamlit UI
    ├── test_pipeline.py          # Validation script
    ├── README.md                 # Technical docs
    └── agents/
        └── __init__.py           # Future agents placeholder
```

### Script Details

#### `document_parser.py` - PDF Extraction
- **Primary**: Docling (97.9% accuracy on tables)
- **Fallback**: PyMuPDF4LLM
- **Features**:
  - Auto-detects document type (contract, statute, case_law, regulation, brief, memo)
  - Extracts metadata: title, jurisdiction, parties, effective date
  - Preserves legal structure (articles, clauses, sections)
  - Tracks page numbers for each section

#### `chunker.py` - Hierarchical Chunking
- **Chunk Levels**:
  - L0: Document summary (500-1000 tokens)
  - L1: Section/Chapter (1000-2000 tokens)
  - L2: Article/Clause (300-800 tokens)
  - L3: Paragraph (100-300 tokens)
- **Features**:
  - 100-token overlap for context preservation
  - Extracts legal cross-references
  - Preserves defined terms
  - Maintains hierarchy path for citations

#### `embeddings.py` - Cohere Embeddings
- **Model**: Cohere embed-v3 (1024 dimensions)
- **Features**:
  - Different `input_type` for documents vs queries
  - File-based caching to avoid re-computation
  - Batch processing (96 texts per batch)
  - Automatic retry with exponential backoff

#### `vector_store.py` - PostgreSQL + pgvector
- **Tables**: `legal_documents`, `document_chunks`
- **Indexes**:
  - IVFFlat for vector similarity (cosine distance)
  - GIN for full-text BM25 search
- **Features**:
  - Row-Level Security ready (multi-tenant)
  - `list_documents()` for persistence on startup
  - Both vector and keyword search methods

#### `retriever.py` - Hybrid Search
- **Pipeline**:
  1. Vector search (60% weight)
  2. Keyword/BM25 search (40% weight)
  3. Reciprocal Rank Fusion (RRF)
  4. Cohere reranking (top 20 → top k)
- **Result**: 40% better precision than pure semantic search

#### `citation.py` - Citation Formatting
- **Formats Supported**:
  - Short: `[Title, Section, Page]`
  - Long: Full document reference
  - Bluebook: Legal citation format
  - OSCOLA: Oxford citation format
- **Tracks**: Document title, hierarchy path, page numbers

#### `demo_app.py` - Streamlit Interface
- **Features**:
  - PDF upload via sidebar
  - Chat interface for queries
  - Expandable source display with citations
  - Document persistence across restarts
  - NVIDIA NIM for answer generation

#### `test_pipeline.py` - Validation
- Tests all components:
  - Chunking
  - Embeddings
  - Vector store operations
  - Full pipeline integration

---

## 4. Database Schema

### Prerequisites
```sql
-- Create database
CREATE DATABASE legal_rag;

-- Enable pgvector extension
CREATE EXTENSION vector;
```

### Tables

#### `legal_documents`
```sql
CREATE TABLE legal_documents (
    document_id UUID PRIMARY KEY,
    title TEXT NOT NULL,
    document_type VARCHAR(50),  -- contract, statute, case_law, regulation, brief, memo
    jurisdiction VARCHAR(100),
    page_count INTEGER,
    metadata JSONB,
    client_id VARCHAR(100),     -- For multi-tenant isolation
    created_at TIMESTAMP DEFAULT NOW()
);
```

#### `document_chunks`
```sql
CREATE TABLE document_chunks (
    chunk_id UUID PRIMARY KEY,
    document_id UUID REFERENCES legal_documents(document_id),
    content TEXT NOT NULL,
    embedding VECTOR(1024),     -- Cohere embed-v3 dimensions
    section_title TEXT,
    hierarchy_path TEXT,        -- e.g., "Document > Article III > Section 3.1"
    page_numbers TEXT,          -- e.g., "1-2" or "3"
    chunk_level INTEGER,        -- 0=summary, 1=section, 2=article, 3=paragraph
    metadata JSONB,
    client_id VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW()
);
```

### Indexes

```sql
-- Vector similarity search (IVFFlat)
CREATE INDEX ON document_chunks
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Full-text search (GIN)
CREATE INDEX ON document_chunks
USING gin (to_tsvector('english', content));

-- Client isolation
CREATE INDEX ON document_chunks (client_id);
CREATE INDEX ON legal_documents (client_id);
```

---

## 5. Environment Setup

### 1. Install PostgreSQL with pgvector

```bash
# macOS with Homebrew
brew install postgresql@17 pgvector
brew services start postgresql@17

# Create database
/opt/homebrew/opt/postgresql@17/bin/createdb legal_rag

# Enable vector extension
/opt/homebrew/opt/postgresql@17/bin/psql -d legal_rag -c "CREATE EXTENSION vector;"
```

### 2. Install Python Dependencies

```bash
pip install -r requirements_legal_rag.txt
```

### 3. Configure Environment Variables

```bash
# Copy template
cp .env.template .env

# Edit with your actual keys
nano .env
```

**Required variables:**
```
COHERE_API_KEY=your-cohere-api-key
NVIDIA_API_KEY=your-nvidia-api-key
POSTGRES_URL=postgresql://localhost:5432/legal_rag
```

### 4. Verify Setup

```bash
python -m execution.legal_rag.test_pipeline
```

Expected output:
```
✅ Chunking: PASSED
✅ Embeddings: PASSED
✅ Vector Store: PASSED
✅ Full Pipeline: PASSED
```

---

## 6. How to Start

### Quick Start (3 steps)

```bash
# 1. Start PostgreSQL
brew services start postgresql@17

# 2. Verify PostgreSQL is running
brew services list | grep postgresql

# 3. Run the Streamlit app
streamlit run execution/legal_rag/demo_app.py
```

**Demo URL:** http://localhost:8501

### Start on Specific Port

```bash
streamlit run execution/legal_rag/demo_app.py --server.port 8502
```

### Start in Headless Mode (for servers)

```bash
streamlit run execution/legal_rag/demo_app.py --server.headless true
```

### Verify Everything is Working

1. Open http://localhost:8501
2. Upload a PDF in the sidebar
3. Wait for "Processing complete"
4. Ask a question in the chat

---

## 7. How to Stop

### Stop Streamlit App

```bash
# Method 1: Kill by name
pkill -f streamlit

# Method 2: Find and kill by port
lsof -i :8501
kill -9 <PID>

# Method 3: If running in foreground
Ctrl+C
```

### Stop PostgreSQL

```bash
brew services stop postgresql@17
```

### Full Shutdown

```bash
# Stop app and database
pkill -f streamlit && brew services stop postgresql@17
```

### Restart App

```bash
# Quick restart
pkill -f streamlit && streamlit run execution/legal_rag/demo_app.py
```

### Clear Streamlit Cache and Restart

```bash
streamlit cache clear && streamlit run execution/legal_rag/demo_app.py
```

---

## 8. Usage Guide

### Using the Demo App

1. **Upload Documents**
   - Click "Browse files" in sidebar
   - Select PDF (contract, statute, case law, etc.)
   - Click "Process Document"
   - Wait for processing to complete

2. **Ask Questions**
   - Type in chat: "What are the termination clauses?"
   - View answer with inline citations [1], [2], etc.
   - Expand "Sources" to see exact passages

3. **Example Queries**
   - "What is the annual license fee?"
   - "How long do confidentiality obligations survive?"
   - "What happens if there's a breach?"
   - "Compare the termination provisions"

### Programmatic Usage

```python
from execution.legal_rag.document_parser import LegalDocumentParser
from execution.legal_rag.chunker import LegalChunker
from execution.legal_rag.embeddings import EmbeddingService
from execution.legal_rag.vector_store import VectorStore
from execution.legal_rag.retriever import HybridRetriever
from execution.legal_rag.citation import CitationExtractor

# Initialize components
parser = LegalDocumentParser()
chunker = LegalChunker()
embeddings = EmbeddingService()
store = VectorStore()
store.connect()
store.initialize_schema()

# Ingest a document
parsed = parser.parse("contract.pdf")
chunks = chunker.chunk(parsed)
chunk_embeddings = embeddings.embed_documents([c.content for c in chunks])

store.insert_document(
    document_id=parsed.metadata.document_id,
    title=parsed.metadata.title,
    document_type=parsed.metadata.document_type,
)
store.insert_chunks([c.to_dict() for c in chunks], chunk_embeddings)

# Query documents
retriever = HybridRetriever(store, embeddings)
results = retriever.retrieve("What are the termination clauses?", top_k=5)

# Format citations
extractor = CitationExtractor()
cited = extractor.extract(results)
for cc in cited:
    print(f"{cc.citation.short_format()}: {cc.content[:100]}...")
```

---

## 9. Cost Estimates

### Per Query

| Component | Cost |
|-----------|------|
| Cohere embedding (query) | ~$0.0001 |
| Cohere reranking (20 docs) | ~$0.001 |
| NVIDIA NIM (Llama 3.1 70B) | ~$0.005-0.01 |
| **Total per query** | **~$0.01-0.02** |

### Per Document Ingestion

| Document Size | Approx. Tokens | Embedding Cost |
|---------------|----------------|----------------|
| 10 pages | ~5,000 | ~$0.0005 |
| 50 pages | ~25,000 | ~$0.0025 |
| 100 pages | ~50,000 | ~$0.005 |

### Monthly (Production Scale)

| Component | 10K docs, 1K queries/day |
|-----------|--------------------------|
| Cohere embeddings | $40/mo |
| Cohere reranking | $80/mo |
| NVIDIA NIM | $150-200/mo |
| PostgreSQL (VPS) | $40/mo |
| **Total** | **~$310-360/mo** |

### Cost Comparison

| Solution | Monthly Cost | Notes |
|----------|--------------|-------|
| **This system** | ~$360 | Self-managed, full control |
| Pinecone + OpenAI | ~$800-1000 | Managed, less control |
| Enterprise RAG SaaS | $1500-3000+ | Fully managed |

---

## 10. Known Limitations

### Current Version (V1.0)

| Limitation | Description | Workaround |
|------------|-------------|------------|
| **Chat history not persisted** | Conversation resets on page refresh | Could add database storage |
| **No document deletion from UI** | Cannot remove documents via interface | Use PostgreSQL directly |
| **No multi-tenant auth** | RLS ready but not enforced | Add Supabase Auth in future |
| **Single-user mode** | Demo app for one user at a time | Add user sessions for production |
| **No streaming responses** | Full response generated before display | Implement SSE streaming |

### Technical Constraints

- **Max file size**: Depends on available memory (recommend <100 pages)
- **Supported formats**: PDF only (could add DOCX, TXT)
- **Languages**: English optimized (Cohere supports others)
- **Concurrent users**: Single user demo (need Redis for scale)

---

## 11. Troubleshooting

### Common Errors

#### "COHERE_API_KEY not found"
```bash
# Check .env file exists and has the key
cat .env | grep COHERE

# Solution: Add to .env
echo "COHERE_API_KEY=your-key" >> .env
```

#### "psql: command not found"
```bash
# Use full path
/opt/homebrew/opt/postgresql@17/bin/psql -d legal_rag

# Or add to PATH in ~/.zshrc
export PATH="/opt/homebrew/opt/postgresql@17/bin:$PATH"
source ~/.zshrc
```

#### "extension vector is not available"
```bash
# Install pgvector
brew install pgvector

# Restart PostgreSQL
brew services restart postgresql@17

# Re-enable extension
/opt/homebrew/opt/postgresql@17/bin/psql -d legal_rag -c "CREATE EXTENSION vector;"
```

#### "Port 8501 already in use"
```bash
# Kill existing Streamlit process
pkill -f streamlit

# Or use different port
streamlit run execution/legal_rag/demo_app.py --server.port 8502
```

#### "NVIDIA API 404 error"
The model name may have changed. Current working model:
```python
model="meta/llama-3.1-70b-instruct"
```

#### "Page numbers showing N/A"
Re-ingest documents after updating to V1.0. Old documents need reprocessing.

### Database Commands

```bash
# Connect to database
/opt/homebrew/opt/postgresql@17/bin/psql -d legal_rag

# List all documents
SELECT document_id, title, document_type, page_count FROM legal_documents;

# Count chunks
SELECT COUNT(*) FROM document_chunks;

# Delete a specific document
DELETE FROM document_chunks WHERE document_id = 'uuid-here';
DELETE FROM legal_documents WHERE document_id = 'uuid-here';

# Reset database (delete all data)
DROP TABLE IF EXISTS document_chunks, legal_documents CASCADE;
```

---

## 12. Future Roadmap

### Phase 3: Agentic Layer
- Query Understanding Agent (interprets user intent)
- Retrieval Planning Agent (decides search strategy)
- Citation Verification Agent (validates sources)
- Response Synthesis Agent (generates answers)

### Phase 4: Multi-Tenant Production
- Supabase Auth integration
- Full Row-Level Security policies
- GDPR compliance (deletion, export)
- Redis caching layer
- Document processing queue (Celery/RQ)

### Phase 5: API & Integration
- FastAPI REST endpoints
- Streaming responses (SSE)
- Webhook support for ingestion
- Admin dashboard
- Usage analytics

### Phase 6: Advanced Features
- Conflict detection across documents
- Timeline extraction
- Obligation tracking
- Custom fine-tuned models
- Multi-language support

---

## Appendix: Quick Reference

### Start Commands
```bash
brew services start postgresql@17
streamlit run execution/legal_rag/demo_app.py
```

### Stop Commands
```bash
pkill -f streamlit
brew services stop postgresql@17
```

### Test Commands
```bash
python -m execution.legal_rag.test_pipeline
```

### Database Access
```bash
/opt/homebrew/opt/postgresql@17/bin/psql -d legal_rag
```

### API Keys Required
- `COHERE_API_KEY` - https://dashboard.cohere.com/api-keys
- `NVIDIA_API_KEY` - https://build.nvidia.com/

---

**Document Version:** 1.0
**Last Updated:** February 3, 2026
**Author:** Built with Claude AI assistance
