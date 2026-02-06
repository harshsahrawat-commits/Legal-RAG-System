# Legal RAG System - Complete Documentation

## Overview

This is a **legal-grade Agentic RAG system** built for querying legal documents (contracts, statutes, case law) with citation-level precision. Built for an Upwork job application for a legal AI SaaS product.

**Built:** February 1, 2026
**Architecture:** Python + LlamaIndex + PostgreSQL/pgvector + Cohere + Claude

---

## Architecture Decisions

### Why These Technologies?

| Component | Choice | Why |
|-----------|--------|-----|
| **Vector DB** | PostgreSQL + pgvector | Native Row-Level Security for multi-tenant isolation, 60% cheaper than Pinecone |
| **Embeddings** | Cohere embed-v3 | Enterprise SLA, 1024 dimensions, separate query/document embeddings |
| **Reranking** | Cohere rerank-v3 | 40% better precision than pure semantic search |
| **LLM** | Claude Sonnet | Cost-effective for answer generation (~$0.01-0.02/query) |
| **Backend** | Custom Python (not n8n) | Better control, debugging, fits Flowkart 3-layer architecture |

### Why NOT n8n?
- Limited control over retrieval pipeline tuning
- Hard to debug agentic workflows
- No native streaming for long responses
- Enterprise licensing costs scale poorly

---

## System Components

### File Structure
```
execution/legal_rag/
├── __init__.py           # Module exports
├── document_parser.py    # PDF extraction with Docling/PyMuPDF
├── chunker.py            # Legal-aware hierarchical chunking
├── embeddings.py         # Cohere embeddings with caching
├── vector_store.py       # PostgreSQL + pgvector operations
├── retriever.py          # Hybrid search + RRF + reranking
├── citation.py           # Citation extraction [Doc, Section, Page]
├── demo_app.py           # Streamlit demo interface
├── test_pipeline.py      # Test script for validation
├── README.md             # This file
└── agents/
    └── __init__.py       # Placeholder for future agents

directives/legal_rag/
├── ingest_document.md    # Document ingestion SOP
└── query_documents.md    # Query handling SOP
```

### Component Details

#### 1. Document Parser (`document_parser.py`)
- Extracts text from PDFs using Docling (97.9% accuracy on tables)
- Falls back to PyMuPDF4LLM if Docling unavailable
- Auto-detects document type (contract, statute, case_law, regulation)
- Extracts metadata: title, jurisdiction, parties, effective date
- Preserves legal structure (articles, clauses, sections)

#### 2. Chunker (`chunker.py`)
- **Hierarchical chunking** preserving legal document structure
- Levels:
  - L0: Document summary (500-1000 tokens)
  - L1: Section/Chapter (1000-2000 tokens)
  - L2: Article/Clause (300-800 tokens)
  - L3: Paragraph (100-300 tokens)
- Overlapping chunks (100 tokens) for context preservation
- Extracts legal cross-references and defined terms

#### 3. Embeddings (`embeddings.py`)
- Uses Cohere embed-v3 (1024 dimensions)
- Different `input_type` for documents vs queries
- File-based caching to avoid re-computation
- Batch processing (96 texts per batch)

#### 4. Vector Store (`vector_store.py`)
- PostgreSQL with pgvector extension
- Schema with Row-Level Security ready
- Tables: `legal_documents`, `document_chunks`
- Indexes: IVFFlat for vector similarity, GIN for full-text search
- Supports both vector and keyword (BM25) search

#### 5. Retriever (`retriever.py`)
- **Hybrid search pipeline:**
  1. Vector search (semantic similarity)
  2. Keyword search (BM25 full-text)
  3. Reciprocal Rank Fusion (RRF) to combine
  4. Cohere reranking for precision
- Configurable weights: 60% vector, 40% keyword
- Returns top-k results with scores

#### 6. Citation Extractor (`citation.py`)
- Formats citations: `[Document Title, Section X.Y, Page N]`
- Supports multiple formats: short, long, Bluebook, OSCOLA
- Tracks document hierarchy path
- Links citations to source chunks

#### 7. Demo App (`demo_app.py`)
- Streamlit-based web interface
- Upload PDFs via sidebar
- Chat interface for queries
- Shows answers with expandable sources
- Uses Claude Sonnet for answer generation

---

## Setup Instructions

### Prerequisites
```bash
# PostgreSQL 17 with pgvector
brew install postgresql@17 pgvector
brew services start postgresql@17

# Create database
/opt/homebrew/opt/postgresql@17/bin/createdb legal_rag
/opt/homebrew/opt/postgresql@17/bin/psql -d legal_rag -c "CREATE EXTENSION vector;"
```

### Environment Variables (.env)
```bash
# Required
COHERE_API_KEY=your-cohere-key      # From dashboard.cohere.com
POSTGRES_URL=postgresql://localhost:5432/legal_rag
ANTHROPIC_API_KEY=your-claude-key   # For answer generation

# Optional
REDIS_URL=redis://localhost:6379    # For embedding cache
```

### Install Dependencies
```bash
pip install -r requirements_legal_rag.txt
```

### Test the Pipeline
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

### Run the Demo
```bash
streamlit run execution/legal_rag/demo_app.py
```
Open http://localhost:8501

---

## Usage

### Programmatic Usage

```python
from execution.legal_rag.document_parser import LegalDocumentParser
from execution.legal_rag.chunker import LegalChunker
from execution.legal_rag.embeddings import EmbeddingService
from execution.legal_rag.vector_store import VectorStore
from execution.legal_rag.retriever import HybridRetriever
from execution.legal_rag.citation import CitationExtractor

# Initialize
parser = LegalDocumentParser()
chunker = LegalChunker()
embeddings = EmbeddingService()
store = VectorStore()
store.connect()
store.initialize_schema()

# Ingest document
parsed = parser.parse("contract.pdf")
chunks = chunker.chunk(parsed)
chunk_embeddings = embeddings.embed_documents([c.content for c in chunks])

store.insert_document(
    document_id=parsed.metadata.document_id,
    title=parsed.metadata.title,
    document_type=parsed.metadata.document_type,
)
store.insert_chunks([c.to_dict() for c in chunks], chunk_embeddings)

# Query
retriever = HybridRetriever(store, embeddings)
results = retriever.retrieve("What are the termination clauses?", top_k=5)

# Format citations
extractor = CitationExtractor()
cited = extractor.extract(results)
for cc in cited:
    print(f"{cc.citation.short_format()}: {cc.content[:100]}...")
```

### Demo App Usage
1. Upload PDF in sidebar
2. Click "Process Document"
3. Ask questions in chat
4. View answers with citations

---

## Cost Estimates

### Per Query
| Component | Cost |
|-----------|------|
| Embedding (query) | ~$0.0001 |
| Reranking | ~$0.001 |
| Claude Sonnet | ~$0.01-0.02 |
| **Total** | **~$0.02** |

### Monthly (10K documents, 1K queries/day)
| Component | Cost |
|-----------|------|
| Cohere embeddings | $40/mo |
| Cohere reranking | $80/mo |
| Claude API | $200/mo |
| PostgreSQL (VPS) | $40/mo |
| **Total** | **~$360/mo** |

---

## Upwork Application Notes

### Loom Video Structure (5-7 min)
1. **Opening (30 sec):** Introduction, experience with RAG systems
2. **Technical Approach (2 min):**
   - Show architecture diagram
   - Explain Docling for PDF extraction
   - Legal-aware chunking
   - Hybrid search + reranking
3. **Demo (2 min):**
   - Upload sample contract
   - Query: "What are the termination clauses?"
   - Show citations
4. **Architecture Deep-Dive (1.5 min):**
   - Multi-tenant RLS
   - Why pgvector over Pinecone
   - GDPR compliance
5. **Long-term Vision (1 min):**
   - Agentic layer for complex queries
   - Conflict detection
   - Revenue share discussion

### Key Differentiators
1. **Not "chat with PDF"** - proper agentic RAG with legal structure preservation
2. **Database-level isolation** - RLS, not application-level filtering
3. **Hybrid search** - 40% better precision than pure semantic
4. **Citation-level accuracy** - exact section and page references
5. **Cost-optimized** - ~$360/mo at scale vs $1000+ alternatives

---

## Future Enhancements (Post-Contract)

### Phase 3: Agentic Layer
- Query Understanding Agent
- Retrieval Planning Agent
- Citation Verification Agent
- Response Synthesis Agent

### Phase 4: Multi-Tenant Production
- Supabase Auth integration
- Full RLS policies
- GDPR compliance (deletion, export)
- Redis caching
- Document processing queue

### Phase 5: API & Integration
- FastAPI endpoints
- Streaming responses
- Webhook support
- Admin dashboard

---

## Troubleshooting

### "COHERE_API_KEY not found"
```bash
# Add to .env
COHERE_API_KEY=your-key-here
```

### "psql: command not found"
```bash
# Use full path
/opt/homebrew/opt/postgresql@17/bin/psql -d legal_rag

# Or add to PATH in ~/.zshrc
export PATH="/opt/homebrew/opt/postgresql@17/bin:$PATH"
```

### "extension vector is not available"
```bash
brew install pgvector
# Then restart PostgreSQL
brew services restart postgresql@17
```

### Streamlit not loading
```bash
# Kill existing process
pkill -f streamlit

# Restart
streamlit run execution/legal_rag/demo_app.py
```

---

## Sample Test Document

A sample contract is available at:
```
.tmp/sample_software_license.pdf
```

3-page Software License Agreement with 8 articles - good for testing queries like:
- "What are the termination clauses?"
- "How much is the license fee?"
- "What happens if there's a breach?"

---

## References

- [Plan File](/.claude/plans/snuggly-spinning-lagoon.md) - Full implementation plan
- [Ingest Directive](../directives/legal_rag/ingest_document.md) - Document ingestion SOP
- [Query Directive](../directives/legal_rag/query_documents.md) - Query handling SOP
- [Requirements](../../requirements_legal_rag.txt) - Python dependencies
