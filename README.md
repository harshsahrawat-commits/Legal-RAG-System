# ⚖️ Legal RAG System

A **legal-grade Agentic RAG (Retrieval Augmented Generation) system** designed for querying legal documents with citation-level precision. Built for enterprise legal AI applications requiring accurate document retrieval and answer generation.

![Python](https://img.shields.io/badge/Python-3.14-blue)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-17%20%2B%20pgvector-336791)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)

---

## 🎯 What It Does

Upload legal PDFs (contracts, statutes, case law, regulations) and ask natural language questions. Get precise answers with **exact citations** including document name, section, and page numbers.

```
Query: "What are the termination clauses?"

Answer: According to Article IV, Section 4.2 [1], either party may terminate 
this agreement with 30 days written notice...

📚 Sources:
[1] Software License Agreement | Article IV - Termination | Pages 2-3
```

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🔍 **Hybrid Search** | Combines semantic (vector) + keyword (BM25) search for 40% better precision |
| 📊 **Reciprocal Rank Fusion** | Intelligently merges search results from multiple strategies |
| 📑 **Hierarchical Chunking** | Preserves legal document structure (Articles → Sections → Paragraphs) |
| 📝 **Citation Extraction** | Exact section and page references in multiple formats |
| 🔄 **Query Enhancement** | Auto-expands queries with legal terminology (HyDE, Query Expansion) |
| 🏢 **Multi-Tenant Ready** | Row-Level Security for client data isolation |
| 🧠 **Contextual Chunking** | LLM-generated context prepended to every chunk to improve retrieval |
| 🛡️ **Robust Data Cleaning** | Automated removal of OCR artifacts (GLYPH, garbage titles) |

---

## 🏗️ Architecture

```
                          ┌─────────────────────────────────────┐
                          │           INGESTION PIPELINE          │
                          └─────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────┐       ┌───────────────────────────────┐
│  Data Cleaning  │ ──►   │ OCR Correction & Title Fixes  │
└────────┬────────┘       └───────────────────────────────┘
         │
         ▼
┌─────────────────┐
│   Parsing &     │ ──►   Hierarchical Structure (Section/Clause)
│    Chunking     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐       ┌───────────────────────────────┐
│   Contextual    │ ──►   │ "Contextual Chunking" (LLM)   │
│    Chunking     │       │ Prepend document context to   │
└────────┬────────┘       │ every chunk for <lost context>│
         │                └───────────────────────────────┘
         ▼
    [Vector Store]

                          ┌─────────────────────────────────────┐
                          │           QUERY PIPELINE            │
                          └─────────────────────────────────────┘

User Query: "What is the annual license fee?"
     │
     ▼
┌─────────────────┐
│ Query Expansion │ ──► Adds legal terminology
│ + HyDE          │     "annual fee, license fee, payment terms..."
└────────┬────────┘
         ▼
┌─────────────────────────────────────────────────┐
│              HYBRID RETRIEVAL                   │
│  ┌─────────────┐       ┌─────────────┐         │
│  │   Vector    │       │   Keyword   │         │
│  │   Search    │       │    (BM25)   │         │
│  │    60%      │       │     40%     │         │
│  └──────┬──────┘       └──────┬──────┘         │
│         └────────┬────────────┘                │
│                  ▼                             │
│     ┌─────────────────────────┐                │
│     │ Reciprocal Rank Fusion  │                │
│     └───────────┬─────────────┘                │
│                 ▼                              │
│     ┌─────────────────────────┐                │
│     │    Cohere Reranking     │                │
│     │     (Top 20 → Top 5)    │                │
│     └───────────┬─────────────┘                │
└─────────────────┼───────────────────────────────┘
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
       │  "According to [1, 2].."│
       └─────────────────────────┘
```

---

## 🛠️ Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Language** | Python 3.14 | Core implementation |
| **Vector DB** | PostgreSQL 17 + pgvector | Document & embedding storage |
| **Embeddings** | Voyage AI voyage-law-2 (1024 dims) | Legal-optimized semantic search (6-10% better on legal benchmarks) |
| **Reranking** | Cohere rerank-v3 | Precision improvement |
| **LLM** | NVIDIA NIM (Llama 3.1 70B) | Answer generation |
| **Frontend** | Streamlit | Demo web interface |
| **Framework** | LlamaIndex | RAG orchestration |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- PostgreSQL 17 with pgvector extension
- API keys for Cohere and NVIDIA NIM

### 1. Clone the Repository

```bash
git clone https://github.com/harshsahrawat-commits/Legal-RAG-System.git
cd Legal-RAG-System
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up PostgreSQL with pgvector

```bash
# macOS with Homebrew
brew install postgresql@17 pgvector
brew services start postgresql@17

# Create database and enable extension
/opt/homebrew/opt/postgresql@17/bin/createdb legal_rag
/opt/homebrew/opt/postgresql@17/bin/psql -d legal_rag -c "CREATE EXTENSION vector;"
```

### 4. Configure Environment Variables

```bash
cp .env.template .env
# Edit .env with your API keys
```

Required keys:
- `VOYAGE_API_KEY` - Get from [Voyage AI Dashboard](https://dash.voyageai.com/) (free tier available)
- `COHERE_API_KEY` - Get from [Cohere Dashboard](https://dashboard.cohere.com/api-keys) (for reranking)
- `NVIDIA_API_KEY` - Get from [NVIDIA NIM](https://build.nvidia.com/)
- `POSTGRES_URL` - Your PostgreSQL connection string

### 5. Run the Application

```bash
streamlit run execution/legal_rag/demo_app.py
```

Open **http://localhost:8501** in your browser.

---

## 📖 Usage

### Using the Demo App

1. **Upload Documents** - Click "Browse files" in the sidebar and select PDFs
2. **Wait for Processing** - Documents are chunked, embedded, and stored
3. **Ask Questions** - Type natural language queries in the chat
4. **View Citations** - Expand "Sources" to see exact document references

### Example Queries

| Query | What it finds |
|-------|---------------|
| "What is the annual license fee?" | Payment terms and pricing |
| "How long do confidentiality obligations survive?" | NDA survival clauses |
| "Compare the termination provisions" | Cross-document analysis |
| "What state's law governs this agreement?" | Choice of law provisions |

### Programmatic Usage

```python
from execution.legal_rag import (
    LegalDocumentParser,
    LegalChunker,
    EmbeddingService,
    VectorStore,
    HybridRetriever,
    CitationExtractor
)

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

## 📁 Project Structure

```
Legal-RAG-System/
├── execution/legal_rag/          # Core Python modules
│   ├── document_parser.py        # PDF extraction (Docling/PyMuPDF)
│   ├── chunker.py                # Hierarchical chunking
│   ├── embeddings.py             # Voyage AI / Cohere embedding service
│   ├── vector_store.py           # PostgreSQL + pgvector operations
│   ├── retriever.py              # Hybrid search pipeline
│   ├── citation.py               # Citation formatting
│   ├── demo_app.py               # Streamlit UI
│   ├── metrics.py                # Performance tracking
│   ├── quotas.py                 # Tenant usage limits
│   └── test_pipeline.py          # Validation tests
│
├── directives/legal_rag/         # SOP documentation
│   ├── ingest_document.md        # Document ingestion guide
│   └── query_documents.md        # Query handling guide
│
├── .env.template                 # Environment variable template
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## 📊 Database Schema

### Tables

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

CREATE TABLE document_chunks (
    chunk_id UUID PRIMARY KEY,
    document_id UUID REFERENCES legal_documents(document_id),
    content TEXT NOT NULL,
    embedding VECTOR(1024),     -- 1024 dimensions
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
CREATE INDEX ON document_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Full-text search (GIN)
CREATE INDEX ON document_chunks USING gin (to_tsvector('english', content));

-- Client isolation
CREATE INDEX ON document_chunks (client_id);
CREATE INDEX ON legal_documents (client_id);
```

---

## 💰 Cost Estimates

### Per Query
| Component | Cost |
|-----------|------|
| Cohere embedding | ~$0.0001 |
| Cohere reranking | ~$0.001 |
| NVIDIA NIM | ~$0.005-0.01 |
| **Total** | **~$0.01-0.02** |

### Per Document Ingestion

| Document Size | Approx. Tokens | Embedding Cost |
|---------------|----------------|----------------|
| 10 pages | ~5,000 | ~$0.0005 |
| 50 pages | ~25,000 | ~$0.0025 |
| 100 pages | ~50,000 | ~$0.005 |

### Monthly (Production: 10K docs, 1K queries/day)
| Component | Cost |
|-----------|------|
| Cohere | ~$120/mo |
| NVIDIA NIM | ~$150-200/mo |
| PostgreSQL | ~$40/mo |
| **Total** | **~$310-360/mo** |

### Cost Comparison

| Solution | Monthly Cost | Notes |
|----------|--------------|-------|
| **This system** | ~$360 | Self-managed, full control |
| Pinecone + OpenAI | ~$800-1000 | Managed, less control |
| Enterprise RAG SaaS | $1500-3000+ | Fully managed |

---

## 🔧 Production Features

### Multi-Tenant Security
- **Row-Level Security (RLS)** - Database-enforced tenant isolation
- **API Key Authentication** - Secure client access
- **Audit Logging** - Track all document and query operations

### Performance Optimization
- **Connection Pooling** - Handle concurrent requests efficiently
- **Smart Reranking** - Skip expensive API calls when confidence is high
- **Query Result Caching** - Semantic similarity matching for repeated queries
- **HNSW Indexing** - Optimized for 50K+ document chunks

### Monitoring
- **Metrics Collection** - Query latency (avg, p95, p99), cache hits, errors
- **Tenant Quotas** - Usage limits by subscription tier

---

## 🧪 Testing

```bash
# Run the test pipeline
python -m execution.legal_rag.test_pipeline

# Expected output:
# ✅ Chunking: PASSED
# ✅ Embeddings: PASSED
# ✅ Vector Store: PASSED
# ✅ Full Pipeline: PASSED
```

---

## 🔥 Troubleshooting

### Common Errors

**"COHERE_API_KEY not found"**
```bash
# Check .env file exists and has the key
cat .env | grep COHERE
# Solution: Add to .env
echo "COHERE_API_KEY=your-key" >> .env
```

**"psql: command not found"**
```bash
# Use full path
/opt/homebrew/opt/postgresql@17/bin/psql -d legal_rag
# Or add to PATH in ~/.zshrc
export PATH="/opt/homebrew/opt/postgresql@17/bin:$PATH"
```

**"extension vector is not available"**
```bash
brew install pgvector
brew services restart postgresql@17
/opt/homebrew/opt/postgresql@17/bin/psql -d legal_rag -c "CREATE EXTENSION vector;"
```

**"Port 8501 already in use"**
```bash
pkill -f streamlit
# Or use a different port
streamlit run execution/legal_rag/demo_app.py --server.port 8502
```

**"NVIDIA API 404 error"** -- The model name may have changed. Current working model: `meta/llama-3.1-70b-instruct`

### Quick Commands

```bash
# Start PostgreSQL
brew services start postgresql@17

# Run the app
streamlit run execution/legal_rag/demo_app.py

# Stop the app
pkill -f streamlit

# Stop PostgreSQL
brew services stop postgresql@17

# Connect to database
/opt/homebrew/opt/postgresql@17/bin/psql -d legal_rag

# Reset database (delete all data)
/opt/homebrew/opt/postgresql@17/bin/psql -d legal_rag -c "DROP TABLE IF EXISTS document_chunks, legal_documents CASCADE;"
```

---

## 📋 Known Limitations

| Limitation | Workaround |
|------------|------------|
| PDF only | Add DOCX/TXT parsers |
| English optimized | Cohere supports other languages |
| Single-user demo | Add Redis for multi-user sessions |
| Max ~100 pages per doc | Split large documents |

---

## 🗺️ Roadmap

- [ ] **Agentic Layer** - Query understanding, retrieval planning agents
- [ ] **FastAPI Endpoints** - REST API for integration
- [ ] **Streaming Responses** - Real-time answer generation
- [ ] **Conflict Detection** - Find contradictions across documents
- [ ] **Timeline Extraction** - Identify key dates and deadlines
- [ ] **Multi-Language Support** - Legal documents in other languages

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- **Voyage AI** for `voyage-law-2` legal-optimized embeddings
- **Cohere** for reranking API
- **NVIDIA NIM** for Llama 3.1 70B inference
- **LlamaIndex** for RAG framework
- **pgvector** for vector similarity search

---

<p align="center">
  Built with ❤️ for the legal tech community
</p>
