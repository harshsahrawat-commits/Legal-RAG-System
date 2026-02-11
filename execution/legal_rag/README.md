# Legal RAG System - Module Documentation

> See the [project README](../../README.md) for full documentation including setup, usage, architecture, database schema, cost estimates, and troubleshooting.

---

## Architecture Decisions

### Why These Technologies?

| Component | Choice | Why |
|-----------|--------|-----|
| **Vector DB** | PostgreSQL + pgvector | Native Row-Level Security for multi-tenant isolation, 60% cheaper than Pinecone |
| **Embeddings** | Voyage AI voyage-law-2 | 6-10% better retrieval on legal benchmarks, 1024 dimensions, document/query types |
| **Reranking** | Cohere rerank-v3 | 40% better precision than pure semantic search |
| **LLM** | NVIDIA NIM (Llama 3.1 70B) | Cost-effective for answer generation (~$0.01-0.02/query) |
| **Backend** | Custom Python (not n8n) | Better control, debugging, fits 3-layer architecture |

### Why NOT n8n?
- Limited control over retrieval pipeline tuning
- Hard to debug agentic workflows
- No native streaming for long responses
- Enterprise licensing costs scale poorly

---

## Key Differentiators

1. **Not "chat with PDF"** -- proper agentic RAG with legal structure preservation
2. **Database-level isolation** -- RLS, not application-level filtering
3. **Hybrid search** -- 40% better precision than pure semantic
4. **Citation-level accuracy** -- exact section and page references
5. **Cost-optimized** -- ~$360/mo at scale vs $1000+ alternatives
