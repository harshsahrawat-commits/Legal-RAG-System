# Query Legal Documents

## Goal
Retrieve relevant information from legal documents and generate cited answers.

## Inputs
- `query`: Natural language question about the documents
- `client_id` (optional): Client identifier for multi-tenant filtering
- `document_id` (optional): Limit search to specific document
- `top_k` (optional): Number of results to return (default: 10)

## Process

### Step 1: Embed Query
```python
from execution.legal_rag.embeddings import EmbeddingService

embeddings = EmbeddingService()
query_embedding = embeddings.embed_query(query)
```

Uses `input_type="search_query"` for better query-document matching.

### Step 2: Hybrid Retrieval
```python
from execution.legal_rag.retriever import HybridRetriever

retriever = HybridRetriever(store, embeddings)
results = retriever.retrieve(
    query=query,
    client_id=client_id,
    document_id=document_id,
    top_k=top_k,
)
```

Pipeline:
1. Vector search (semantic similarity)
2. Keyword search (BM25)
3. Reciprocal Rank Fusion
4. Cohere reranking

### Step 3: Extract Citations
```python
from execution.legal_rag.citation import CitationExtractor

extractor = CitationExtractor(document_titles)
cited_contents = extractor.extract(results)
```

Creates formatted citations: `[Document Title, Section X.Y, Page N]`

### Step 4: Generate Response (Optional)
Use Claude for answer synthesis:
```python
import anthropic

client = anthropic.Anthropic()

context = "\n\n".join([
    f"[{i+1}] {cc.citation.short_format()}: {cc.content}"
    for i, cc in enumerate(cited_contents)
])

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1500,
    system="You are a legal research assistant. Cite sources as [N].",
    messages=[{
        "role": "user",
        "content": f"Based on these sources, answer: {query}\n\n{context}"
    }],
)
```

## Outputs
- `results`: List of relevant chunks with scores
- `citations`: Formatted citations for each result
- `answer` (if LLM used): Generated response with inline citations

## Search Strategies

### For Clause Lookup
```python
# Direct keyword match for specific clauses
results = retriever.retrieve(
    query="termination clause section 8",
    top_k=5,
)
```

### For Concept Search
```python
# Semantic search for related concepts
results = retriever.retrieve(
    query="what happens if either party wants to end the agreement early",
    top_k=10,
)
```

### For Comparison
```python
# Search multiple documents
results_a = retriever.retrieve(query=query, document_id=doc_a_id)
results_b = retriever.retrieve(query=query, document_id=doc_b_id)
```

## Edge Cases
- **No results**: Check if documents are ingested, try broader query
- **Low scores**: Results may not be relevant, show confidence warning
- **Conflicting sources**: Note dates, prefer newer unless grandfathered

## Performance
- Vector search: ~50ms for 100K chunks
- Keyword search: ~30ms for 100K chunks
- Reranking: ~200ms per 20 candidates
- Total: ~300-500ms typical

## Example
```python
from execution.legal_rag.embeddings import EmbeddingService
from execution.legal_rag.vector_store import VectorStore
from execution.legal_rag.retriever import HybridRetriever
from execution.legal_rag.citation import CitationExtractor

# Initialize
store = VectorStore()
store.connect()
embeddings = EmbeddingService()
retriever = HybridRetriever(store, embeddings)
extractor = CitationExtractor()

# Query
query = "What are the termination clauses?"
results = retriever.retrieve(query, top_k=5)

# Format with citations
cited = extractor.extract(results)
for cc in cited:
    print(f"\n{cc.citation.short_format()}")
    print(cc.content[:200])
```
