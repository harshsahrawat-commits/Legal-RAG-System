# Ingest Legal Document

## Goal
Process a legal PDF document and store it in the vector database for querying.

## Inputs
- `file_path`: Path to the PDF file
- `client_id` (optional): Client identifier for multi-tenant isolation
- `document_type` (optional): Override auto-detection (contract, statute, case_law, regulation, brief, memo)

## Process

### Step 1: Parse Document
Use `execution/legal_rag/document_parser.py`:
```python
from execution.legal_rag.document_parser import LegalDocumentParser

parser = LegalDocumentParser()
parsed = parser.parse(file_path)
```

This extracts:
- Document structure (articles, clauses, sections)
- Metadata (title, type, jurisdiction, parties)
- Raw text and markdown

### Step 2: Chunk Document
Use `execution/legal_rag/chunker.py`:
```python
from execution.legal_rag.chunker import LegalChunker

chunker = LegalChunker()
chunks = chunker.chunk(parsed)
```

Creates hierarchical chunks:
- L0: Document summary (500-1000 tokens)
- L1: Section/Chapter (1000-2000 tokens)
- L2: Article/Clause (300-800 tokens)
- L3: Paragraph (100-300 tokens)

### Step 3: Generate Embeddings
Use `execution/legal_rag/embeddings.py`:
```python
from execution.legal_rag.embeddings import EmbeddingService

embeddings_service = EmbeddingService()
chunk_texts = [c.content for c in chunks]
embeddings = embeddings_service.embed_documents(chunk_texts)
```

### Step 4: Store in Vector Database
Use `execution/legal_rag/vector_store.py`:
```python
from execution.legal_rag.vector_store import VectorStore

store = VectorStore()
store.connect()
store.initialize_schema()

# Store document metadata
store.insert_document(
    document_id=parsed.metadata.document_id,
    title=parsed.metadata.title,
    document_type=parsed.metadata.document_type,
    client_id=client_id,
)

# Store chunks with embeddings
chunk_dicts = [c.to_dict() for c in chunks]
store.insert_chunks(chunk_dicts, embeddings, client_id=client_id)
```

## Outputs
- `document_id`: UUID of the stored document
- `chunk_count`: Number of chunks created
- `metadata`: Extracted document metadata

## Edge Cases
- **Large PDFs (100+ pages)**: Process in batches of 50 pages
- **Scanned documents**: Docling handles OCR automatically
- **Complex tables**: Docling extracts with 97.9% accuracy
- **Missing structure**: Creates single document-level chunk

## Errors
- `FileNotFoundError`: PDF file doesn't exist
- `ImportError`: Docling/PyMuPDF not installed
- `DatabaseError`: PostgreSQL connection failed
- `ValueError`: Invalid PDF format

## Cost Estimates
- Embedding: ~$0.10 per 1M tokens (Cohere)
- Typical 100-page contract: ~50K tokens = ~$0.005

## Example
```python
# Full ingestion pipeline
from execution.legal_rag.document_parser import LegalDocumentParser
from execution.legal_rag.chunker import LegalChunker
from execution.legal_rag.embeddings import EmbeddingService
from execution.legal_rag.vector_store import VectorStore

# Initialize
parser = LegalDocumentParser()
chunker = LegalChunker()
embeddings = EmbeddingService()
store = VectorStore()
store.connect()
store.initialize_schema()

# Process
parsed = parser.parse("contract.pdf")
chunks = chunker.chunk(parsed)
chunk_embeddings = embeddings.embed_documents([c.content for c in chunks])

# Store
store.insert_document(
    document_id=parsed.metadata.document_id,
    title=parsed.metadata.title,
    document_type=parsed.metadata.document_type,
)
store.insert_chunks([c.to_dict() for c in chunks], chunk_embeddings)

print(f"Ingested {len(chunks)} chunks from {parsed.metadata.title}")
```
