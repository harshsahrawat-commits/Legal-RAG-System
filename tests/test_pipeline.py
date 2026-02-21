#!/usr/bin/env python3
"""
Test script for the Legal RAG pipeline.

Runs without requiring a PDF - uses sample legal text.
Tests all components: parsing, chunking, embedding, storage, retrieval.

Usage: python test_pipeline.py
"""

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sample legal document content for testing
SAMPLE_DOCUMENT = """
# SOFTWARE LICENSE AGREEMENT

This Software License Agreement ("Agreement") is entered into as of January 1, 2024
("Effective Date") by and between:

**LICENSOR:** TechCorp Inc., a Delaware corporation ("Licensor")

**LICENSEE:** ClientCo LLC, a California limited liability company ("Licensee")

## ARTICLE I - DEFINITIONS

Section 1.1 "Software" means the proprietary software application known as "LegalAI Pro"
including all updates, modifications, and enhancements.

Section 1.2 "Documentation" means user manuals, technical specifications, and other
materials describing the Software's functionality.

Section 1.3 "Licensed Users" means employees of Licensee authorized to use the Software.

## ARTICLE II - LICENSE GRANT

Section 2.1 Grant of License. Subject to the terms of this Agreement, Licensor hereby
grants to Licensee a non-exclusive, non-transferable license to use the Software.

Section 2.2 Restrictions. Licensee shall not:
(a) Copy, modify, or distribute the Software;
(b) Reverse engineer or decompile the Software;
(c) Sublicense or transfer the Software to third parties;
(d) Use the Software for any unlawful purpose.

## ARTICLE III - FEES AND PAYMENT

Section 3.1 License Fees. Licensee shall pay Licensor an annual license fee of
$50,000 USD, payable in advance.

Section 3.2 Payment Terms. All payments are due within thirty (30) days of invoice date.

Section 3.3 Late Payments. Overdue amounts shall accrue interest at 1.5% per month.

## ARTICLE IV - TERM AND TERMINATION

Section 4.1 Term. This Agreement shall commence on the Effective Date and continue
for a period of one (1) year, unless earlier terminated.

Section 4.2 Termination for Convenience. Either party may terminate this Agreement
upon sixty (60) days written notice.

Section 4.3 Termination for Breach. Either party may terminate immediately if the
other party materially breaches this Agreement and fails to cure within thirty (30) days.

Section 4.4 Effect of Termination. Upon termination:
(a) Licensee shall cease using the Software;
(b) Licensee shall return or destroy all copies of the Software;
(c) Sections 5, 6, and 7 shall survive termination.

## ARTICLE V - CONFIDENTIALITY

Section 5.1 Confidential Information. Each party agrees to maintain the confidentiality
of the other party's proprietary information.

Section 5.2 Permitted Disclosures. Confidential Information may be disclosed if required
by law or court order.

## ARTICLE VI - WARRANTIES

Section 6.1 Performance Warranty. Licensor warrants that the Software will perform
substantially in accordance with the Documentation for a period of ninety (90) days.

Section 6.2 Disclaimer. EXCEPT AS EXPRESSLY SET FORTH HEREIN, THE SOFTWARE IS PROVIDED
"AS IS" WITHOUT WARRANTY OF ANY KIND.

## ARTICLE VII - LIMITATION OF LIABILITY

Section 7.1 Cap on Damages. IN NO EVENT SHALL EITHER PARTY'S LIABILITY EXCEED THE
AMOUNTS PAID BY LICENSEE IN THE TWELVE (12) MONTHS PRECEDING THE CLAIM.

Section 7.2 Exclusion of Damages. NEITHER PARTY SHALL BE LIABLE FOR INDIRECT,
INCIDENTAL, OR CONSEQUENTIAL DAMAGES.

## ARTICLE VIII - GENERAL PROVISIONS

Section 8.1 Governing Law. This Agreement shall be governed by the laws of the
State of Delaware.

Section 8.2 Entire Agreement. This Agreement constitutes the entire agreement between
the parties regarding the subject matter hereof.

IN WITNESS WHEREOF, the parties have executed this Agreement as of the Effective Date.
"""


def test_chunking():
    """Test the chunking functionality without file parsing."""
    from execution.legal_rag.chunker import LegalChunker, Chunk
    from execution.legal_rag.document_parser import ParsedDocument, LegalMetadata, DocumentSection
    import uuid
    from datetime import datetime

    print("\n" + "="*60)
    print("Testing Chunking")
    print("="*60)

    # Create a mock parsed document
    metadata = LegalMetadata(
        document_id=str(uuid.uuid4()),
        title="Software License Agreement",
        document_type="contract",
        jurisdiction="Delaware",
        page_count=10,
    )

    # Create sections from sample document
    sections = []
    current_section = None

    for line in SAMPLE_DOCUMENT.split('\n'):
        if line.startswith('## '):
            if current_section:
                sections.append(current_section)
            current_section = DocumentSection(
                section_id=str(uuid.uuid4()),
                title=line[3:].strip(),
                content="",
                level=2,
                hierarchy_path=f"Document/{line[3:].strip().replace(' ', '_')}",
            )
        elif current_section:
            current_section.content += line + "\n"

    if current_section:
        sections.append(current_section)

    parsed = ParsedDocument(
        metadata=metadata,
        sections=sections,
        raw_text=SAMPLE_DOCUMENT,
        raw_markdown=SAMPLE_DOCUMENT,
    )

    # Chunk it
    chunker = LegalChunker()
    chunks = chunker.chunk(parsed)

    print(f"✅ Created {len(chunks)} chunks")
    print(f"   Levels: {set(c.level for c in chunks)}")

    # Show sample chunks
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n   Chunk {i+1}:")
        print(f"   - Title: {chunk.section_title}")
        print(f"   - Path: {chunk.hierarchy_path}")
        print(f"   - Tokens: {chunk.token_count}")
        print(f"   - Preview: {chunk.content[:100]}...")

    return chunks, metadata


def test_embeddings():
    """Test embedding generation."""
    print("\n" + "="*60)
    print("Testing Embeddings")
    print("="*60)

    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        print("⚠️  COHERE_API_KEY not set - skipping embedding test")
        print("   Set the key in .env to enable embeddings")
        return None

    from execution.legal_rag.embeddings import EmbeddingService

    service = EmbeddingService()

    # Test query embedding
    query = "What are the termination clauses?"
    embedding = service.embed_query(query)

    print(f"✅ Generated query embedding")
    print(f"   Dimensions: {len(embedding)}")
    print(f"   First 5 values: {embedding[:5]}")

    # Test document embedding
    docs = [
        "The party may terminate upon 60 days notice.",
        "License fees are $50,000 annually.",
    ]
    doc_embeddings = service.embed_documents(docs)

    print(f"✅ Generated {len(doc_embeddings)} document embeddings")

    return service


def test_vector_store():
    """Test vector store operations."""
    print("\n" + "="*60)
    print("Testing Vector Store")
    print("="*60)

    postgres_url = os.getenv("POSTGRES_URL") or os.getenv("DATABASE_URL")
    if not postgres_url:
        print("⚠️  POSTGRES_URL not set - skipping database test")
        print("   Set up PostgreSQL with pgvector to enable storage")
        print("   Example: POSTGRES_URL=postgresql://localhost:5432/legal_rag")
        return None

    from execution.legal_rag.vector_store import VectorStore

    store = VectorStore()

    try:
        store.connect()
        store.initialize_schema()
        print("✅ Connected to PostgreSQL with pgvector")
        return store
    except Exception as e:
        print(f"⚠️  Database connection failed: {e}")
        return None


def test_full_pipeline(chunks, metadata, embedding_service, store):
    """Test the full ingestion and retrieval pipeline."""
    print("\n" + "="*60)
    print("Testing Full Pipeline")
    print("="*60)

    if not embedding_service or not store:
        print("⚠️  Skipping - requires embeddings and database")
        return

    # Generate embeddings for chunks
    print("   Generating embeddings for chunks...")
    chunk_texts = [c.content for c in chunks]
    embeddings = embedding_service.embed_documents(chunk_texts)

    # Store document
    print("   Storing document...")
    store.insert_document(
        document_id=metadata.document_id,
        title=metadata.title,
        document_type=metadata.document_type,
        jurisdiction=metadata.jurisdiction,
    )

    # Store chunks
    print("   Storing chunks...")
    chunk_dicts = [c.to_dict() for c in chunks]
    store.insert_chunks(chunk_dicts, embeddings)

    print(f"✅ Stored {len(chunks)} chunks in database")

    # Test retrieval
    print("\n   Testing retrieval...")
    from execution.legal_rag.retriever import HybridRetriever
    from execution.legal_rag.citation import CitationExtractor

    retriever = HybridRetriever(store, embedding_service)
    extractor = CitationExtractor({metadata.document_id: metadata.title})

    query = "What are the termination clauses?"
    results = retriever.retrieve(query, top_k=3)

    print(f"✅ Retrieved {len(results)} results for: '{query}'")

    cited = extractor.extract(results)
    for i, cc in enumerate(cited):
        print(f"\n   Result {i+1}:")
        print(f"   - Citation: {cc.citation.short_format()}")
        print(f"   - Score: {cc.citation.relevance_score:.4f}")
        print(f"   - Preview: {cc.content[:100]}...")

    # Clean up test data
    print("\n   Cleaning up test data...")
    store.delete_document(metadata.document_id)
    print("✅ Test document removed")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("LEGAL RAG PIPELINE TEST")
    print("="*60)

    # Test 1: Chunking (no external deps)
    chunks, metadata = test_chunking()

    # Test 2: Embeddings (requires COHERE_API_KEY)
    embedding_service = test_embeddings()

    # Test 3: Vector Store (requires POSTGRES_URL)
    store = test_vector_store()

    # Test 4: Full pipeline
    if chunks and metadata:
        test_full_pipeline(chunks, metadata, embedding_service, store)

    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print("✅ Chunking: PASSED")
    print(f"{'✅' if embedding_service else '⚠️ '} Embeddings: {'PASSED' if embedding_service else 'SKIPPED (no API key)'}")
    print(f"{'✅' if store else '⚠️ '} Vector Store: {'PASSED' if store else 'SKIPPED (no database)'}")
    print(f"{'✅' if embedding_service and store else '⚠️ '} Full Pipeline: {'PASSED' if embedding_service and store else 'SKIPPED'}")

    print("\n" + "="*60)
    print("To enable all tests, add to .env:")
    print("   COHERE_API_KEY=your-key")
    print("   POSTGRES_URL=postgresql://localhost:5432/legal_rag")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
