"""
Legal RAG System - Agentic RAG for Legal Documents

This module provides a production-ready system for:
- Ingesting large legal documents (PDFs, contracts, case law)
- Legal-aware chunking that preserves document structure
- Hybrid search with citation extraction
- Multi-tenant isolation for law firms

Architecture follows the Flowkart 3-layer pattern:
- Layer 1 (Directives): SOPs in directives/legal_rag/
- Layer 2 (Orchestration): Claude Code / main orchestrator
- Layer 3 (Execution): This module and its submodules
"""

from .document_parser import LegalDocumentParser
from .chunker import LegalChunker
from .embeddings import EmbeddingService
from .vector_store import VectorStore
from .retriever import HybridRetriever
from .citation import CitationExtractor

__all__ = [
    "LegalDocumentParser",
    "LegalChunker",
    "EmbeddingService",
    "VectorStore",
    "HybridRetriever",
    "CitationExtractor",
]

__version__ = "0.1.0"
