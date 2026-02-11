"""
Pydantic models for the Legal RAG FastAPI backend.
"""

from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request body for RAG query endpoint."""
    query: str = Field(..., min_length=1, max_length=2000)
    document_id: Optional[str] = None
    top_k: int = Field(default=10, ge=1, le=50)


class SourceInfo(BaseModel):
    """Citation source in a query response."""
    document_title: str
    section: str
    page_numbers: list[int]
    hierarchy_path: str
    chunk_id: str
    document_id: str
    relevance_score: float
    short_citation: str
    long_citation: str
    content: str = ""
    context_before: Optional[str] = ""
    context_after: Optional[str] = ""


class QueryResponse(BaseModel):
    """Response body for RAG query endpoint."""
    answer: str
    sources: list[SourceInfo]
    latency_ms: float


class DocumentInfo(BaseModel):
    """Information about a stored document."""
    id: str
    title: str
    document_type: str
    jurisdiction: Optional[str] = None
    page_count: int = 0
    chunks: Optional[int] = None
    created_at: Optional[str] = None


class UploadResponse(BaseModel):
    """Response body for document upload."""
    id: str
    title: str
    document_type: str
    jurisdiction: Optional[str] = None
    page_count: int
    chunks: int


class TenantConfigUpdate(BaseModel):
    """Request body for updating tenant configuration."""
    language: Optional[str] = Field(None, pattern=r"^(en|el)$")
    embedding_model: Optional[str] = None
    embedding_provider: Optional[str] = None
    llm_model: Optional[str] = None
    reranker_model: Optional[str] = None


class TenantConfigResponse(BaseModel):
    """Response body for tenant configuration."""
    language: str
    embedding_model: str
    embedding_provider: str
    llm_model: str
    reranker_model: str
    fts_language: str


class HealthResponse(BaseModel):
    """Response body for health check."""
    status: str
    version: str
    database: str
