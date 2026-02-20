"""
Pydantic models for the Legal RAG FastAPI backend.
"""

from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field


class SourceToggles(BaseModel):
    """Controls which legal databases to include in a query."""
    cylaw: bool = True
    hudoc: bool = True
    eurlex: bool = True
    families: list[str] = []  # family UUIDs to include


class QueryRequest(BaseModel):
    """Request body for RAG query endpoint."""
    query: str = Field(..., min_length=1, max_length=2000)
    document_id: Optional[str] = None
    top_k: int = Field(default=10, ge=1, le=50)
    sources: SourceToggles = SourceToggles()


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
    cylaw_url: Optional[str] = None
    source_origin: str = "cylaw"
    external_url: Optional[str] = None


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
    cylaw_url: Optional[str] = None
    family_id: Optional[str] = None


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


# =========================================================================
# Auth models
# =========================================================================

class GoogleAuthRequest(BaseModel):
    """Request body for Google OAuth token exchange."""
    id_token: str


class AuthResponse(BaseModel):
    """Response body for auth endpoints."""
    token: str
    user: "UserInfo"


class UserInfo(BaseModel):
    """User profile information."""
    id: str
    email: str
    name: Optional[str] = None
    avatar_url: Optional[str] = None


# =========================================================================
# Conversation models
# =========================================================================

class ConversationCreate(BaseModel):
    """Request body for creating a conversation."""
    title: str = "New Chat"


class ConversationResponse(BaseModel):
    """Response body for a conversation."""
    id: str
    title: str
    created_at: str
    updated_at: str


class ConversationRename(BaseModel):
    """Request body for renaming a conversation."""
    title: str = Field(..., min_length=1, max_length=500)


class MessageResponse(BaseModel):
    """A single message in a conversation."""
    id: str
    role: str
    content: str
    sources: Optional[list] = None
    latency_ms: Optional[float] = None
    created_at: str


class StreamQueryRequest(BaseModel):
    """Request body for streaming query with conversation support."""
    query: str = Field(..., min_length=1, max_length=2000)
    conversation_id: Optional[str] = None
    document_id: Optional[str] = None
    top_k: int = Field(default=10, ge=1, le=50)
    sources: SourceToggles = SourceToggles()


# =========================================================================
# Document Family models
# =========================================================================

class FamilyCreate(BaseModel):
    """Request body for creating a document family."""
    name: str = Field(..., min_length=1, max_length=100)


class FamilyRename(BaseModel):
    """Request body for renaming a document family."""
    name: str = Field(..., min_length=1, max_length=100)


class FamilySetActive(BaseModel):
    """Request body for toggling family active status."""
    is_active: bool


class FamilyResponse(BaseModel):
    """Response body for a document family."""
    id: str
    name: str
    is_active: bool
    document_count: int = 0
    created_at: str
    updated_at: str


class MoveDocumentRequest(BaseModel):
    """Request body for moving a document to a family."""
    family_id: Optional[str] = None  # None to unassign
