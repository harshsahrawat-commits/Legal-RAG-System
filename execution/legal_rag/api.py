"""
FastAPI Backend for Legal RAG System

Provides REST API endpoints for document upload, RAG queries, and tenant config.
Decouples the frontend from backend so Docling can run on adequate infrastructure.

Run with: uvicorn execution.legal_rag.api:app --host 0.0.0.0 --port 8000
"""

import os
import time
import uuid
import logging
import tempfile
from typing import Optional
from pathlib import Path
from collections import defaultdict

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from .api_models import (
    QueryRequest, QueryResponse, SourceInfo,
    DocumentInfo, UploadResponse,
    TenantConfigUpdate, TenantConfigResponse,
    HealthResponse,
)
from .language_config import TenantLanguageConfig
from .language_patterns import LLM_PROMPTS

# Load environment variables
load_dotenv()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Legal RAG API",
    description="REST API for Legal Document Intelligence with multilingual support",
    version="0.2.0",
)

# Configure CORS: use CORS_ORIGINS env var (comma-separated) or default to localhost
_cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _cors_origins],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Rate Limiting
# =============================================================================

class RateLimiter:
    """Simple in-memory rate limiter using sliding window."""

    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self._max_requests = max_requests
        self._window = window_seconds
        self._requests: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, key: str) -> bool:
        """Check if request is allowed for the given key."""
        now = time.time()
        window_start = now - self._window

        # Clean old entries
        self._requests[key] = [t for t in self._requests[key] if t > window_start]

        if len(self._requests[key]) >= self._max_requests:
            return False

        self._requests[key].append(now)
        return True


_rate_limiter = RateLimiter(
    max_requests=int(os.getenv("RATE_LIMIT_RPM", "60")),
    window_seconds=60,
)


async def check_rate_limit(request: Request):
    """FastAPI dependency that enforces rate limiting per API key."""
    api_key = request.headers.get("x-api-key", request.client.host)
    if not _rate_limiter.is_allowed(api_key):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again later.")


# =============================================================================
# Service Container - caches services per language config
# =============================================================================

class ServiceContainer:
    """Singleton that caches embedding services and retrievers per language config."""

    def __init__(self):
        self._store = None
        self._services = {}  # keyed by language code
        self._llm_client = None  # Cached OpenAI client for NVIDIA NIM

    def get_store(self):
        if self._store is None:
            from .vector_store import VectorStore
            self._store = VectorStore()
            self._store.connect()
            self._store.initialize_schema()
            try:
                self._store.initialize_auth_schema()
                self._store.initialize_tenant_config_schema()
            except Exception as e:
                logger.warning(f"Schema init partial: {e}")
        return self._store

    def get_services(self, lang_config: TenantLanguageConfig):
        """Get or create cached services for a language configuration."""
        lang = lang_config.language
        if lang not in self._services:
            from .document_parser import LegalDocumentParser
            from .chunker import LegalChunker
            from .embeddings import get_embedding_service
            from .retriever import HybridRetriever
            from .citation import CitationExtractor

            store = self.get_store()
            embeddings = get_embedding_service(
                provider=lang_config.embedding_provider,
                language_config=lang_config,
            )
            retriever = HybridRetriever(
                store, embeddings, language_config=lang_config,
            )
            parser = LegalDocumentParser(language_config=lang_config)
            chunker = LegalChunker(language_config=lang_config)
            citation_ext = CitationExtractor(language_config=lang_config)

            self._services[lang] = {
                "parser": parser,
                "chunker": chunker,
                "embeddings": embeddings,
                "retriever": retriever,
                "citation_extractor": citation_ext,
            }
        return self._services[lang]

    def get_llm_client(self):
        """Get or create cached OpenAI client for NVIDIA NIM API."""
        if self._llm_client is None:
            from openai import OpenAI
            self._llm_client = OpenAI(
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=os.getenv("NVIDIA_API_KEY"),
                timeout=30.0,
            )
        return self._llm_client


_container = ServiceContainer()


# =============================================================================
# Authentication dependency
# =============================================================================

async def get_authenticated_client(x_api_key: str = Header(...)) -> dict:
    """Validate API key and return client info."""
    store = _container.get_store()
    result = store.validate_api_key(x_api_key)
    if not result:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return result


async def get_tenant_config(
    client: dict = Depends(get_authenticated_client),
) -> TenantLanguageConfig:
    """Load tenant language config from DB."""
    store = _container.get_store()
    config = store.get_tenant_config(client["client_id"])
    if config is None:
        config = TenantLanguageConfig.for_language("en")
    return config


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/api/v1/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    db_status = "unknown"
    try:
        store = _container.get_store()
        db_status = "connected"
    except Exception as e:
        logger.warning(f"Health check: database disconnected: {e}")
        db_status = "disconnected"

    return HealthResponse(
        status="ok",
        version="0.2.0",
        database=db_status,
    )


@app.post("/api/v1/documents/upload", response_model=UploadResponse, dependencies=[Depends(check_rate_limit)])
async def upload_document(
    file: UploadFile = File(...),
    client: dict = Depends(get_authenticated_client),
    lang_config: TenantLanguageConfig = Depends(get_tenant_config),
):
    """Upload and process a PDF document."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    client_id = client["client_id"]
    services = _container.get_services(lang_config)
    store = _container.get_store()

    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Parse
        parsed = services["parser"].parse(tmp_path, client_id=client_id)

        # Chunk
        chunks = services["chunker"].chunk(parsed)

        # Contextualize
        doc_summary = parsed.raw_text[:2000] if parsed.raw_text else ""
        if doc_summary:
            chunks = services["chunker"].contextualize_chunks(
                chunks=chunks, document_summary=doc_summary,
            )

        # Embed
        chunk_texts = [c.content for c in chunks]
        embeddings = services["embeddings"].embed_documents(chunk_texts)

        # Store
        store.insert_document(
            document_id=parsed.metadata.document_id,
            title=parsed.metadata.title,
            document_type=parsed.metadata.document_type,
            client_id=client_id,
            jurisdiction=parsed.metadata.jurisdiction,
            file_path=tmp_path,
            page_count=parsed.metadata.page_count,
        )
        chunk_dicts = [c.to_dict() for c in chunks]
        store.insert_chunks(chunk_dicts, embeddings, client_id=client_id)

        # Audit
        store.log_audit(
            client_id=client_id,
            action="ingest",
            resource_type="document",
            resource_id=parsed.metadata.document_id,
            details={"title": parsed.metadata.title, "chunks": len(chunks)},
        )

        return UploadResponse(
            id=parsed.metadata.document_id,
            title=parsed.metadata.title,
            document_type=parsed.metadata.document_type,
            jurisdiction=parsed.metadata.jurisdiction,
            page_count=parsed.metadata.page_count,
            chunks=len(chunks),
        )
    finally:
        os.unlink(tmp_path)


@app.get("/api/v1/documents", response_model=list[DocumentInfo])
async def list_documents(
    client: dict = Depends(get_authenticated_client),
):
    """List all documents for the authenticated tenant."""
    store = _container.get_store()
    docs = store.list_documents(client_id=client["client_id"])
    return [
        DocumentInfo(
            id=str(d["id"]),
            title=d["title"],
            document_type=d["document_type"],
            jurisdiction=d.get("jurisdiction"),
            page_count=d.get("page_count", 0),
            created_at=str(d.get("created_at", "")),
        )
        for d in docs
    ]


@app.delete("/api/v1/documents/{document_id}")
async def delete_document(
    document_id: str,
    client: dict = Depends(get_authenticated_client),
):
    """Delete a document and all its chunks (tenant-isolated)."""
    store = _container.get_store()
    client_id = client["client_id"]
    try:
        deleted = store.delete_document(document_id, client_id=client_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Document not found")
        store.log_audit(
            client_id=client_id,
            action="delete",
            resource_type="document",
            resource_id=document_id,
        )
        return {"status": "deleted", "document_id": document_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/query", response_model=QueryResponse, dependencies=[Depends(check_rate_limit)])
async def query_documents(
    request: QueryRequest,
    client: dict = Depends(get_authenticated_client),
    lang_config: TenantLanguageConfig = Depends(get_tenant_config),
):
    """RAG query with citations."""
    start_time = time.time()
    client_id = client["client_id"]
    services = _container.get_services(lang_config)
    store = _container.get_store()

    # Audit
    store.log_audit(
        client_id=client_id,
        action="query",
        details={"query": request.query[:200]},
    )

    # Retrieve
    results = services["retriever"].retrieve(
        query=request.query,
        client_id=client_id,
        document_id=request.document_id,
        top_k=request.top_k,
    )

    # Filter out summary chunks
    results = [r for r in results if r.hierarchy_path != "Document"]

    if not results:
        return QueryResponse(
            answer="No relevant information found in the uploaded documents.",
            sources=[],
            latency_ms=(time.time() - start_time) * 1000,
        )

    # Look up real document titles for citation accuracy
    doc_ids = list(set(r.document_id for r in results))
    doc_titles = store.get_document_titles(doc_ids, client_id=client_id)

    # Citations
    cited_contents = services["citation_extractor"].extract(
        results, document_titles=doc_titles
    )

    # Build context
    context = "\n\n---\n\n".join([
        f"**[{i+1}]** {cc.citation.short_format()}:\n{cc.content}"
        for i, cc in enumerate(cited_contents)
    ])

    # Generate answer
    lang = lang_config.language
    try:
        llm_client = _container.get_llm_client()

        system_prompt = LLM_PROMPTS.get(lang, LLM_PROMPTS["en"])["rag_system"]

        response = llm_client.chat.completions.create(
            model=lang_config.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Based on the following sources, answer this question: {request.query}\n\nSOURCES:\n{context}\n\nProvide a clear, well-cited answer."},
            ],
            max_tokens=1500,
            temperature=0.2,
        )
        answer = response.choices[0].message.content

    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        answer = "I found relevant information but couldn't generate a summary. See sources below."

    # Format sources
    sources = []
    for cc in cited_contents:
        sources.append(SourceInfo(
            document_title=cc.citation.document_title,
            section=cc.citation.section,
            page_numbers=cc.citation.page_numbers,
            hierarchy_path=cc.citation.hierarchy_path,
            chunk_id=cc.citation.chunk_id,
            document_id=cc.citation.document_id,
            relevance_score=cc.citation.relevance_score,
            short_citation=cc.citation.short_format(),
            long_citation=cc.citation.long_format(),
            content=cc.content,
            context_before=cc.context_before or "",
            context_after=cc.context_after or "",
        ))

    return QueryResponse(
        answer=answer,
        sources=sources,
        latency_ms=(time.time() - start_time) * 1000,
    )


@app.get("/api/v1/config", response_model=TenantConfigResponse)
async def get_config(
    lang_config: TenantLanguageConfig = Depends(get_tenant_config),
):
    """Get tenant language/model configuration."""
    return TenantConfigResponse(
        language=lang_config.language,
        embedding_model=lang_config.embedding_model,
        embedding_provider=lang_config.embedding_provider,
        llm_model=lang_config.llm_model,
        reranker_model=lang_config.reranker_model,
        fts_language=lang_config.fts_language,
    )


@app.put("/api/v1/config", response_model=TenantConfigResponse)
async def update_config(
    update: TenantConfigUpdate,
    client: dict = Depends(get_authenticated_client),
):
    """Update tenant language/model configuration."""
    store = _container.get_store()
    client_id = client["client_id"]

    # Get current config or create from requested language
    current = store.get_tenant_config(client_id)
    if current is None:
        lang = update.language or "en"
        current = TenantLanguageConfig.for_language(lang)

    # Apply updates
    if update.language is not None:
        new_config = TenantLanguageConfig.for_language(update.language)
        current = new_config
    if update.embedding_model is not None:
        current.embedding_model = update.embedding_model
    if update.embedding_provider is not None:
        current.embedding_provider = update.embedding_provider
    if update.llm_model is not None:
        current.llm_model = update.llm_model
    if update.reranker_model is not None:
        current.reranker_model = update.reranker_model

    store.set_tenant_config(client_id, current)

    # Create Greek FTS index if switching to Greek
    if current.language == "el":
        try:
            store.create_greek_fts_index()
        except Exception as e:
            logger.warning(f"Greek FTS index creation failed: {e}")

    store.log_audit(
        client_id=client_id,
        action="config_update",
        details={"language": current.language},
    )

    return TenantConfigResponse(
        language=current.language,
        embedding_model=current.embedding_model,
        embedding_provider=current.embedding_provider,
        llm_model=current.llm_model,
        reranker_model=current.reranker_model,
        fts_language=current.fts_language,
    )
