"""
FastAPI Backend for Legal RAG System

Provides REST API endpoints for document upload, RAG queries, and tenant config.
Decouples the frontend from backend so Docling can run on adequate infrastructure.

Run with: uvicorn execution.legal_rag.api:app --host 0.0.0.0 --port 8000
"""

import os
import json
import time
import uuid
import shutil
import logging
import tempfile
from typing import Optional
from pathlib import Path
from collections import defaultdict

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, Header, Request
from fastapi.responses import StreamingResponse
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

# Persistent storage directory for uploaded document files
DOCUMENT_STORAGE_DIR = Path(os.getenv("DOCUMENT_STORAGE_DIR", "document_files"))
DOCUMENT_STORAGE_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# CyLaw URL helpers & file lookup
# =============================================================================

def _find_document_file(document_id: str) -> Path | None:
    """Find a stored document file regardless of extension."""
    candidates = list(DOCUMENT_STORAGE_DIR.glob(f"{document_id}.*"))
    if candidates:
        return candidates[0]
    # Legacy: try bare .pdf
    legacy = DOCUMENT_STORAGE_DIR / f"{document_id}.pdf"
    if legacy.exists():
        return legacy
    return None


def generate_cylaw_url(stem: str) -> str | None:
    """Generate a CyLaw index-page URL from a document filename stem.

    Returns the index URL (the browsable page), or None if the stem
    doesn't match the expected pattern.
    """
    if not stem:
        return None
    # Handle _EKS suffix — strip it for URL generation
    clean = stem.replace("_EKS", "")
    parts = clean.split("_")
    # Standard pattern: YYYY_V_NNN  (e.g. 1960_1_002)
    if len(parts) >= 3 and parts[0].isdigit() and parts[-1].isdigit():
        # Index URLs strip leading zeros from the number segment
        parts_norm = list(parts)
        parts_norm[-1] = str(int(parts_norm[-1]))
        normalized = "_".join(parts_norm)
        return f"https://www.cylaw.org/nomoi/indexes/{normalized}.html"
    # CAP-style (e.g. CAP351) — no index page available
    return None


def _stem_from_file_path(file_path: str | None) -> str | None:
    """Extract filename stem from a stored file_path value."""
    if not file_path:
        return None
    return Path(file_path).stem

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
                timeout=120.0,
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
# Cache Invalidation
# =============================================================================

def _invalidate_client_cache(client_id: str, lang_config: TenantLanguageConfig = None):
    """Invalidate all cached answers for a client when documents change.

    Called on document upload and delete to prevent stale cached answers.
    """
    try:
        # Try all language services that have been initialized
        for lang, svc in _container._services.items():
            retriever = svc.get("retriever")
            if retriever and hasattr(retriever, "_result_cache"):
                retriever._result_cache.invalidate_for_client(client_id)
    except Exception as e:
        logger.warning(f"Cache invalidation failed for client {client_id}: {e}")


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

        # Persist original PDF for later viewing/download
        stored_path = DOCUMENT_STORAGE_DIR / f"{parsed.metadata.document_id}.pdf"
        shutil.copy2(tmp_path, stored_path)

        # Store
        store.insert_document(
            document_id=parsed.metadata.document_id,
            title=parsed.metadata.title,
            document_type=parsed.metadata.document_type,
            client_id=client_id,
            jurisdiction=parsed.metadata.jurisdiction,
            file_path=str(stored_path),
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

        # Invalidate answer cache for this client (new document may change answers)
        _invalidate_client_cache(client_id, lang_config)

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
    result = []
    for d in docs:
        # Derive CyLaw URL from metadata stem or file_path
        meta = d.get("metadata") or {}
        stem = meta.get("source_stem") or _stem_from_file_path(d.get("file_path"))
        result.append(DocumentInfo(
            id=str(d["id"]),
            title=d["title"],
            document_type=d["document_type"],
            jurisdiction=d.get("jurisdiction"),
            page_count=d.get("page_count", 0),
            created_at=str(d.get("created_at", "")),
            cylaw_url=generate_cylaw_url(stem),
        ))
    return result


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

        # Remove stored file (any extension)
        stored_file = _find_document_file(document_id)
        if stored_file:
            stored_file.unlink()

        # Invalidate answer cache (deleted document may have been in cached answers)
        _invalidate_client_cache(client_id)

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

    # Quota enforcement
    try:
        from .quotas import get_quota_manager, QuotaExceededError
        quota_manager = get_quota_manager(store)
        quota_manager.check_query_quota(client_id, tier=client.get("tier", "default"))
    except QuotaExceededError as e:
        raise HTTPException(status_code=429, detail=str(e))
    except Exception as e:
        logger.warning(f"Quota check failed (allowing request): {e}")

    # Audit
    store.log_audit(
        client_id=client_id,
        action="query",
        details={"query": request.query[:200]},
    )

    # Check full-pipeline cache first (answer + sources)
    retriever = services["retriever"]
    original_embedding = retriever.embeddings.embed_query(request.query)
    cache_hit = retriever._result_cache.get(
        request.query, client_id, request.document_id,
        query_embedding=original_embedding,
    )
    if cache_hit is not None:
        cached_results, answer_data = cache_hit
        if answer_data is not None:
            elapsed = (time.time() - start_time) * 1000
            logger.info(f"Full-pipeline cache hit: returning cached answer in {elapsed:.0f}ms")
            return QueryResponse(
                answer=answer_data["answer"],
                sources=[SourceInfo(**s) for s in answer_data["sources"]],
                latency_ms=elapsed,
            )

    # Retrieve (cache miss or no answer cached)
    # Pass pre-computed embedding to avoid redundant Voyage API call inside retriever
    results = retriever.retrieve(
        query=request.query,
        client_id=client_id,
        document_id=request.document_id,
        top_k=request.top_k,
        query_embedding=original_embedding,
    )

    # Filter out summary chunks
    results = [r for r in results if r.hierarchy_path != "Document"]

    if not results:
        return QueryResponse(
            answer="No relevant information found in the uploaded documents.",
            sources=[],
            latency_ms=(time.time() - start_time) * 1000,
        )

    # Look up document titles + metadata for citation accuracy & CyLaw URLs
    doc_ids = list(set(r.document_id for r in results))
    doc_source_meta = store.get_document_source_meta(doc_ids, client_id=client_id)
    doc_titles = {did: m["title"] for did, m in doc_source_meta.items()}

    # Pre-compute CyLaw URLs per document
    doc_cylaw_urls: dict[str, str | None] = {}
    for did, meta in doc_source_meta.items():
        stem = (meta.get("metadata") or {}).get("source_stem") or _stem_from_file_path(meta.get("file_path"))
        doc_cylaw_urls[did] = generate_cylaw_url(stem)

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
            max_tokens=3500,
            temperature=0.2,
        )
        answer = response.choices[0].message.content

    except Exception as e:
        from openai import APITimeoutError
        if isinstance(e, APITimeoutError):
            logger.error(f"LLM generation timed out for query ({lang}): {request.query[:100]}")
            answer = "I found relevant information but the answer generation timed out. See sources below."
        else:
            logger.error(f"LLM generation failed ({lang}): {type(e).__name__}: {e}")
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
            cylaw_url=doc_cylaw_urls.get(cc.citation.document_id),
        ))

    # Cache the full pipeline result (answer + sources) for future identical queries
    answer_data = {
        "answer": answer,
        "sources": [s.model_dump() for s in sources],
    }
    retriever._result_cache.set(
        request.query, results, client_id, request.document_id,
        query_embedding=original_embedding,
        answer_data=answer_data,
    )

    return QueryResponse(
        answer=answer,
        sources=sources,
        latency_ms=(time.time() - start_time) * 1000,
    )


@app.post("/api/v1/query/stream", dependencies=[Depends(check_rate_limit)])
async def query_documents_stream(
    request: QueryRequest,
    client: dict = Depends(get_authenticated_client),
    lang_config: TenantLanguageConfig = Depends(get_tenant_config),
):
    """Streaming RAG query with SSE.

    Sends events:
      - {"event": "sources", "data": [...]}  (after retrieval)
      - {"event": "token", "data": "..."}    (during generation)
      - {"event": "done", "data": {"latency_ms": ...}}  (final)
      - {"event": "error", "data": "..."}    (on failure)
    """
    start_time = time.time()
    client_id = client["client_id"]
    services = _container.get_services(lang_config)
    store = _container.get_store()

    # Quota enforcement
    try:
        from .quotas import get_quota_manager, QuotaExceededError
        quota_manager = get_quota_manager(store)
        quota_manager.check_query_quota(client_id, tier=client.get("tier", "default"))
    except QuotaExceededError as e:
        raise HTTPException(status_code=429, detail=str(e))
    except Exception as e:
        logger.warning(f"Quota check failed (allowing request): {e}")

    # Audit
    store.log_audit(client_id=client_id, action="query", details={"query": request.query[:200]})

    def _sse_event(event: str, data) -> str:
        """Format a Server-Sent Event."""
        return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    def generate():
        # Check full-pipeline cache first
        retriever = services["retriever"]
        original_embedding = retriever.embeddings.embed_query(request.query)
        cache_hit = retriever._result_cache.get(
            request.query, client_id, request.document_id,
            query_embedding=original_embedding,
        )
        if cache_hit is not None:
            cached_results, answer_data = cache_hit
            if answer_data is not None:
                elapsed = (time.time() - start_time) * 1000
                logger.info(f"Stream: full-pipeline cache hit in {elapsed:.0f}ms")
                yield _sse_event("sources", answer_data["sources"])
                yield _sse_event("token", answer_data["answer"])
                yield _sse_event("done", {"latency_ms": elapsed})
                return

        # Retrieve — pass pre-computed embedding to avoid redundant Voyage API call
        results = retriever.retrieve(
            query=request.query,
            client_id=client_id,
            document_id=request.document_id,
            top_k=request.top_k,
            query_embedding=original_embedding,
        )
        results = [r for r in results if r.hierarchy_path != "Document"]

        if not results:
            yield _sse_event("token", "No relevant information found in the uploaded documents.")
            yield _sse_event("sources", [])
            yield _sse_event("done", {"latency_ms": (time.time() - start_time) * 1000})
            return

        # Build citations + sources
        doc_ids = list(set(r.document_id for r in results))
        doc_source_meta = store.get_document_source_meta(doc_ids, client_id=client_id)
        doc_titles = {did: m["title"] for did, m in doc_source_meta.items()}
        doc_cylaw_urls: dict[str, str | None] = {}
        for did, meta in doc_source_meta.items():
            stem = (meta.get("metadata") or {}).get("source_stem") or _stem_from_file_path(meta.get("file_path"))
            doc_cylaw_urls[did] = generate_cylaw_url(stem)
        cited_contents = services["citation_extractor"].extract(results, document_titles=doc_titles)

        sources_list = []
        for cc in cited_contents:
            sources_list.append(SourceInfo(
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
                cylaw_url=doc_cylaw_urls.get(cc.citation.document_id),
            ))

        # Send sources immediately so frontend can show them while answer streams
        yield _sse_event("sources", [s.model_dump() for s in sources_list])

        # Build context for LLM
        context = "\n\n---\n\n".join([
            f"**[{i+1}]** {cc.citation.short_format()}:\n{cc.content}"
            for i, cc in enumerate(cited_contents)
        ])

        # Stream answer generation
        lang = lang_config.language
        full_answer = ""
        try:
            llm_client = _container.get_llm_client()
            system_prompt = LLM_PROMPTS.get(lang, LLM_PROMPTS["en"])["rag_system"]

            stream = llm_client.chat.completions.create(
                model=lang_config.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Based on the following sources, answer this question: {request.query}\n\nSOURCES:\n{context}\n\nProvide a clear, well-cited answer."},
                ],
                max_tokens=3500,
                temperature=0.2,
                stream=True,
            )

            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    full_answer += token
                    yield _sse_event("token", token)

        except Exception as e:
            from openai import APITimeoutError
            if isinstance(e, APITimeoutError):
                logger.error(f"Stream: LLM timed out ({lang}): {request.query[:100]}")
                fallback = "I found relevant information but the answer generation timed out. See sources below."
            else:
                logger.error(f"Stream: LLM failed ({lang}): {type(e).__name__}: {e}")
                fallback = "I found relevant information but couldn't generate a summary. See sources below."
            full_answer = fallback
            yield _sse_event("token", fallback)

        # Cache the full pipeline result
        answer_data = {
            "answer": full_answer,
            "sources": [s.model_dump() for s in sources_list],
        }
        retriever._result_cache.set(
            request.query, results, client_id, request.document_id,
            query_embedding=original_embedding,
            answer_data=answer_data,
        )

        yield _sse_event("done", {"latency_ms": (time.time() - start_time) * 1000})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
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
