"""
FastAPI Backend for Legal RAG System

Provides REST API endpoints for document upload, RAG queries, and tenant config.
Decouples the frontend from backend so Docling can run on adequate infrastructure.

Run with: uvicorn execution.legal_rag.api:app --host 0.0.0.0 --port 8000
"""

import os
import re
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
    GoogleAuthRequest, AuthResponse, UserInfo,
    ConversationCreate, ConversationResponse, ConversationRename,
    MessageResponse, StreamQueryRequest,
    FamilyCreate, FamilyRename, FamilySetActive, FamilyResponse,
    MoveDocumentRequest,
)
from .language_config import TenantLanguageConfig
from .language_patterns import LLM_PROMPTS
from .auth import verify_google_token, create_session_jwt, verify_session_jwt

# Load environment variables
load_dotenv()
logger = logging.getLogger(__name__)

# Greek Unicode range regex for per-query language detection
_GREEK_RE = re.compile(r'[\u0370-\u03FF\u1F00-\u1FFF]')

def detect_query_language(query: str) -> str:
    """Detect whether a query is Greek or English based on character content.

    Returns "el" if >=30% of alphabetic characters are Greek, otherwise "en".
    This allows mixed-language queries to be handled appropriately.
    """
    alpha_chars = [c for c in query if c.isalpha()]
    if not alpha_chars:
        return "en"
    greek_count = sum(1 for c in alpha_chars if _GREEK_RE.match(c))
    return "el" if greek_count / len(alpha_chars) >= 0.3 else "en"

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
    """Generate a CyLaw PDF URL from a document filename stem.

    Points to /nomoi/arith/ which hosts the original law PDFs and has
    coverage for all laws (unlike /nomoi/indexes/ which only has
    consolidated texts for a subset).
    """
    if not stem:
        return None
    # Handle _EKS suffix — strip it for URL generation
    clean = stem.replace("_EKS", "")
    parts = clean.split("_")
    # Standard pattern: YYYY_V_NNN  (e.g. 1960_1_002, 2007_2_029, 2004_3_018)
    if len(parts) >= 3 and parts[0].isdigit() and parts[-1].isdigit():
        return f"https://www.cylaw.org/nomoi/arith/{clean}.pdf"
    # CAP-style (e.g. CAP154, CAP029A)
    if clean.upper().startswith("CAP"):
        return f"https://www.cylaw.org/nomoi/arith/{clean.upper()}.pdf"
    # Special documents (Constitution, etc.)
    if clean.lower() == "syntagma":
        return "https://www.cylaw.org/nomoi/arith/syntagma.pdf"
    return None


def _stem_from_file_path(file_path: str | None) -> str | None:
    """Extract filename stem from a stored file_path value."""
    if not file_path:
        return None
    return Path(file_path).stem


def _stem_from_title(title: str | None) -> str | None:
    """Derive a CyLaw filename stem from a document title.

    Cypriot law titles contain the law identifier, e.g.:
    'Ο περί ... Νόμος του 2020 (123(I)/2020)'   → '2020_1_123'
    'Ο περί ... Νόμος του 2011 (47(Ι)/2011)'    → '2011_1_047'
    'Οι περί ... Κανονισμοί του 2007 (29(II)/2007)' → '2007_2_029'
    'Ο περί ... Διάταγμα (18(III)/2004)'         → '2004_3_018'
    Handles both Latin (I/II/III) and Greek (Ι/ΙΙ/ΙΙΙ) iota.
    """
    if not title:
        return None
    import re
    # Volume mapping: (I)→1, (II)→2, (III)→3 — Latin or Greek iota
    for roman, volume in [("III", "3"), ("II", "2"), ("I", "1")]:
        # Match both Latin I and Greek Ι
        greek_roman = roman.replace("I", "Ι")
        pattern = rf"(\d+)\s*\((?:{roman}|{greek_roman})\)\s*/?\s*(\d{{4}})"
        m = re.search(pattern, title)
        if m:
            number = m.group(1)
            year = m.group(2)
            return f"{year}_{volume}_{number.zfill(3)}"
    # CAP-style: "Κεφ. 154" or "Cap. 154" or "CAP 154"
    m = re.search(r"(?:Κεφ\.|Cap\.|CAP)\s*(\d+[A-Z]?)", title, re.IGNORECASE)
    if m:
        return f"CAP{m.group(1)}"
    # Constitution: "ΣΥΝΤΑΓΜΑ" or "Σύνταγμα"
    if "ΣΥΝΤΑΓΜΑ" in title.upper() or "SYNTAGMA" in title.upper():
        return "syntagma"
    return None

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
    """FastAPI dependency that enforces rate limiting per user/IP."""
    auth_header = request.headers.get("authorization", "")
    identifier = auth_header[7:20] if auth_header.startswith("Bearer ") else request.client.host
    if not _rate_limiter.is_allowed(identifier):
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
                self._store.migrate_add_user_auth_schema()
                self._store.migrate_add_document_families()
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
# Authentication dependencies
# =============================================================================

async def get_authenticated_user(request: Request) -> dict:
    """Authenticate via JWT Bearer token (Google OAuth).

    Returns dict with:
      - user_id: str (UUID of the users table row)
      - client_id: str (alias for user_id, backward compat)
      - email, name: from JWT payload
      - auth_method: "jwt"
    """
    auth_header = request.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        user_info = verify_session_jwt(token)
        if user_info:
            return {
                "user_id": user_info["user_id"],
                "client_id": user_info["user_id"],  # alias for backward compat
                "email": user_info["email"],
                "name": user_info["name"],
                "tier": "default",
                "auth_method": "jwt",
            }
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    raise HTTPException(status_code=401, detail="Missing authentication. Please sign in with Google.")


# Backward-compatible alias
async def get_authenticated_client(request: Request) -> dict:
    """Alias for get_authenticated_user for backward compatibility."""
    return await get_authenticated_user(request)


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
# Multi-Source Retrieval Helpers
# =============================================================================

def _source_toggle_key(sources) -> str:
    """Build a string key from source toggles for cache differentiation."""
    families = ",".join(sorted(sources.families)) if sources.families else ""
    return f"cylaw={int(sources.cylaw)},hudoc={int(sources.hudoc)},eurlex={int(sources.eurlex)},fam={families}"


# Origin label mapping for LLM context
_ORIGIN_LABELS = {
    "cylaw": "CyLaw",
    "hudoc": "ECHR",
    "eurlex": "EU Law",
}


def _retrieve_from_db(query, retriever, client_id, document_id, top_k,
                      query_embedding, services, store, source_origins,
                      family_ids=None, conversation_id=None, query_lang=None):
    """Unified DB retrieval for all sources (CyLaw, HUDOC, EUR-Lex, families, session docs).

    All sources are stored in the same pgvector DB with a `source_origin` column.
    This replaces the old 3-way parallel retrieval (DB + HUDOC API + EUR-Lex SPARQL).

    Returns (cited_contents, all_sources) where cited_contents is the list used
    for building LLM context, and all_sources is the list of SourceInfo objects.
    """
    try:
        results = retriever.retrieve(
            query=query,
            client_id=client_id,
            document_id=document_id,
            top_k=top_k,
            query_embedding=query_embedding,
            source_origins=source_origins,
            family_ids=family_ids,
            conversation_id=conversation_id,
            query_lang=query_lang,
        )
        results = [r for r in results if r.hierarchy_path != "Document"]
        if not results:
            return [], []

        # Fetch document metadata for all unique doc IDs
        doc_ids = list(set(r.document_id for r in results))
        doc_source_meta = store.get_document_source_meta(doc_ids)
        doc_titles = {did: m["title"] for did, m in doc_source_meta.items()}

        # Build citations
        cited_contents = services["citation_extractor"].extract(results, document_titles=doc_titles)

        # Build SourceInfo per result, determining origin from document metadata
        sources_list = []
        for cc in cited_contents:
            did = cc.citation.document_id
            meta = doc_source_meta.get(did, {})
            doc_metadata = meta.get("metadata") or {}
            origin = doc_metadata.get("source_origin", "cylaw")

            # Determine URLs based on origin
            if origin == "cylaw":
                stem = (doc_metadata.get("source_stem")
                        or _stem_from_file_path(meta.get("file_path"))
                        or _stem_from_title(meta.get("title")))
                cylaw_url = generate_cylaw_url(stem)
                external_url = None
            elif origin == "hudoc":
                cylaw_url = None
                external_url = (
                    doc_metadata.get("source_url")
                    or (f"https://hudoc.echr.coe.int/eng?i={doc_metadata['item_id']}"
                        if doc_metadata.get("item_id") else None)
                )
            elif origin == "eurlex":
                cylaw_url = None
                external_url = (
                    doc_metadata.get("source_url")
                    or (f"https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:{doc_metadata['celex']}"
                        if doc_metadata.get("celex") else None)
                )
            else:
                # Unknown origin — try source_url as generic fallback
                cylaw_url = None
                external_url = doc_metadata.get("source_url")

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
                cylaw_url=cylaw_url,
                source_origin=origin,
                external_url=external_url,
            ))

        return cited_contents, sources_list
    except Exception as e:
        logger.warning(f"DB retrieval failed: {e}")
        return [], []


def _build_context_string(cited_contents, sources_list):
    """Build LLM context string with origin labels from cited contents + sources.

    Returns the context string for the LLM prompt.
    """
    context_parts = []
    for idx, (cc, src) in enumerate(zip(cited_contents, sources_list), 1):
        label = _ORIGIN_LABELS.get(src.source_origin, src.source_origin)
        context_parts.append(f"**[{idx}]** ({label}) {cc.citation.short_format()}:\n{cc.content}")

    return "\n\n---\n\n".join(context_parts)


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
    family_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
    upload_scope: str = "persistent",
    client: dict = Depends(get_authenticated_client),
    lang_config: TenantLanguageConfig = Depends(get_tenant_config),
):
    """Upload and process a PDF document.

    Query params:
      - family_id: Optional family UUID to assign the document to
      - conversation_id: Optional conversation UUID for chat-scoped uploads
      - upload_scope: 'persistent' (default) or 'session' (chat-scoped)
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    client_id = client["client_id"]
    user_id = client.get("user_id", client_id)
    services = _container.get_services(lang_config)
    store = _container.get_store()

    # Determine source_origin based on context
    if upload_scope == "session":
        source_origin = "session"
    elif family_id:
        source_origin = "user"
    else:
        source_origin = "user"

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
            family_id=family_id,
            upload_scope=upload_scope,
            conversation_id=conversation_id if upload_scope == "session" else None,
            user_id=user_id,
        )
        chunk_dicts = [c.to_dict() for c in chunks]
        store.insert_chunks(
            chunk_dicts, embeddings, client_id=client_id,
            source_origin=source_origin, family_id=family_id,
        )

        # Audit
        store.log_audit(
            client_id=client_id,
            action="ingest",
            resource_type="document",
            resource_id=parsed.metadata.document_id,
            details={"title": parsed.metadata.title, "chunks": len(chunks),
                      "family_id": family_id, "upload_scope": upload_scope},
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
    """List all persistent documents for the authenticated tenant (excludes session docs)."""
    store = _container.get_store()
    docs = store.list_documents(client_id=client["client_id"], exclude_session=True)
    result = []
    for d in docs:
        # Derive CyLaw URL from metadata stem, file_path, or title
        meta = d.get("metadata") or {}
        stem = meta.get("source_stem") or _stem_from_file_path(d.get("file_path")) or _stem_from_title(d.get("title"))
        result.append(DocumentInfo(
            id=str(d["id"]),
            title=d["title"],
            document_type=d["document_type"],
            jurisdiction=d.get("jurisdiction"),
            page_count=d.get("page_count", 0),
            created_at=str(d.get("created_at", "")),
            cylaw_url=generate_cylaw_url(stem),
            family_id=str(d["family_id"]) if d.get("family_id") else None,
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
    """RAG query with citations. Supports multi-source toggles (CyLaw, HUDOC, EUR-Lex)."""
    start_time = time.time()
    client_id = client["client_id"]
    services = _container.get_services(lang_config)
    store = _container.get_store()
    sources_cfg = request.sources

    # Check that at least one source is enabled
    if not sources_cfg.cylaw and not sources_cfg.hudoc and not sources_cfg.eurlex and not sources_cfg.families:
        return QueryResponse(
            answer="No sources are enabled. Please enable at least one source to search.",
            sources=[],
            latency_ms=(time.time() - start_time) * 1000,
        )

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
        details={"query": request.query[:200], "sources": _source_toggle_key(sources_cfg)},
    )

    # Check full-pipeline cache (key includes toggle state)
    retriever = services["retriever"]
    original_embedding = retriever.embeddings.embed_query(request.query)
    toggle_key = _source_toggle_key(sources_cfg)
    cache_doc_id = f"{request.document_id or ''}|{toggle_key}"
    cache_hit = retriever._result_cache.get(
        request.query, client_id, cache_doc_id,
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

    # Unified DB retrieval with source_origins filter
    source_origins = []
    if sources_cfg.cylaw:
        source_origins.append("cylaw")
    if sources_cfg.hudoc:
        source_origins.append("hudoc")
    if sources_cfg.eurlex:
        source_origins.append("eurlex")

    # Include user-uploaded docs when family toggles are active
    family_ids = sources_cfg.families if sources_cfg.families else None

    # Per-query language detection
    query_lang = detect_query_language(request.query)

    cited_contents, all_sources = _retrieve_from_db(
        request.query, retriever, client_id, request.document_id,
        request.top_k, original_embedding, services, store, source_origins,
        family_ids=family_ids, query_lang=query_lang,
    )

    if not all_sources:
        return QueryResponse(
            answer="No relevant information found in the enabled sources.",
            sources=[],
            latency_ms=(time.time() - start_time) * 1000,
        )

    # Build LLM context
    context = _build_context_string(cited_contents, all_sources)
    try:
        llm_client = _container.get_llm_client()
        system_prompt = LLM_PROMPTS.get(query_lang, LLM_PROMPTS["en"])["rag_system"]

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
            logger.error(f"LLM generation timed out for query ({query_lang}): {request.query[:100]}")
            answer = "I found relevant information but the answer generation timed out. See sources below."
        else:
            logger.error(f"LLM generation failed ({query_lang}): {type(e).__name__}: {e}")
            answer = "I found relevant information but couldn't generate a summary. See sources below."

    # Cache the full pipeline result
    answer_data = {
        "answer": answer,
        "sources": [s.model_dump() for s in all_sources],
    }
    retriever._result_cache.set(
        request.query, [], client_id, cache_doc_id,
        query_embedding=original_embedding,
        answer_data=answer_data,
    )

    return QueryResponse(
        answer=answer,
        sources=all_sources,
        latency_ms=(time.time() - start_time) * 1000,
    )


@app.post("/api/v1/query/stream", dependencies=[Depends(check_rate_limit)])
async def query_documents_stream(
    request: StreamQueryRequest,
    client: dict = Depends(get_authenticated_client),
    lang_config: TenantLanguageConfig = Depends(get_tenant_config),
):
    """Streaming RAG query with SSE. Supports multi-source toggles and conversation persistence.

    Sends events:
      - {"event": "sources", "data": [...]}  (after retrieval)
      - {"event": "token", "data": "..."}    (during generation)
      - {"event": "done", "data": {"latency_ms": ..., "conversation_id": ...}}  (final)
      - {"event": "error", "data": "..."}    (on failure)
    """
    start_time = time.time()
    client_id = client["client_id"]
    user_id = client.get("user_id", client_id)
    services = _container.get_services(lang_config)
    store = _container.get_store()
    sources_cfg = request.sources
    conversation_id = request.conversation_id

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
    store.log_audit(client_id=client_id, action="query", details={"query": request.query[:200], "sources": _source_toggle_key(sources_cfg)})

    def _sse_event(event: str, data) -> str:
        """Format a Server-Sent Event."""
        return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    def _save_messages(conv_id, query_text, answer_text, sources_data, latency):
        """Save user + assistant messages to DB (best-effort)."""
        if not conv_id:
            return
        try:
            store.add_message(conv_id, "user", query_text)
            store.add_message(
                conv_id, "assistant", answer_text,
                sources=sources_data,
                latency_ms=latency,
            )
            store.touch_conversation(conv_id)
        except Exception as exc:
            logger.warning(f"Failed to save messages to conversation {conv_id}: {exc}")

    def _auto_title(conv_id, query_text):
        """Auto-generate conversation title from first user query (best-effort)."""
        if not conv_id:
            return
        try:
            msgs = store.get_messages(conv_id, user_id)
            # Only auto-title if this is the first exchange (0 existing messages before we saved)
            user_msgs = [m for m in msgs if m["role"] == "user"]
            if len(user_msgs) <= 1:
                title = query_text[:80].strip()
                if len(query_text) > 80:
                    title += "..."
                store.rename_conversation(conv_id, user_id, title)
        except Exception as exc:
            logger.warning(f"Auto-title failed for conversation {conv_id}: {exc}")

    def generate():
        nonlocal conversation_id

        # Auto-create conversation if JWT user sends without conversation_id
        if not conversation_id and client.get("auth_method") == "jwt":
            try:
                c = store.create_conversation(user_id, title="New Chat")
                conversation_id = c["id"]
            except Exception as exc:
                logger.warning(f"Auto-create conversation failed: {exc}")

        # Send conversation_id to frontend so it can track it
        if conversation_id:
            yield _sse_event("conversation_id", conversation_id)

        # Check no-sources-enabled edge case
        if not sources_cfg.cylaw and not sources_cfg.hudoc and not sources_cfg.eurlex and not sources_cfg.families:
            no_src_msg = "No sources are enabled. Please enable at least one source to search."
            yield _sse_event("token", no_src_msg)
            yield _sse_event("sources", [])
            elapsed = (time.time() - start_time) * 1000
            _save_messages(conversation_id, request.query, no_src_msg, [], elapsed)
            yield _sse_event("done", {"latency_ms": elapsed, "conversation_id": conversation_id})
            return

        # Check full-pipeline cache (key includes toggle state)
        retriever = services["retriever"]
        original_embedding = retriever.embeddings.embed_query(request.query)
        toggle_key = _source_toggle_key(sources_cfg)
        cache_doc_id = f"{request.document_id or ''}|{toggle_key}"
        cache_hit = retriever._result_cache.get(
            request.query, client_id, cache_doc_id,
            query_embedding=original_embedding,
        )
        if cache_hit is not None:
            cached_results, answer_data = cache_hit
            if answer_data is not None:
                elapsed = (time.time() - start_time) * 1000
                logger.info(f"Stream: full-pipeline cache hit in {elapsed:.0f}ms")
                yield _sse_event("sources", answer_data["sources"])
                yield _sse_event("token", answer_data["answer"])
                _save_messages(conversation_id, request.query, answer_data["answer"], answer_data["sources"], elapsed)
                _auto_title(conversation_id, request.query)
                yield _sse_event("done", {"latency_ms": elapsed, "conversation_id": conversation_id})
                return

        # Unified DB retrieval with source_origins filter
        source_origins = []
        if sources_cfg.cylaw:
            source_origins.append("cylaw")
        if sources_cfg.hudoc:
            source_origins.append("hudoc")
        if sources_cfg.eurlex:
            source_origins.append("eurlex")

        # Include user-uploaded docs when family toggles are active
        family_ids = sources_cfg.families if sources_cfg.families else None

        # Per-query language detection
        query_lang = detect_query_language(request.query)

        cited_contents, all_sources = _retrieve_from_db(
            request.query, retriever, client_id, request.document_id,
            request.top_k, original_embedding, services, store, source_origins,
            family_ids=family_ids, conversation_id=conversation_id, query_lang=query_lang,
        )

        if not all_sources:
            no_result_msg = "No relevant information found in the enabled sources."
            yield _sse_event("token", no_result_msg)
            yield _sse_event("sources", [])
            elapsed = (time.time() - start_time) * 1000
            _save_messages(conversation_id, request.query, no_result_msg, [], elapsed)
            yield _sse_event("done", {"latency_ms": elapsed, "conversation_id": conversation_id})
            return

        # Send sources immediately so frontend can show them while answer streams
        sources_serialized = [s.model_dump() for s in all_sources]
        yield _sse_event("sources", sources_serialized)

        # Build LLM context
        context = _build_context_string(cited_contents, all_sources)
        full_answer = ""
        try:
            llm_client = _container.get_llm_client()
            system_prompt = LLM_PROMPTS.get(query_lang, LLM_PROMPTS["en"])["rag_system"]

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
                logger.error(f"Stream: LLM timed out ({query_lang}): {request.query[:100]}")
                fallback = "I found relevant information but the answer generation timed out. See sources below."
            else:
                logger.error(f"Stream: LLM failed ({query_lang}): {type(e).__name__}: {e}")
                fallback = "I found relevant information but couldn't generate a summary. See sources below."
            full_answer = fallback
            yield _sse_event("token", fallback)

        # Cache the full pipeline result
        answer_data = {
            "answer": full_answer,
            "sources": sources_serialized,
        }
        retriever._result_cache.set(
            request.query, [], client_id, cache_doc_id,
            query_embedding=original_embedding,
            answer_data=answer_data,
        )

        # Save messages to DB
        elapsed = (time.time() - start_time) * 1000
        _save_messages(conversation_id, request.query, full_answer, sources_serialized, elapsed)
        _auto_title(conversation_id, request.query)

        yield _sse_event("done", {"latency_ms": elapsed, "conversation_id": conversation_id})

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


# =============================================================================
# Document Family Endpoints
# =============================================================================

@app.get("/api/v1/families", response_model=list[FamilyResponse])
async def list_families(
    client: dict = Depends(get_authenticated_user),
):
    """List all document families for the authenticated user."""
    store = _container.get_store()
    families = store.list_families(client["user_id"])
    return [
        FamilyResponse(
            id=f["id"],
            name=f["name"],
            is_active=f["is_active"],
            document_count=f.get("document_count", 0),
            created_at=f["created_at"],
            updated_at=f["updated_at"],
        )
        for f in families
    ]


@app.post("/api/v1/families", response_model=FamilyResponse)
async def create_family(
    body: FamilyCreate,
    client: dict = Depends(get_authenticated_user),
):
    """Create a new document family."""
    store = _container.get_store()
    try:
        f = store.create_family(client["user_id"], body.name)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return FamilyResponse(
        id=f["id"],
        name=f["name"],
        is_active=f["is_active"],
        document_count=0,
        created_at=f["created_at"],
        updated_at=f["updated_at"],
    )


@app.put("/api/v1/families/{family_id}/name")
async def rename_family_endpoint(
    family_id: str,
    body: FamilyRename,
    client: dict = Depends(get_authenticated_user),
):
    """Rename a document family."""
    store = _container.get_store()
    try:
        updated = store.rename_family(family_id, client["user_id"], body.name)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    if not updated:
        raise HTTPException(status_code=404, detail="Family not found")
    return {"status": "updated", "name": body.name}


@app.put("/api/v1/families/{family_id}/active")
async def set_family_active_endpoint(
    family_id: str,
    body: FamilySetActive,
    client: dict = Depends(get_authenticated_user),
):
    """Toggle a family's active status (max 3 active enforced)."""
    store = _container.get_store()
    if body.is_active:
        count = store.get_active_family_count(client["user_id"])
        if count >= 3:
            raise HTTPException(
                status_code=400,
                detail="Maximum 3 active families allowed. Deactivate one first.",
            )
    updated = store.set_family_active(family_id, client["user_id"], body.is_active)
    if not updated:
        raise HTTPException(status_code=404, detail="Family not found")
    return {"status": "updated", "is_active": body.is_active}


@app.delete("/api/v1/families/{family_id}")
async def delete_family_endpoint(
    family_id: str,
    client: dict = Depends(get_authenticated_user),
):
    """Delete a document family. Documents are unassigned (not deleted)."""
    store = _container.get_store()
    deleted = store.delete_family(family_id, client["user_id"])
    if not deleted:
        raise HTTPException(status_code=404, detail="Family not found")
    _invalidate_client_cache(client["client_id"])
    return {"status": "deleted", "family_id": family_id}


@app.put("/api/v1/documents/{document_id}/family")
async def move_document_to_family_endpoint(
    document_id: str,
    body: MoveDocumentRequest,
    client: dict = Depends(get_authenticated_user),
):
    """Move a document to a family (or unassign with family_id=null)."""
    store = _container.get_store()
    updated = store.move_document_to_family(document_id, body.family_id, client["user_id"])
    if not updated:
        raise HTTPException(status_code=404, detail="Document not found")
    _invalidate_client_cache(client["client_id"])
    return {"status": "updated", "family_id": body.family_id}


# =============================================================================
# Auth Endpoints
# =============================================================================

@app.post("/api/v1/auth/google", response_model=AuthResponse)
async def google_auth(request: GoogleAuthRequest):
    """Exchange a Google ID token for a session JWT."""
    user_info = verify_google_token(request.id_token)
    if not user_info:
        raise HTTPException(status_code=401, detail="Invalid Google token")

    store = _container.get_store()
    user = store.create_or_get_user(
        google_sub=user_info["google_sub"],
        email=user_info["email"],
        name=user_info["name"],
        avatar_url=user_info["avatar_url"],
    )

    token = create_session_jwt(user["id"], user["email"], user["name"])

    return AuthResponse(
        token=token,
        user=UserInfo(
            id=user["id"],
            email=user["email"],
            name=user["name"],
            avatar_url=user["avatar_url"],
        ),
    )


@app.get("/api/v1/auth/me", response_model=UserInfo)
async def get_me(client: dict = Depends(get_authenticated_user)):
    """Get current authenticated user info."""
    if client.get("auth_method") == "jwt":
        store = _container.get_store()
        user = store.get_user_by_id(client["user_id"])
        if user:
            return UserInfo(
                id=user["id"],
                email=user["email"],
                name=user["name"],
                avatar_url=user["avatar_url"],
            )
    return UserInfo(
        id=client.get("user_id", ""),
        email=client.get("email", ""),
        name=client.get("name", ""),
    )


# =============================================================================
# Conversation Endpoints
# =============================================================================

@app.get("/api/v1/conversations", response_model=list[ConversationResponse])
async def list_conversations_endpoint(
    client: dict = Depends(get_authenticated_user),
):
    """List all conversations for the authenticated user (newest first)."""
    store = _container.get_store()
    convos = store.list_conversations(client["user_id"])
    return [
        ConversationResponse(
            id=c["id"],
            title=c["title"],
            created_at=c["created_at"],
            updated_at=c["updated_at"],
        )
        for c in convos
    ]


@app.post("/api/v1/conversations", response_model=ConversationResponse)
async def create_conversation_endpoint(
    body: ConversationCreate,
    client: dict = Depends(get_authenticated_user),
):
    """Create a new conversation."""
    store = _container.get_store()
    c = store.create_conversation(client["user_id"], title=body.title)
    return ConversationResponse(
        id=c["id"],
        title=c["title"],
        created_at=c["created_at"],
        updated_at=c["updated_at"],
    )


@app.delete("/api/v1/conversations/{conversation_id}")
async def delete_conversation_endpoint(
    conversation_id: str,
    client: dict = Depends(get_authenticated_user),
):
    """Delete a conversation and all its messages."""
    store = _container.get_store()
    deleted = store.delete_conversation(conversation_id, client["user_id"])
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"status": "deleted", "conversation_id": conversation_id}


@app.put("/api/v1/conversations/{conversation_id}/title")
async def rename_conversation_endpoint(
    conversation_id: str,
    body: ConversationRename,
    client: dict = Depends(get_authenticated_user),
):
    """Rename a conversation."""
    store = _container.get_store()
    updated = store.rename_conversation(conversation_id, client["user_id"], body.title)
    if not updated:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"status": "updated", "title": body.title}


@app.get("/api/v1/conversations/{conversation_id}/messages", response_model=list[MessageResponse])
async def get_messages_endpoint(
    conversation_id: str,
    client: dict = Depends(get_authenticated_user),
):
    """Get all messages for a conversation."""
    store = _container.get_store()
    msgs = store.get_messages(conversation_id, client["user_id"])
    return [
        MessageResponse(
            id=m["id"],
            role=m["role"],
            content=m["content"],
            sources=m["sources"],
            latency_ms=m["latency_ms"],
            created_at=m["created_at"],
        )
        for m in msgs
    ]
