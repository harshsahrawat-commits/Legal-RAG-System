"""
Hybrid Retriever for Legal Documents

Combines semantic (vector) search with keyword (BM25) search
using Reciprocal Rank Fusion, then reranks with Cohere.

This multi-stage approach provides ~40% better precision than
pure semantic search for legal document retrieval.
"""

import os
import re
import hashlib
import logging
import time
import numpy as np
from typing import Optional
from dataclasses import dataclass
from collections import OrderedDict

try:
    from .vector_store import VectorStore, SearchResult
    from .embeddings import EmbeddingService
    from .language_config import TenantLanguageConfig
    from .language_patterns import (
        QUERY_CLASSIFICATION,
        PARAGRAPH_REFERENCE_PATTERNS,
        LLM_PROMPTS,
        CROSS_LINGUAL_PROMPTS,
    )
except ImportError:
    from vector_store import VectorStore, SearchResult
    from embeddings import EmbeddingService
    from language_config import TenantLanguageConfig
    from language_patterns import (
        QUERY_CLASSIFICATION,
        PARAGRAPH_REFERENCE_PATTERNS,
        LLM_PROMPTS,
        CROSS_LINGUAL_PROMPTS,
    )

logger = logging.getLogger(__name__)


# ============================================================================
# Query Classification Configs (Advanced Latency Optimization)
# ============================================================================

# Different pipeline configurations for different query types
# This allows skipping expensive operations for simple queries

QUERY_CONFIGS = {
    "simple": {
        "use_query_expansion": False,
        "use_hyde": False,
        "use_multi_query": False,
        "description": "Short queries - skip all enhancement for speed"
    },
    "factual": {
        "use_query_expansion": True,
        "use_hyde": False,
        "use_multi_query": False,
        "description": "Factual queries - use expansion only"
    },
    "analytical": {
        "use_query_expansion": True,
        "use_hyde": True,
        "use_multi_query": True,
        "description": "Complex queries - use full enhancement"
    },
    "standard": {
        "use_query_expansion": True,
        "use_hyde": False,
        "use_multi_query": True,
        "description": "Default queries - expansion + multi-query"
    },
}


# ============================================================================
# Semantic Result Cache (Advanced Latency Optimization)
# ============================================================================

class QueryResultCache:
    """
    Cache retrieval results and generated answers with semantic similarity matching.

    If a new query is semantically similar to a cached query (>threshold),
    return cached results instead of re-running the full pipeline.

    Full-pipeline caching stores the generated answer + sources alongside
    retrieval results, eliminating all LLM calls on cache hits.

    Cache is invalidated per-client when documents are uploaded or deleted.
    """

    def __init__(
        self,
        embedding_service,
        similarity_threshold: float = 0.92,
        max_size: int = 500,
        ttl_seconds: int = 86400,  # 24 hours (legal docs are static)
    ):
        self._embeddings = embedding_service
        self._threshold = similarity_threshold
        self._max_size = max_size
        self._ttl = ttl_seconds

        # LRU cache: {cache_key: (query, client_id, doc_id, results, answer_data, timestamp)}
        # answer_data is Optional[dict] with keys: answer, sources (serialized SourceInfo list)
        self._cache = OrderedDict()
        # Store embeddings separately for comparison
        self._embedding_cache = {}  # {cache_key: embedding}

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        a_arr = np.array(a)
        b_arr = np.array(b)
        norm_a = np.linalg.norm(a_arr)
        norm_b = np.linalg.norm(b_arr)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a_arr, b_arr) / (norm_a * norm_b))

    def _make_cache_key(self, embedding: list[float]) -> str:
        """Create cache key from full embedding to avoid collisions."""
        # Sample every 32nd value for a 32-value fingerprint (fast + collision-resistant)
        sampled = embedding[::32] if len(embedding) > 32 else embedding
        return hashlib.sha256(str(sampled).encode()).hexdigest()[:16]

    def get(
        self,
        query: str,
        client_id: Optional[str] = None,
        document_id: Optional[str] = None,
        query_embedding: Optional[list[float]] = None,
    ) -> Optional[tuple]:
        """
        Check cache for semantically similar query.

        Returns (results, answer_data) tuple if found, None otherwise.
        answer_data may be None if only retrieval results were cached.
        """
        if not self._cache:
            return None

        # Use provided embedding to avoid redundant API calls
        query_emb = query_embedding if query_embedding is not None else self._embeddings.embed_query(query)
        current_time = time.time()

        # Check each cached query for similarity
        expired_keys = []
        for cache_key, (cached_query, cached_client, cached_doc, results, answer_data, timestamp) in self._cache.items():
            # Check TTL
            if current_time - timestamp > self._ttl:
                expired_keys.append(cache_key)
                continue

            # Must match client_id and document_id filters
            if cached_client != client_id or cached_doc != document_id:
                continue

            # Check semantic similarity
            cached_emb = self._embedding_cache.get(cache_key)
            if cached_emb is None:
                continue

            similarity = self._cosine_similarity(query_emb, cached_emb)
            if similarity >= self._threshold:
                logger.info(f"Cache hit! Query similarity: {similarity:.3f} >= {self._threshold}")
                # Move to end (most recently used)
                self._cache.move_to_end(cache_key)
                return (results, answer_data)

        # Clean up expired entries
        for key in expired_keys:
            self._cache.pop(key, None)
            self._embedding_cache.pop(key, None)

        return None

    def set(
        self,
        query: str,
        results: list[SearchResult],
        client_id: Optional[str] = None,
        document_id: Optional[str] = None,
        query_embedding: Optional[list[float]] = None,
        answer_data: Optional[dict] = None,
    ) -> None:
        """Cache query results and optionally the generated answer.

        Args:
            answer_data: Optional dict with 'answer' (str) and 'sources' (list of dicts).
                         When present, cache hits skip answer generation entirely.
        """
        query_emb = query_embedding if query_embedding is not None else self._embeddings.embed_query(query)
        cache_key = self._make_cache_key(query_emb)

        # Evict oldest if at capacity
        while len(self._cache) >= self._max_size:
            oldest_key, _ = self._cache.popitem(last=False)
            self._embedding_cache.pop(oldest_key, None)

        self._cache[cache_key] = (query, client_id, document_id, results, answer_data, time.time())
        self._embedding_cache[cache_key] = query_emb

    def invalidate_for_client(self, client_id: str) -> int:
        """Invalidate all cache entries for a client (call on document upload/delete).

        Returns the number of entries removed.
        """
        keys_to_remove = [
            k for k, (_, cached_client, *_) in self._cache.items()
            if cached_client == client_id
        ]
        for key in keys_to_remove:
            self._cache.pop(key, None)
            self._embedding_cache.pop(key, None)
        if keys_to_remove:
            logger.info(f"Invalidated {len(keys_to_remove)} cache entries for client {client_id}")
        return len(keys_to_remove)

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._embedding_cache.clear()

    @property
    def size(self) -> int:
        """Return current cache size."""
        return len(self._cache)


@dataclass
class RetrievalConfig:
    """Configuration for hybrid retrieval."""
    # Number of results from each search method
    vector_top_k: int = 40
    keyword_top_k: int = 40

    # Final results after reranking
    final_top_k: int = 10

    # RRF parameter (higher = more weight to top results)
    rrf_k: int = 60

    # Weight for vector vs keyword results
    vector_weight: float = 0.6
    keyword_weight: float = 0.4

    # Whether to use reranking
    use_reranking: bool = True

    # Document diversity: penalize same-document clustering in results
    use_document_diversity: bool = True
    diversity_decay_factor: float = 0.7  # 0.5 = aggressive, 0.8 = mild, 1.0 = off

    # Smart reranking: skip reranking when top result is highly confident
    # This reduces Cohere API costs by 30-50%
    use_smart_reranking: bool = True
    smart_rerank_threshold: float = 0.85  # Skip if top score > this
    smart_rerank_gap: float = 0.15  # Skip if gap to 2nd result > this

    # Query enhancement options (industry-standard improvements)
    use_query_expansion: bool = True  # Expand queries with legal terminology
    use_hyde: bool = True  # Hypothetical Document Embeddings
    use_multi_query: bool = True  # Generate multiple query variants


class HybridRetriever:
    """
    Multi-stage retrieval pipeline for legal documents.

    Pipeline:
    1. Parallel vector and keyword search
    2. Reciprocal Rank Fusion to combine results
    3. Cohere reranking for precision
    4. Return top results with citations
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_service: EmbeddingService,
        config: Optional[RetrievalConfig] = None,
        language_config: Optional[TenantLanguageConfig] = None,
    ):
        """
        Initialize retriever.

        Args:
            vector_store: Vector store instance
            embedding_service: Embedding service instance
            config: Optional retrieval configuration
            language_config: Per-tenant language configuration. Defaults to English.
        """
        self.store = vector_store
        self.embeddings = embedding_service
        self.config = config or RetrievalConfig()
        self._language_config = language_config or TenantLanguageConfig.for_language("en")
        self._lang = self._language_config.language
        self._reranker = None
        self._llm_client = None  # Cached OpenAI client for answer-quality tasks
        self._enhancement_llm_client = None  # Faster model for query enhancement
        self._alt_embedding_services = {}  # Cached alt embedding services by model name

        # Quick Win #4: Reranking cache to reduce API costs (bounded LRU)
        self._rerank_cache = OrderedDict()
        self._rerank_cache_max_size = 1000

        # Advanced Optimization: Semantic result cache
        # Caches full retrieval results + generated answers for similar queries.
        # 24-hour TTL is safe for legal documents which are rarely updated.
        self._result_cache = QueryResultCache(
            embedding_service=embedding_service,
            similarity_threshold=0.92,
            max_size=500,
            ttl_seconds=86400,  # 24 hours
        )

        # Initialize reranker if enabled
        if self.config.use_reranking:
            self._init_reranker()

    def _init_reranker(self):
        """Initialize Cohere reranker."""
        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            logger.warning("COHERE_API_KEY not found. Reranking disabled.")
            self.config.use_reranking = False
            return

        try:
            import cohere
            self._reranker = cohere.Client(api_key)
            logger.info("Cohere reranker initialized")
        except ImportError:
            logger.warning("Cohere not installed. Reranking disabled.")
            self.config.use_reranking = False

    # Faster model for query enhancement (expansion, HyDE, multi-query).
    # These are simple generation tasks that don't need a 235B model.
    ENHANCEMENT_MODEL = "meta/llama-3.3-70b-instruct"

    def _get_llm_client(self):
        """Get or create cached OpenAI client for NVIDIA NIM API."""
        if self._llm_client is None:
            from openai import OpenAI
            self._llm_client = OpenAI(
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=os.getenv("NVIDIA_API_KEY"),
                timeout=60.0,
            )
        return self._llm_client

    def _get_enhancement_llm_client(self):
        """Get or create cached OpenAI client for fast query enhancement.

        Uses a smaller, faster model (Llama 3.3 70B) for query expansion,
        HyDE, and multi-query generation — tasks that don't need a 235B model.
        This cuts enhancement latency from ~5-8s to ~1-2s per call.
        """
        if self._enhancement_llm_client is None:
            from openai import OpenAI
            self._enhancement_llm_client = OpenAI(
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=os.getenv("NVIDIA_API_KEY"),
                timeout=30.0,
            )
        return self._enhancement_llm_client

    def _get_alt_embedding_service(self, model_name: str):
        """Get or create an embedding service for an alternate embedding model.

        Used when searching sources ingested with a different embedding model
        than the tenant's default (e.g., CyLaw uses voyage-multilingual-2
        while an English tenant defaults to voyage-law-2).
        """
        if model_name not in self._alt_embedding_services:
            from .embeddings import get_embedding_service
            if model_name == "voyage-multilingual-2":
                alt_config = TenantLanguageConfig.for_language("el")
            else:
                alt_config = TenantLanguageConfig.for_language("en")
            self._alt_embedding_services[model_name] = get_embedding_service(
                provider=alt_config.embedding_provider,
                language_config=alt_config,
            )
        return self._alt_embedding_services[model_name]

    def _classify_query(self, query: str, lang: Optional[str] = None) -> str:
        """
        Classify query type for adaptive processing (language-aware).

        Different query types use different retrieval pipelines:
        - simple: Very short queries (≤3 words, no question words) - skip all enhancement
        - factual: "What/who/when/where is" - use expansion only
        - analytical: "Explain/analyze/compare/why" - use full enhancement
        - standard: Default - expansion + multi-query

        Returns:
            Query classification: "simple", "factual", "analytical", or "standard"
        """
        lang_patterns = QUERY_CLASSIFICATION.get(lang or self._lang, QUERY_CLASSIFICATION["en"])
        effective_lang = lang or self._lang
        query_lower = query.lower().strip()
        word_count = len(query.split())

        # Check for analytical question patterns FIRST (highest priority)
        for pattern in lang_patterns["analytical"]:
            if re.search(pattern, query_lower):
                # Cap Greek queries at "standard" — HyDE generates English-style
                # hypothetical docs which are less useful for Greek legal text,
                # and the extra LLM call adds ~30s latency causing timeouts.
                if effective_lang == "el":
                    logger.info(f"Query classified as 'standard' (analytical capped for Greek)")
                    return "standard"
                logger.info(f"Query classified as 'analytical' (pattern match)")
                return "analytical"

        # Check for factual question patterns SECOND
        for pattern in lang_patterns["factual"]:
            if re.search(pattern, query_lower):
                logger.info(f"Query classified as 'factual' (pattern match)")
                return "factual"

        # Check for specific legal terminology (use standard pipeline)
        legal_ref = lang_patterns["legal_reference"]
        if re.search(legal_ref, query_lower):
            logger.info(f"Query classified as 'standard' (legal reference)")
            return "standard"

        # Simple queries: very short AND no question words
        question_starters = lang_patterns["question_starters"]
        first_word = query_lower.split()[0] if query_lower.split() else ""
        if word_count <= 4 and first_word not in question_starters:
            logger.info(f"Query classified as 'simple' ({word_count} words, no question word)")
            return "simple"

        # Default to standard
        logger.info(f"Query classified as 'standard' (default, {word_count} words)")
        return "standard"

    def _get_effective_config(self, query_type: str) -> RetrievalConfig:
        """
        Get effective retrieval config based on query classification.

        This creates a modified config with query-type-specific settings
        while preserving other settings from the base config.
        """
        query_config = QUERY_CONFIGS.get(query_type, QUERY_CONFIGS["standard"])

        # Create a copy of the base config with query-specific overrides
        from dataclasses import replace
        effective = replace(
            self.config,
            use_query_expansion=query_config["use_query_expansion"],
            use_hyde=query_config["use_hyde"],
            use_multi_query=query_config["use_multi_query"],
        )

        logger.info(f"Using '{query_type}' config: {query_config['description']}")
        return effective

    def _extract_paragraph_reference(self, query: str, lang: Optional[str] = None) -> Optional[int]:
        """
        Extract paragraph number from query if present (language-aware).

        Detects patterns like:
        - "paragraph 28", "para 28", "para. 28", "¶28"
        - Greek: "παράγραφος 28", "παρ. 28"

        Returns:
            Paragraph number if found, None otherwise
        """
        patterns = PARAGRAPH_REFERENCE_PATTERNS.get(lang or self._lang, PARAGRAPH_REFERENCE_PATTERNS["en"])
        query_lower = query.lower()

        for pattern in patterns:
            match = re.search(pattern, query_lower, re.IGNORECASE)
            if match:
                para_num = int(match.group(1))
                logger.info(f"Detected paragraph reference: {para_num} in query")
                return para_num

        return None

    def _expand_query(self, query: str, lang: Optional[str] = None) -> str:
        """
        Expand query with legal terminology and synonyms (language-aware).

        Uses an LLM to add relevant legal terms that improve retrieval.
        This addresses the semantic gap between user queries and legal documents.
        """
        try:
            client = self._get_enhancement_llm_client()

            system_prompt = LLM_PROMPTS.get(lang or self._lang, LLM_PROMPTS["en"])["query_expansion_system"]

            response = client.chat.completions.create(
                model=self.ENHANCEMENT_MODEL,
                messages=[{
                    "role": "system",
                    "content": system_prompt,
                }, {
                    "role": "user",
                    "content": f"Expand: {query}"
                }],
                max_tokens=150,
                temperature=0.2,
            )

            raw = response.choices[0].message.content
            expanded = raw.strip() if raw else query
            logger.info(f"Query expanded: '{query}' → '{expanded[:100]}...'")
            return expanded

        except Exception as e:
            logger.warning(f"Query expansion failed: {e}. Using original query.")
            return query

    def _generate_hyde_document(self, query: str, lang: Optional[str] = None) -> str:
        """
        Generate a Hypothetical Document for embedding (HyDE, language-aware).

        Creates a hypothetical answer that would contain the information
        the user is looking for. The embedding of this hypothetical
        document often retrieves better results than the query alone.
        """
        try:
            client = self._get_enhancement_llm_client()

            system_prompt = LLM_PROMPTS.get(lang or self._lang, LLM_PROMPTS["en"])["hyde_system"]

            response = client.chat.completions.create(
                model=self.ENHANCEMENT_MODEL,
                messages=[{
                    "role": "system",
                    "content": system_prompt,
                }, {
                    "role": "user",
                    "content": f"Write a court document excerpt answering: {query}"
                }],
                max_tokens=200,
                temperature=0.3,
            )

            raw = response.choices[0].message.content
            hyde_doc = raw.strip() if raw else query
            logger.info(f"Generated HyDE document for query: '{query[:50]}...'")
            return hyde_doc

        except Exception as e:
            logger.warning(f"HyDE generation failed: {e}. Using original query.")
            return query

    def _generate_query_variants(self, query: str, lang: Optional[str] = None) -> list[str]:
        """
        Generate multiple query variants for multi-query retrieval (language-aware).

        Different phrasings can retrieve different relevant documents.
        Results from all variants are combined using RRF.
        """
        try:
            client = self._get_enhancement_llm_client()

            system_prompt = LLM_PROMPTS.get(lang or self._lang, LLM_PROMPTS["en"])["multi_query_system"]

            response = client.chat.completions.create(
                model=self.ENHANCEMENT_MODEL,
                messages=[{
                    "role": "system",
                    "content": system_prompt,
                }, {
                    "role": "user",
                    "content": f"Generate variants for: {query}"
                }],
                max_tokens=200,
                temperature=0.7,
            )

            raw = response.choices[0].message.content
            if not raw:
                return [query]
            variants = [v.strip() for v in raw.strip().split('\n') if v.strip()]
            logger.info(f"Generated {len(variants)} query variants")
            return [query] + variants[:3]  # Original + up to 3 variants

        except Exception as e:
            logger.warning(f"Query variant generation failed: {e}. Using original query only.")
            return [query]

    # Source language mapping: which document sources are in which language
    SOURCE_LANGUAGES = {
        "cylaw": "el",   # CyLaw docs are in Greek
        "hudoc": "en",   # HUDOC docs are in English
        "eurlex": "en",  # EUR-Lex docs are in English
    }

    # Source embedding model mapping: which model was used to embed each source's documents.
    # Must match the model used during batch ingestion for meaningful cosine similarity.
    SOURCE_EMBEDDING_MODELS = {
        "cylaw": "voyage-multilingual-2",  # Greek legal docs
        "hudoc": "voyage-law-2",           # English ECHR docs
        "eurlex": "voyage-law-2",          # English EU law docs
    }

    def _translate_query(self, query: str, target_lang: str) -> Optional[str]:
        """Translate a query to the target language for cross-lingual retrieval.

        Used when the query language doesn't match the source document language
        (e.g. Greek query searching English HUDOC docs, or English query searching Greek CyLaw).

        Args:
            query: Original query text
            target_lang: Target language code ("en" or "el")

        Returns:
            Translated query string, or None if translation fails
        """
        prompt_key = f"to_{target_lang}"
        system_prompt = CROSS_LINGUAL_PROMPTS.get(prompt_key)
        if not system_prompt:
            return None

        try:
            client = self._get_enhancement_llm_client()
            response = client.chat.completions.create(
                model=self.ENHANCEMENT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Translate: {query}"},
                ],
                max_tokens=150,
                temperature=0.2,
            )
            raw = response.choices[0].message.content
            translated = raw.strip() if raw else None
            if translated:
                logger.info(f"Cross-lingual translation ({target_lang}): '{query[:50]}' → '{translated[:50]}'")
            return translated
        except Exception as e:
            logger.warning(f"Cross-lingual translation to {target_lang} failed: {e}")
            return None

    def _needs_cross_lingual(self, query_lang: str, source_origins: Optional[list[str]]) -> Optional[str]:
        """Check if cross-lingual search is needed and return the other language if so.

        Returns the "other" language code if sources in a different language are active,
        or None if all active sources match the query language.
        """
        if not source_origins:
            return None

        source_langs = set(self.SOURCE_LANGUAGES.get(s, "en") for s in source_origins)

        # If sources include a language different from the query, we need cross-lingual
        other_langs = source_langs - {query_lang}
        if other_langs:
            return other_langs.pop()  # Return the other language ("en" or "el")
        return None

    def retrieve(
        self,
        query: str,
        client_id: Optional[str] = None,
        document_id: Optional[str] = None,
        top_k: Optional[int] = None,
        use_cache: bool = True,
        query_embedding: Optional[list[float]] = None,
        source_origins: Optional[list[str]] = None,
        family_ids: Optional[list[str]] = None,
        conversation_id: Optional[str] = None,
        query_lang: Optional[str] = None,
    ) -> list[SearchResult]:
        """
        Retrieve relevant chunks for a query.

        Enhanced with industry-standard techniques:
        - Query classification: Adaptive pipeline for different query types
        - Semantic caching: Skip retrieval for similar queries
        - Query expansion: Add legal terminology
        - HyDE: Hypothetical Document Embeddings
        - Multi-query: Multiple query variants

        Args:
            query: Search query string
            client_id: Optional client filter
            document_id: Optional document filter
            top_k: Number of results (defaults to config)
            use_cache: Whether to check/use semantic result cache
            source_origins: Optional list of source origins to filter by (e.g. ["cylaw", "hudoc", "eurlex"])
            query_lang: Override language for this query ("en" or "el"). If None, uses tenant config.

        Returns:
            List of SearchResult objects, ranked by relevance
        """
        start_time = time.time()
        top_k = top_k or self.config.final_top_k

        # Per-query language override for prompts and FTS
        effective_lang = query_lang or self._lang
        effective_fts = "greek" if effective_lang == "el" else self._language_config.fts_language

        logger.info(f"Retrieving for query ({effective_lang}): {query[:50]}...")

        # ============================================================
        # DIRECT PARAGRAPH LOOKUP: Skip full pipeline for paragraph refs
        # ============================================================
        para_num = self._extract_paragraph_reference(query, effective_lang)
        if para_num is not None and document_id:
            para_results = self.store.search_by_paragraph(
                document_id=document_id,
                paragraph_number=para_num,
                client_id=client_id,
            )
            if para_results:
                elapsed = (time.time() - start_time) * 1000
                logger.info(f"Paragraph {para_num} found directly in {elapsed:.0f}ms")
                return para_results[:top_k]
            # Fall through to normal retrieval if paragraph not found

        # ============================================================
        # ADVANCED OPTIMIZATION 1: Check semantic result cache
        # ============================================================
        # Compute the original query embedding once — reuse for cache + search
        # Accept pre-computed embedding from caller to avoid redundant API calls
        original_embedding = query_embedding if query_embedding is not None else self.embeddings.embed_query(query)

        if use_cache:
            cache_hit = self._result_cache.get(
                query, client_id, document_id, query_embedding=original_embedding,
            )
            if cache_hit is not None:
                cached_results, _ = cache_hit  # answer_data handled by API layer
                elapsed = (time.time() - start_time) * 1000
                logger.info(f"Cache hit! Returning {len(cached_results)} cached results in {elapsed:.0f}ms")
                return cached_results[:top_k]

        # ============================================================
        # ADVANCED OPTIMIZATION 2: Query classification
        # ============================================================
        query_type = self._classify_query(query, effective_lang)
        effective_config = self._get_effective_config(query_type)

        # Query Enhancement (based on query classification)
        # Run independent LLM enhancement calls in parallel for speed
        search_queries = [query]  # Always include original
        search_embeddings = []

        needs_expansion = effective_config.use_query_expansion
        needs_hyde = effective_config.use_hyde
        needs_multi = effective_config.use_multi_query
        # Check if cross-lingual translation is needed (query lang != source lang)
        other_lang = self._needs_cross_lingual(effective_lang, source_origins)
        needs_translate = other_lang is not None and query_type != "simple"

        from concurrent.futures import ThreadPoolExecutor, as_completed

        if any([needs_expansion, needs_hyde, needs_multi, needs_translate]):
            futures = {}
            with ThreadPoolExecutor(max_workers=4) as executor:
                if needs_expansion:
                    futures["expand"] = executor.submit(self._expand_query, query, effective_lang)
                if needs_hyde:
                    futures["hyde"] = executor.submit(self._generate_hyde_document, query, effective_lang)
                if needs_multi:
                    futures["variants"] = executor.submit(self._generate_query_variants, query, effective_lang)
                if needs_translate:
                    futures["translate"] = executor.submit(self._translate_query, query, other_lang)

            # Collect results
            if "expand" in futures:
                expanded_query = futures["expand"].result()
                if expanded_query != query:
                    search_queries.append(expanded_query)

            if "hyde" in futures:
                hyde_doc = futures["hyde"].result()
                if hyde_doc != query:
                    hyde_embedding = self.embeddings.embed_query(hyde_doc)
                    search_embeddings.append(("hyde", hyde_doc, hyde_embedding))

            if "variants" in futures:
                variants = futures["variants"].result()
                for variant in variants[1:]:
                    if variant not in search_queries:
                        search_queries.append(variant)

            translated_query = futures["translate"].result() if "translate" in futures else None
        else:
            translated_query = None

        # Cap search queries to avoid excessive DB calls (original + 2 best)
        if len(search_queries) > 3:
            search_queries = search_queries[:3]

        # ============================================================
        # Stage 1: PARALLEL search — per-source-model embeddings
        #
        # Different sources may have been ingested with different embedding
        # models (e.g. CyLaw with voyage-multilingual-2, HUDOC with
        # voyage-law-2). We group sources by model and generate the
        # correct query embedding for each group so cosine similarity
        # is computed within the same vector space.
        # ============================================================
        all_vector_results = []
        all_keyword_results = []

        # 1a. Group source origins by their ingestion embedding model
        primary_model = self._language_config.embedding_model
        model_groups = {}  # {model_name: [source_origins]}
        if source_origins:
            for s in source_origins:
                model = self.SOURCE_EMBEDDING_MODELS.get(s, primary_model)
                model_groups.setdefault(model, []).append(s)
        # Family and session docs are user-uploaded → use tenant's primary model.
        # Ensure a primary model group exists when these are active.
        if (family_ids or conversation_id) and primary_model not in model_groups:
            model_groups[primary_model] = []
        # Fallback: no sources and no families → search everything with primary model
        if not model_groups:
            model_groups[primary_model] = None

        # 1b. Batch-compute embeddings for primary model
        query_embeddings = []
        for i, sq in enumerate(search_queries):
            if i == 0 and sq == query:
                query_embeddings.append(original_embedding)
            else:
                query_embeddings.append(self.embeddings.embed_query(sq))

        # Reduced top_k for variant queries (they add diversity, not depth)
        variant_top_k = max(effective_config.vector_top_k // 2, 15)

        # 1c. Build search tasks per embedding-model group
        search_tasks = []  # (result_type, callable)

        for model_name, group_sources in model_groups.items():
            # Compute embeddings for this group
            if model_name == primary_model:
                group_embeddings = query_embeddings
                group_hyde = [(name, emb) for name, _text, emb in search_embeddings]
            else:
                alt_service = self._get_alt_embedding_service(model_name)
                group_embeddings = [alt_service.embed_query(sq) for sq in search_queries]
                group_hyde = [(name, alt_service.embed_query(text))
                              for name, text, _emb in search_embeddings]

            # FTS language for this group (based on source document languages)
            if group_sources:
                group_langs = set(self.SOURCE_LANGUAGES.get(s, "en") for s in group_sources)
                group_fts = "greek" if "el" in group_langs else "english"
            else:
                group_fts = effective_fts

            # Only include family_ids/conversation_id with the tenant's primary model
            include_extras = (model_name == primary_model)
            group_common = dict(
                client_id=client_id,
                document_id=document_id,
                source_origins=group_sources,
                family_ids=family_ids if include_extras else None,
                conversation_id=conversation_id if include_extras else None,
            )

            for i, (sq, emb) in enumerate(zip(search_queries, group_embeddings)):
                tk = effective_config.vector_top_k if i == 0 else variant_top_k
                search_tasks.append(("vector", lambda e=emb, t=tk, gc=group_common: self.store.search(
                    query_embedding=e, top_k=t, **gc,
                )))
                search_tasks.append(("keyword", lambda q=sq, t=tk, f=group_fts, gc=group_common: self.store.keyword_search(
                    query=q, top_k=t, fts_language=f, **gc,
                )))

            # HyDE vector searches for this group
            for _name, hyde_emb in group_hyde:
                search_tasks.append(("vector", lambda e=hyde_emb, gc=group_common: self.store.search(
                    query_embedding=e, top_k=variant_top_k, **gc,
                )))

        # Cross-lingual searches: translate query for cross-language sources
        if translated_query:
            cross_fts = "greek" if other_lang == "el" else "english"
            cross_sources = [s for s in (source_origins or [])
                            if self.SOURCE_LANGUAGES.get(s, "en") == other_lang]
            if cross_sources:
                logger.info(f"Cross-lingual search ({effective_lang}->{other_lang}): '{translated_query[:60]}' sources={cross_sources}")
                # Use the embedding model that matches the cross-lingual target sources
                cross_model = self.SOURCE_EMBEDDING_MODELS.get(cross_sources[0], primary_model)
                if cross_model != primary_model:
                    cross_emb_service = self._get_alt_embedding_service(cross_model)
                    cross_embedding = cross_emb_service.embed_query(translated_query)
                else:
                    cross_embedding = self.embeddings.embed_query(translated_query)
                cross_common = dict(
                    client_id=client_id,
                    document_id=document_id,
                    source_origins=cross_sources,
                    family_ids=None,
                    conversation_id=None,
                )
                search_tasks.append(("vector", lambda e=cross_embedding: self.store.search(
                    query_embedding=e, top_k=variant_top_k, **cross_common,
                )))
                search_tasks.append(("keyword", lambda q=translated_query, f=cross_fts: self.store.keyword_search(
                    query=q, top_k=variant_top_k, fts_language=f, **cross_common,
                )))

        # 1d. Execute all search tasks in parallel
        # Cap workers at 6 to stay within Neon Postgres connection limits
        logger.info(f"Running {len(search_tasks)} search tasks in parallel")
        with ThreadPoolExecutor(max_workers=min(len(search_tasks), 6)) as executor:
            future_map = {executor.submit(fn): rtype for rtype, fn in search_tasks}
            for future in as_completed(future_map):
                rtype = future_map[future]
                try:
                    results = future.result()
                    if rtype == "vector":
                        all_vector_results.extend(results)
                    else:
                        all_keyword_results.extend(results)
                except Exception as e:
                    logger.warning(f"Search task ({rtype}) failed: {e}")

        logger.debug(f"Total vector results: {len(all_vector_results)}, keyword results: {len(all_keyword_results)}")

        # Stage 2: Combine with Reciprocal Rank Fusion
        fused_results = self._reciprocal_rank_fusion(
            all_vector_results,
            all_keyword_results,
        )
        logger.debug(f"RRF produced {len(fused_results)} combined results")

        # Stage 3: Rerank if enabled (with smart skip for cost savings)
        if effective_config.use_reranking and self._reranker and fused_results:
            # Check if we can skip reranking (smart reranking for cost savings)
            if self._should_skip_reranking(fused_results):
                final_results = fused_results[:top_k * 2]
            else:
                reranked_results = self._rerank(query, fused_results[:top_k * 2])
                final_results = reranked_results[:top_k * 2]
        else:
            final_results = fused_results[:top_k * 2]

        # Stage 4: Document diversity enforcement
        # Only for analytical/standard queries across multiple documents —
        # simple/factual queries typically target one document, and
        # single-document filters make diversity meaningless.
        if (effective_config.use_document_diversity
                and query_type in ("analytical", "standard")
                and not document_id):
            final_results = self._enforce_document_diversity(
                final_results,
                top_k=top_k,
                decay_factor=effective_config.diversity_decay_factor,
            )
        else:
            final_results = final_results[:top_k]

        # ============================================================
        # Cache results for future similar queries
        # ============================================================
        if use_cache and final_results:
            self._result_cache.set(
                query, final_results, client_id, document_id,
                query_embedding=original_embedding,
            )

        elapsed = (time.time() - start_time) * 1000
        logger.info(f"Returning {len(final_results)} results ({query_type} pipeline) in {elapsed:.0f}ms")
        return final_results

    def _reciprocal_rank_fusion(
        self,
        vector_results: list[SearchResult],
        keyword_results: list[SearchResult],
    ) -> list[SearchResult]:
        """
        Combine results using Reciprocal Rank Fusion.

        RRF score = sum(1 / (k + rank)) across result lists
        This is more robust than simple score averaging.
        """
        k = self.config.rrf_k
        scores = {}
        result_map = {}

        # Score vector results
        for rank, result in enumerate(vector_results):
            chunk_id = result.chunk_id
            rrf_score = self.config.vector_weight / (k + rank + 1)
            scores[chunk_id] = scores.get(chunk_id, 0) + rrf_score
            result_map[chunk_id] = result

        # Score keyword results
        for rank, result in enumerate(keyword_results):
            chunk_id = result.chunk_id
            rrf_score = self.config.keyword_weight / (k + rank + 1)
            scores[chunk_id] = scores.get(chunk_id, 0) + rrf_score
            if chunk_id not in result_map:
                result_map[chunk_id] = result

        # Sort by RRF score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        # Build result list with copies to avoid mutating cached originals
        # Preserve original vector score for display purposes
        from dataclasses import replace as _replace
        results = []
        for chunk_id in sorted_ids:
            original = result_map[chunk_id]
            new_metadata = {**original.metadata, "original_score": original.score}
            result = _replace(original, score=scores[chunk_id], metadata=new_metadata)
            results.append(result)

        return results

    def _get_rerank_cache_key(self, query: str, chunk_ids: list[str]) -> str:
        """Generate cache key from query and chunk IDs."""
        content = f"{query}::{','.join(sorted(chunk_ids))}"
        return hashlib.sha256(content.encode()).hexdigest()

    def _should_skip_reranking(self, results: list[SearchResult]) -> bool:
        """
        Determine if reranking can be skipped for cost savings.

        Smart reranking: Skip the expensive reranker API call when:
        1. Top result has very high confidence score
        2. There's a significant gap between top and second result

        This reduces Cohere API costs by 30-50% without hurting quality
        on "easy" queries where the initial retrieval is clearly correct.

        Returns:
            True if reranking should be skipped, False otherwise
        """
        if not self.config.use_smart_reranking:
            return False

        if len(results) < 2:
            return True  # Not enough results to rerank

        # Use original cosine similarity scores, not RRF scores
        # (RRF scores max out at ~0.01, so comparing against 0.85 threshold would never match)
        top_score = results[0].metadata.get("original_score", results[0].score)
        second_score = results[1].metadata.get("original_score", results[1].score)

        # Skip if top result is very confident AND has a clear lead
        if (top_score > self.config.smart_rerank_threshold and
            (top_score - second_score) > self.config.smart_rerank_gap):
            logger.info(
                f"Smart rerank: SKIPPING (top={top_score:.3f}, gap={top_score-second_score:.3f})"
            )
            return True

        return False

    def _enforce_document_diversity(
        self,
        results: list[SearchResult],
        top_k: int,
        decay_factor: float = 0.7,
    ) -> list[SearchResult]:
        """
        Re-score results to penalize same-document clustering.

        Each subsequent chunk from the same document has its score
        multiplied by decay_factor^(occurrence_count). The list is
        then re-sorted by adjusted score and truncated to top_k.

        With decay_factor=0.7: 1st chunk keeps 100%, 2nd 70%, 3rd 49%, 4th 34%.
        This ensures analytical queries surface chunks from multiple documents
        without hard caps that could hurt precision when one document truly dominates.
        """
        if not results or len(results) <= 1:
            return results

        doc_counts: dict[str, int] = {}
        adjusted = []

        for result in results:
            doc_id = result.document_id
            count = doc_counts.get(doc_id, 0)
            penalty = decay_factor ** count
            adjusted_score = result.score * penalty
            adjusted.append((adjusted_score, count, result))
            doc_counts[doc_id] = count + 1

        # Re-sort by adjusted score
        adjusted.sort(key=lambda x: x[0], reverse=True)

        from dataclasses import replace as _replace
        final = []
        for adj_score, doc_occurrence, result in adjusted[:top_k]:
            new_metadata = {
                **result.metadata,
                "diversity_score": adj_score,
                "doc_occurrence": doc_occurrence,
            }
            final.append(_replace(result, metadata=new_metadata))

        unique_docs = len(set(r.document_id for r in final))
        logger.info(f"Document diversity: {unique_docs} unique docs in top-{len(final)} (decay={decay_factor})")
        return final

    def _rerank(
        self,
        query: str,
        results: list[SearchResult],
    ) -> list[SearchResult]:
        """
        Rerank results using Cohere's reranker with caching.

        The reranker is trained specifically for relevance ranking
        and provides significant precision improvements.

        Quick Win #4: Cache rerank results to reduce API costs.
        """
        if not results:
            return results

        # Check cache first
        chunk_ids = [r.chunk_id for r in results]
        cache_key = self._get_rerank_cache_key(query, chunk_ids)

        if cache_key in self._rerank_cache:
            logger.debug("Rerank cache hit - returning cached results")
            self._rerank_cache.move_to_end(cache_key)
            return self._rerank_cache[cache_key]

        try:
            response = self._reranker.rerank(
                model=self._language_config.reranker_model,
                query=query,
                documents=[r.content for r in results],
                top_n=len(results),
            )

            # Reorder results based on reranker scores (copy to avoid mutating originals)
            from dataclasses import replace as _replace
            reranked = []
            for item in response.results:
                original = results[item.index]
                new_metadata = {**original.metadata, "rerank_score": item.relevance_score}
                result = _replace(original, score=item.relevance_score, metadata=new_metadata)
                reranked.append(result)

            # Cache the result (bounded LRU)
            self._rerank_cache[cache_key] = reranked
            while len(self._rerank_cache) > self._rerank_cache_max_size:
                self._rerank_cache.popitem(last=False)
            logger.debug(f"Cached rerank results (cache size: {len(self._rerank_cache)})")

            return reranked

        except Exception as e:
            logger.warning(f"Reranking failed: {e}. Using original order.")
            return results



# Factory function
def get_retriever(
    vector_store: VectorStore,
    embedding_service: EmbeddingService,
    use_hybrid: bool = True,
) -> HybridRetriever:
    """
    Get configured retriever instance.

    Args:
        vector_store: Vector store instance
        embedding_service: Embedding service instance
        use_hybrid: Unused, kept for backward compatibility. Always returns HybridRetriever.

    Returns:
        Configured HybridRetriever instance
    """
    return HybridRetriever(vector_store, embedding_service)


# CLI for testing
if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    # Initialize components
    store = VectorStore()
    store.connect()

    embeddings = EmbeddingService()
    retriever = HybridRetriever(store, embeddings)

    # Test query
    query = sys.argv[1] if len(sys.argv) > 1 else "termination clause"

    print(f"\nSearching for: {query}")
    print("-" * 50)

    results = retriever.retrieve(query, top_k=5)

    for i, result in enumerate(results, 1):
        print(f"\n{i}. [{result.section_title}] (score: {result.score:.4f})")
        print(f"   Path: {result.hierarchy_path}")
        print(f"   Preview: {result.content[:200]}...")
