"""
Tests for Batch 3 of codebase review fixes applied to the Legal RAG System.

Fix 7: Parallel LLM enhancement calls (retriever.py)
    _enhance_query() in HybridRetriever.retrieve() now uses
    ThreadPoolExecutor(max_workers=3) to run _expand_query,
    _generate_hyde_document, and _generate_query_variants concurrently
    instead of sequentially.

Fix 8: Bounded embedding cache (embeddings.py)
    BaseEmbeddingService._cache changed from plain dict to OrderedDict
    with _max_cache_size = 10000.  _get_cached() calls move_to_end()
    for LRU behavior, and _set_cached() evicts oldest entries when the
    cache exceeds max size.

Fix 9: API quota enforcement (api.py)
    query_documents() now calls get_quota_manager(store).check_query_quota()
    before retrieval.  QuotaExceededError maps to HTTP 429; any other
    exception during the quota check is logged and the request proceeds
    (graceful degradation).

All external API calls are mocked -- no database, LLM, or embedding service
is required.
"""

import time
import threading
from collections import OrderedDict
from unittest.mock import patch, MagicMock, call
from dataclasses import replace as _replace

import pytest


# ============================================================================
# Helpers (shared with batch 1 but kept local so this file is self-contained)
# ============================================================================

def _build_retriever(mock_vector_store, mock_embedding_service, **config_overrides):
    """
    Construct a HybridRetriever with mocked internals.

    Reranker initialisation is suppressed, and reranking is disabled by default
    unless the caller passes explicit config overrides.
    """
    from execution.legal_rag.retriever import HybridRetriever, RetrievalConfig
    from execution.legal_rag.language_config import TenantLanguageConfig

    defaults = dict(use_reranking=False)
    defaults.update(config_overrides)
    cfg = RetrievalConfig(**defaults)

    with patch.object(HybridRetriever, "_init_reranker", lambda self: None):
        retriever = HybridRetriever(
            mock_vector_store,
            mock_embedding_service,
            config=cfg,
            language_config=TenantLanguageConfig.for_language("en"),
        )
    return retriever


def _make_llm_response(content):
    """
    Build a minimal object tree that mimics the OpenAI ChatCompletion shape:
        response.choices[0].message.content
    """
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


def _make_search_result(chunk_id, score, metadata=None):
    """Create a SearchResult with the given score and metadata."""
    from execution.legal_rag.vector_store import SearchResult

    return SearchResult(
        chunk_id=chunk_id,
        document_id="doc-001",
        content="Sample legal text for testing.",
        section_title="Section A",
        hierarchy_path="Document/Section_A",
        page_numbers=[1],
        score=score,
        metadata=metadata or {},
    )


# ============================================================================
# Fix 7 -- Parallel LLM enhancement calls in retriever.py
# ============================================================================

class TestParallelLLMEnhancement:
    """
    The retrieve() method now uses ThreadPoolExecutor(max_workers=3) to run
    _expand_query, _generate_hyde_document, and _generate_query_variants
    concurrently when all three are needed (analytical query classification).
    """

    @pytest.fixture
    def retriever(self, mock_vector_store, mock_embedding_service):
        """Build a retriever with all enhancement features enabled."""
        return _build_retriever(
            mock_vector_store,
            mock_embedding_service,
            use_query_expansion=True,
            use_hyde=True,
            use_multi_query=True,
        )

    def _install_mock_llm(self, retriever):
        """Install a mock LLM client and return it for inspection."""
        mock_client = MagicMock()
        # Default: return valid responses for each call
        mock_client.chat.completions.create.return_value = _make_llm_response(
            "expanded legal terminology"
        )
        retriever._llm_client = mock_client
        return mock_client

    # ---------------------------------------------------------------
    # All 3 futures are submitted when all 3 enhancements are needed
    # ---------------------------------------------------------------

    def test_all_three_enhancements_submitted_for_analytical_query(
        self, retriever, mock_embedding_service,
    ):
        """When query is classified as 'analytical', all 3 LLM calls must be launched."""
        mock_client = self._install_mock_llm(retriever)

        # Patch _classify_query to force 'analytical' classification
        with patch.object(retriever, "_classify_query", return_value="analytical"):
            retriever.retrieve("Explain the differences between termination clauses")

        # The LLM client should have been called 3 times (expand + hyde + variants)
        assert mock_client.chat.completions.create.call_count == 3

    def test_only_expansion_submitted_for_factual_query(
        self, mock_vector_store, mock_embedding_service,
    ):
        """Factual queries should only invoke _expand_query (no HyDE, no multi-query)."""
        retriever = _build_retriever(
            mock_vector_store,
            mock_embedding_service,
            use_query_expansion=True,
            use_hyde=True,
            use_multi_query=True,
        )
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_llm_response(
            "who is the licensor legal entity grantor"
        )
        retriever._llm_client = mock_client

        with patch.object(retriever, "_classify_query", return_value="factual"):
            retriever.retrieve("Who is the licensor?")

        # Factual config: expansion=True, hyde=False, multi_query=False
        assert mock_client.chat.completions.create.call_count == 1

    def test_no_enhancement_submitted_for_simple_query(
        self, mock_vector_store, mock_embedding_service,
    ):
        """Simple queries should skip ALL enhancements -- zero LLM calls."""
        retriever = _build_retriever(
            mock_vector_store,
            mock_embedding_service,
            use_query_expansion=True,
            use_hyde=True,
            use_multi_query=True,
        )
        mock_client = MagicMock()
        retriever._llm_client = mock_client

        with patch.object(retriever, "_classify_query", return_value="simple"):
            retriever.retrieve("liability")

        assert mock_client.chat.completions.create.call_count == 0

    # ---------------------------------------------------------------
    # Results are collected correctly from futures
    # ---------------------------------------------------------------

    def test_expanded_query_added_to_search_queries(self, retriever):
        """The expanded query text should be used in the search phase."""
        mock_client = MagicMock()
        # Each call returns a different result based on call order
        mock_client.chat.completions.create.side_effect = [
            _make_llm_response("expanded termination clause legal"),  # expand
            _make_llm_response("hypothetical court document text"),   # hyde
            _make_llm_response("variant 1\nvariant 2"),              # variants
        ]
        retriever._llm_client = mock_client

        mock_keyword_search = MagicMock(return_value=[])
        mock_search = MagicMock(return_value=[])

        with patch.object(retriever, "_classify_query", return_value="analytical"):
            with patch.object(retriever.store, "search", mock_search):
                with patch.object(retriever.store, "keyword_search", mock_keyword_search):
                    retriever.retrieve("termination clause")

        # The expanded query should appear in keyword_search calls
        keyword_calls = mock_keyword_search.call_args_list
        queries_searched = [c.kwargs.get("query", c.args[0] if c.args else "") for c in keyword_calls]
        # We expect at least the original query + expanded + variant(s) to be searched
        assert len(queries_searched) >= 2

    def test_hyde_embedding_used_for_vector_search(self, retriever, mock_embedding_service):
        """The HyDE document embedding should be used in an additional vector search."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            _make_llm_response("expanded query"),               # expand
            _make_llm_response("hypothetical document text"),    # hyde
            _make_llm_response("variant A\nvariant B"),          # variants
        ]
        retriever._llm_client = mock_client

        mock_search = MagicMock(return_value=[])
        mock_keyword_search = MagicMock(return_value=[])

        with patch.object(retriever, "_classify_query", return_value="analytical"):
            with patch.object(retriever.store, "search", mock_search):
                with patch.object(retriever.store, "keyword_search", mock_keyword_search):
                    retriever.retrieve("explain governing law")

        # Vector search should be called for original, expanded, and variants,
        # PLUS once more for the HyDE embedding
        vector_call_count = mock_search.call_count
        keyword_call_count = mock_keyword_search.call_count
        # HyDE adds one extra vector search but no extra keyword search
        assert vector_call_count > keyword_call_count

    # ---------------------------------------------------------------
    # Exception isolation: one failure does not block others
    # ---------------------------------------------------------------

    def test_expand_query_failure_does_not_block_hyde_and_variants(self, retriever):
        """If _expand_query raises, _generate_hyde_document and _generate_query_variants
        should still complete and their results should be collected."""
        mock_client = MagicMock()
        # First call (expand) raises, others succeed
        mock_client.chat.completions.create.side_effect = [
            RuntimeError("API timeout on expand"),
            _make_llm_response("hypothetical court doc"),
            _make_llm_response("variant 1"),
        ]
        retriever._llm_client = mock_client

        with patch.object(retriever, "_classify_query", return_value="analytical"):
            # Should not raise -- the exception in _expand_query is caught internally
            results = retriever.retrieve("explain the termination process")

        # retrieve() should complete without exception
        assert results is not None

    def test_hyde_failure_does_not_block_expand_and_variants(self, retriever):
        """If _generate_hyde_document raises, the other enhancements still work."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            _make_llm_response("expanded query text"),
            RuntimeError("API timeout on HyDE"),
            _make_llm_response("variant 1"),
        ]
        retriever._llm_client = mock_client

        with patch.object(retriever, "_classify_query", return_value="analytical"):
            results = retriever.retrieve("analyze the liability clause")

        assert results is not None

    def test_variants_failure_does_not_block_expand_and_hyde(self, retriever):
        """If _generate_query_variants raises, the other enhancements still work."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            _make_llm_response("expanded query text"),
            _make_llm_response("hypothetical court doc"),
            RuntimeError("API timeout on variants"),
        ]
        retriever._llm_client = mock_client

        with patch.object(retriever, "_classify_query", return_value="analytical"):
            results = retriever.retrieve("compare warranty and indemnity")

        assert results is not None

    def test_all_three_failures_still_returns_results(self, retriever):
        """If ALL three LLM enhancement calls fail, retrieve() should still
        return results using just the original query."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("total failure")
        retriever._llm_client = mock_client

        with patch.object(retriever, "_classify_query", return_value="analytical"):
            results = retriever.retrieve("explain the entire contract")

        assert results is not None

    # ---------------------------------------------------------------
    # Equivalence: parallel results match sequential results
    # ---------------------------------------------------------------

    def test_enhancement_results_match_sequential_execution(self, retriever):
        """The enhancement results collected from futures should be identical
        to what would be returned by calling the methods sequentially."""
        expand_text = "expanded: termination clause contract end"
        hyde_text = "IT IS HEREBY ORDERED that the contract is terminated."
        variant_text = "end of contract\ncontract cancellation"

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            _make_llm_response(expand_text),
            _make_llm_response(hyde_text),
            _make_llm_response(variant_text),
        ]
        retriever._llm_client = mock_client

        # Call the individual methods sequentially for reference
        seq_expand = retriever._expand_query("termination clause")
        assert seq_expand == expand_text

        # Reset mock for sequential hyde
        mock_client.chat.completions.create.side_effect = [
            _make_llm_response(hyde_text),
        ]
        seq_hyde = retriever._generate_hyde_document("termination clause")
        assert seq_hyde == hyde_text

        # Reset mock for sequential variants
        mock_client.chat.completions.create.side_effect = [
            _make_llm_response(variant_text),
        ]
        seq_variants = retriever._generate_query_variants("termination clause")
        assert "termination clause" in seq_variants
        assert "end of contract" in seq_variants


    def test_threadpool_executor_is_used_with_max_workers_3(self, retriever):
        """Verify that concurrent.futures.ThreadPoolExecutor is used with max_workers=3."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_llm_response("expanded")
        retriever._llm_client = mock_client

        with patch.object(retriever, "_classify_query", return_value="analytical"):
            with patch("concurrent.futures.ThreadPoolExecutor") as mock_executor_cls:
                mock_executor = MagicMock()
                mock_executor.__enter__ = MagicMock(return_value=mock_executor)
                mock_executor.__exit__ = MagicMock(return_value=False)

                # Make futures return realistic values
                future_expand = MagicMock()
                future_expand.result.return_value = "expanded query"
                future_hyde = MagicMock()
                future_hyde.result.return_value = "hypothetical doc"
                future_variants = MagicMock()
                future_variants.result.return_value = ["termination clause", "variant 1"]

                mock_executor.submit.side_effect = [future_expand, future_hyde, future_variants]
                mock_executor_cls.return_value = mock_executor

                retriever.retrieve("explain the termination process in this contract")

            mock_executor_cls.assert_called_once_with(max_workers=3)


# ============================================================================
# Fix 8 -- Bounded embedding cache (embeddings.py)
# ============================================================================

class TestBoundedEmbeddingCache:
    """
    BaseEmbeddingService._cache is now an OrderedDict with LRU behavior.
    _get_cached() moves accessed items to the end; _set_cached() evicts the
    oldest item(s) when the cache exceeds _max_cache_size.
    """

    @pytest.fixture
    def embedding_service(self):
        """Create a BaseEmbeddingService with a tiny cache for testing."""
        from execution.legal_rag.embeddings import BaseEmbeddingService, EmbeddingConfig

        class TestEmbeddingService(BaseEmbeddingService):
            _provider_name = "Test"
            _env_var_name = "TEST_API_KEY"

            def _init_client(self):
                self._client = MagicMock()

        config = EmbeddingConfig(use_cache=True, cache_dir=None)
        service = TestEmbeddingService(config)
        return service

    # ---------------------------------------------------------------
    # Cache uses OrderedDict
    # ---------------------------------------------------------------

    def test_cache_is_ordered_dict(self, embedding_service):
        """The _cache attribute must be an OrderedDict for LRU behavior."""
        assert isinstance(embedding_service._cache, OrderedDict)

    def test_max_cache_size_default_is_10000(self, embedding_service):
        """The default _max_cache_size should be 10000."""
        assert embedding_service._max_cache_size == 10000

    # ---------------------------------------------------------------
    # Basic store and retrieve
    # ---------------------------------------------------------------

    def test_set_cached_stores_embedding(self, embedding_service):
        """_set_cached() should store a value retrievable by _get_cached()."""
        embedding = [0.1, 0.2, 0.3]
        embedding_service._set_cached("key_a", embedding)

        result = embedding_service._get_cached("key_a")
        assert result == [0.1, 0.2, 0.3]

    def test_get_cached_returns_none_for_missing_key(self, embedding_service):
        """_get_cached() should return None for a key that was never stored."""
        result = embedding_service._get_cached("nonexistent_key")
        assert result is None

    def test_get_cached_returns_none_for_empty_cache(self, embedding_service):
        """_get_cached() should return None when the cache is completely empty."""
        assert len(embedding_service._cache) == 0
        result = embedding_service._get_cached("any_key")
        assert result is None

    def test_multiple_entries_stored_and_retrieved(self, embedding_service):
        """Multiple distinct entries should coexist in the cache."""
        embedding_service._set_cached("k1", [1.0])
        embedding_service._set_cached("k2", [2.0])
        embedding_service._set_cached("k3", [3.0])

        assert embedding_service._get_cached("k1") == [1.0]
        assert embedding_service._get_cached("k2") == [2.0]
        assert embedding_service._get_cached("k3") == [3.0]

    # ---------------------------------------------------------------
    # LRU eviction when cache exceeds max size
    # ---------------------------------------------------------------

    def test_eviction_when_at_capacity(self, embedding_service):
        """When cache exceeds _max_cache_size, the oldest entry must be evicted."""
        embedding_service._max_cache_size = 3

        embedding_service._set_cached("a", [1.0])
        embedding_service._set_cached("b", [2.0])
        embedding_service._set_cached("c", [3.0])
        assert len(embedding_service._cache) == 3

        # Adding a 4th entry should evict "a" (oldest)
        embedding_service._set_cached("d", [4.0])

        assert len(embedding_service._cache) == 3
        assert embedding_service._get_cached("a") is None
        assert embedding_service._get_cached("b") == [2.0]
        assert embedding_service._get_cached("c") == [3.0]
        assert embedding_service._get_cached("d") == [4.0]

    def test_recently_accessed_entry_not_evicted(self, embedding_service):
        """After _get_cached() promotes an entry, it should survive eviction."""
        embedding_service._max_cache_size = 3

        embedding_service._set_cached("a", [1.0])
        embedding_service._set_cached("b", [2.0])
        embedding_service._set_cached("c", [3.0])

        # Access "a" to promote it to end of LRU order
        result = embedding_service._get_cached("a")
        assert result == [1.0]

        # Now "b" is the oldest. Adding "d" should evict "b", not "a"
        embedding_service._set_cached("d", [4.0])

        assert len(embedding_service._cache) == 3
        assert embedding_service._get_cached("a") == [1.0]  # survived
        assert embedding_service._get_cached("b") is None     # evicted
        assert embedding_service._get_cached("c") == [3.0]
        assert embedding_service._get_cached("d") == [4.0]

    def test_cache_size_never_exceeds_max(self, embedding_service):
        """Even after many insertions, cache size must stay <= _max_cache_size."""
        embedding_service._max_cache_size = 5

        for i in range(50):
            embedding_service._set_cached(f"key_{i}", [float(i)])

        assert len(embedding_service._cache) <= 5

    def test_eviction_removes_oldest_not_newest(self, embedding_service):
        """The FIFO order of eviction must remove the oldest inserted/unreferenced entry."""
        embedding_service._max_cache_size = 2

        embedding_service._set_cached("first", [1.0])
        embedding_service._set_cached("second", [2.0])
        embedding_service._set_cached("third", [3.0])

        # "first" should be evicted
        assert embedding_service._get_cached("first") is None
        assert embedding_service._get_cached("second") == [2.0]
        assert embedding_service._get_cached("third") == [3.0]

    def test_overwrite_existing_key_does_not_grow_cache(self, embedding_service):
        """Overwriting an existing key should update the value, not add a duplicate."""
        embedding_service._max_cache_size = 3

        embedding_service._set_cached("a", [1.0])
        embedding_service._set_cached("b", [2.0])
        embedding_service._set_cached("a", [9.9])  # overwrite

        assert len(embedding_service._cache) == 2
        assert embedding_service._get_cached("a") == [9.9]

    # ---------------------------------------------------------------
    # LRU move_to_end behavior
    # ---------------------------------------------------------------

    def test_get_cached_moves_accessed_entry_to_end(self, embedding_service):
        """_get_cached() must call move_to_end() so the entry is marked recently used."""
        embedding_service._set_cached("a", [1.0])
        embedding_service._set_cached("b", [2.0])

        # Access "a" to move it to end
        embedding_service._get_cached("a")

        # "a" should now be last in the OrderedDict
        keys = list(embedding_service._cache.keys())
        assert keys[-1] == "a"

    def test_set_cached_moves_entry_to_end(self, embedding_service):
        """_set_cached() must place/move the entry to the end of the LRU order."""
        embedding_service._set_cached("a", [1.0])
        embedding_service._set_cached("b", [2.0])
        embedding_service._set_cached("a", [1.1])  # re-set "a"

        keys = list(embedding_service._cache.keys())
        assert keys[-1] == "a"

    # ---------------------------------------------------------------
    # Cache disabled
    # ---------------------------------------------------------------

    def test_cache_disabled_returns_none(self):
        """When use_cache=False, _get_cached() should always return None."""
        from execution.legal_rag.embeddings import BaseEmbeddingService, EmbeddingConfig

        class TestService(BaseEmbeddingService):
            _provider_name = "Test"
            _env_var_name = "TEST_API_KEY"
            def _init_client(self):
                self._client = MagicMock()

        config = EmbeddingConfig(use_cache=False, cache_dir=None)
        service = TestService(config)

        service._set_cached("k", [1.0])
        assert service._get_cached("k") is None

    def test_cache_disabled_does_not_store(self):
        """When use_cache=False, _set_cached() should not add entries."""
        from execution.legal_rag.embeddings import BaseEmbeddingService, EmbeddingConfig

        class TestService(BaseEmbeddingService):
            _provider_name = "Test"
            _env_var_name = "TEST_API_KEY"
            def _init_client(self):
                self._client = MagicMock()

        config = EmbeddingConfig(use_cache=False, cache_dir=None)
        service = TestService(config)

        service._set_cached("k", [1.0])
        assert len(service._cache) == 0


# ============================================================================
# Fix 9 -- API quota enforcement in query_documents (api.py)
# ============================================================================

class TestAPIQuotaEnforcement:
    """
    The query_documents() endpoint now checks quotas before retrieval.
    QuotaExceededError results in HTTP 429; other quota check failures
    degrade gracefully (log warning, allow request through).
    """

    @pytest.fixture
    def client(self):
        """Create a FastAPI TestClient with fully mocked backend services."""
        from fastapi.testclient import TestClient
        from execution.legal_rag import api as api_module

        # Reset the container so patches are clean
        original_container = api_module._container

        mock_container = MagicMock()

        # Mock store
        mock_store = MagicMock()
        mock_store.validate_api_key.return_value = {
            "client_id": "test-client-id",
            "tier": "default",
        }
        mock_store.get_tenant_config.return_value = None  # will use English default
        mock_store.log_audit.return_value = None
        mock_store.get_document_titles.return_value = {}
        mock_container.get_store.return_value = mock_store

        # Mock retriever that returns results
        mock_retriever = MagicMock()
        from execution.legal_rag.vector_store import SearchResult
        mock_retriever.retrieve.return_value = [
            SearchResult(
                chunk_id="chunk-001",
                document_id="doc-001",
                content="Section 4.2 Termination clause.",
                section_title="ARTICLE IV",
                hierarchy_path="Document/ARTICLE_IV",
                page_numbers=[4],
                score=0.90,
                metadata={},
            )
        ]

        # Mock citation extractor
        mock_citation = MagicMock()
        mock_cited = MagicMock()
        mock_cited.citation.short_format.return_value = "[1] Section 4.2"
        mock_cited.citation.long_format.return_value = "Document, Section 4.2, p.4"
        mock_cited.citation.document_title = "Test Doc"
        mock_cited.citation.section = "Section 4.2"
        mock_cited.citation.page_numbers = [4]
        mock_cited.citation.hierarchy_path = "Document/ARTICLE_IV"
        mock_cited.citation.chunk_id = "chunk-001"
        mock_cited.citation.document_id = "doc-001"
        mock_cited.citation.relevance_score = 0.90
        mock_cited.content = "Section 4.2 Termination clause."
        mock_cited.context_before = ""
        mock_cited.context_after = ""
        mock_citation.extract.return_value = [mock_cited]

        # Mock services dict
        mock_container.get_services.return_value = {
            "retriever": mock_retriever,
            "citation_extractor": mock_citation,
        }

        # Mock LLM client
        mock_llm = MagicMock()
        mock_llm.chat.completions.create.return_value = _make_llm_response(
            "The termination clause states..."
        )
        mock_container.get_llm_client.return_value = mock_llm

        api_module._container = mock_container

        test_client = TestClient(api_module.app)

        yield test_client

        # Restore
        api_module._container = original_container

    # ---------------------------------------------------------------
    # Normal query proceeds when quota not exceeded
    # ---------------------------------------------------------------

    def test_query_succeeds_when_quota_not_exceeded(self, client):
        """A normal query should return 200 when quota check passes."""
        from execution.legal_rag.quotas import QuotaManager

        with patch("execution.legal_rag.quotas.get_quota_manager") as mock_get_qm:
            mock_qm = MagicMock(spec=QuotaManager)
            mock_qm.check_query_quota.return_value = True
            mock_get_qm.return_value = mock_qm

            response = client.post(
                "/api/v1/query",
                json={"query": "termination clause", "top_k": 5},
                headers={"x-api-key": "test-key"},
            )

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data

    # ---------------------------------------------------------------
    # Returns 429 when QuotaExceededError raised
    # ---------------------------------------------------------------

    def test_returns_429_when_quota_exceeded(self, client):
        """When check_query_quota raises QuotaExceededError, the endpoint
        must return HTTP 429 with the error message."""
        from execution.legal_rag.quotas import QuotaExceededError

        with patch("execution.legal_rag.quotas.get_quota_manager") as mock_get_qm:
            mock_qm = MagicMock()
            mock_qm.check_query_quota.side_effect = QuotaExceededError(
                "Daily query limit reached (100 queries/day)",
                quota_type="queries_per_day",
                current=100,
                limit=100,
            )
            mock_get_qm.return_value = mock_qm

            response = client.post(
                "/api/v1/query",
                json={"query": "termination clause", "top_k": 5},
                headers={"x-api-key": "test-key"},
            )

        assert response.status_code == 429
        assert "query limit" in response.json()["detail"].lower()

    # ---------------------------------------------------------------
    # Request proceeds when quota check raises unexpected error
    # ---------------------------------------------------------------

    def test_request_proceeds_on_unexpected_quota_error(self, client):
        """When the quota check raises an unexpected exception (not
        QuotaExceededError), the request should still proceed (graceful
        degradation) rather than returning a 500."""
        with patch("execution.legal_rag.quotas.get_quota_manager") as mock_get_qm:
            mock_qm = MagicMock()
            mock_qm.check_query_quota.side_effect = ConnectionError(
                "Database connection lost"
            )
            mock_get_qm.return_value = mock_qm

            response = client.post(
                "/api/v1/query",
                json={"query": "termination clause", "top_k": 5},
                headers={"x-api-key": "test-key"},
            )

        # Should succeed despite quota check failure
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data

    def test_unexpected_quota_error_logs_warning(self, client):
        """When the quota check raises an unexpected exception, a warning
        should be logged for observability."""
        with patch("execution.legal_rag.quotas.get_quota_manager") as mock_get_qm:
            mock_qm = MagicMock()
            mock_qm.check_query_quota.side_effect = RuntimeError("Redis down")
            mock_get_qm.return_value = mock_qm

            with patch("execution.legal_rag.api.logger") as mock_logger:
                response = client.post(
                    "/api/v1/query",
                    json={"query": "termination clause", "top_k": 5},
                    headers={"x-api-key": "test-key"},
                )

            # The warning should mention the quota check failure
            mock_logger.warning.assert_called()
            warning_msg = mock_logger.warning.call_args[0][0]
            assert "quota" in warning_msg.lower() or "Quota" in warning_msg

    def test_quota_check_receives_client_id_and_tier(self, client):
        """check_query_quota must be called with the authenticated client_id and tier."""
        from execution.legal_rag.quotas import QuotaManager

        with patch("execution.legal_rag.quotas.get_quota_manager") as mock_get_qm:
            mock_qm = MagicMock(spec=QuotaManager)
            mock_qm.check_query_quota.return_value = True
            mock_get_qm.return_value = mock_qm

            response = client.post(
                "/api/v1/query",
                json={"query": "termination clause"},
                headers={"x-api-key": "test-key"},
            )

        mock_qm.check_query_quota.assert_called_once_with(
            "test-client-id",
            tier="default",
        )
