"""
Tests for Batch 2 of codebase review fixes applied to the Legal RAG System.

Fix 4: Paragraph extraction routing in retrieve()
    retrieve() now checks _extract_paragraph_reference(query) at the start.
    If a paragraph number is found AND document_id is provided, it calls
    store.search_by_paragraph() and returns results immediately without
    running the full retrieval pipeline.

Fix 5: Cache double-embed fix in QueryResultCache
    QueryResultCache.get() and .set() now accept an optional query_embedding
    parameter.  In retrieve(), the query embedding is computed ONCE with
    self.embeddings.embed_query(query) and then reused for cache lookups,
    cache stores, and the first vector search iteration, avoiding redundant
    embed_query calls.

Fix 6: Lazy DB connection checking in VectorStore._get_connection()
    _get_connection() no longer executes SELECT 1 to verify the connection.
    It only checks self._conn.closed -- if True, reconnects; if False,
    returns the existing connection.

All external API calls are mocked -- no database, LLM, or embedding service
is required.
"""

from unittest.mock import patch, MagicMock, call
from dataclasses import replace as _replace

import pytest


# ============================================================================
# Helpers (same patterns as test_review_fixes.py)
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


def _make_search_result(chunk_id, score, metadata=None, content="Sample legal text.",
                        document_id="doc-001", paragraph_start=None,
                        paragraph_end=None, original_paragraph_numbers=None):
    """Create a SearchResult with the given fields."""
    from execution.legal_rag.vector_store import SearchResult

    return SearchResult(
        chunk_id=chunk_id,
        document_id=document_id,
        content=content,
        section_title="Section A",
        hierarchy_path="Document/Section_A",
        page_numbers=[1],
        score=score,
        metadata=metadata or {},
        paragraph_start=paragraph_start,
        paragraph_end=paragraph_end,
        original_paragraph_numbers=original_paragraph_numbers,
    )


# ============================================================================
# Fix 4 -- Paragraph extraction routing in retrieve()
# ============================================================================

class TestParagraphExtractionRouting:
    """
    retrieve() now calls _extract_paragraph_reference(query) at the start.
    When a paragraph number is detected AND a document_id is provided,
    it calls store.search_by_paragraph() and returns immediately if
    results are found, completely bypassing the full retrieval pipeline.
    """

    @pytest.fixture
    def retriever(self, mock_vector_store, mock_embedding_service):
        return _build_retriever(mock_vector_store, mock_embedding_service)

    # -- _extract_paragraph_reference unit tests ----------------------------

    def test_extract_paragraph_reference_with_paragraph_keyword(self, retriever):
        """'paragraph 5' should extract paragraph number 5."""
        result = retriever._extract_paragraph_reference("paragraph 5")
        assert result == 5

    def test_extract_paragraph_reference_with_para_abbreviation(self, retriever):
        """'para 12' should extract paragraph number 12."""
        result = retriever._extract_paragraph_reference("para 12")
        assert result == 12

    def test_extract_paragraph_reference_with_para_dot(self, retriever):
        """'para. 7' should extract paragraph number 7."""
        result = retriever._extract_paragraph_reference("para. 7")
        assert result == 7

    def test_extract_paragraph_reference_with_pilcrow(self, retriever):
        """The pilcrow symbol followed by a number should be detected."""
        result = retriever._extract_paragraph_reference("What does \u00b628 say?")
        assert result == 28

    def test_extract_paragraph_reference_no_match_returns_none(self, retriever):
        """Queries without paragraph references should return None."""
        result = retriever._extract_paragraph_reference("What is the contract?")
        assert result is None

    def test_extract_paragraph_reference_with_sentence_context(self, retriever):
        """Paragraph reference embedded in a longer query should still be detected."""
        result = retriever._extract_paragraph_reference(
            "Explain the meaning of paragraph 15 regarding indemnity"
        )
        assert result == 15

    def test_extract_paragraph_reference_no_number_after_keyword(self, retriever):
        """'paragraph' without a number should not match."""
        result = retriever._extract_paragraph_reference("the paragraph is unclear")
        assert result is None

    # -- retrieve() paragraph routing integration tests ---------------------

    def test_retrieve_with_paragraph_ref_and_document_id_calls_search_by_paragraph(
        self, mock_vector_store, mock_embedding_service
    ):
        """
        When the query contains a paragraph reference AND document_id is provided,
        retrieve() must call store.search_by_paragraph and return its results
        without running the full pipeline.
        """
        retriever = _build_retriever(mock_vector_store, mock_embedding_service)

        para_results = [
            _make_search_result("chunk-para-5", score=1.0, paragraph_start=5, paragraph_end=5),
        ]
        mock_vector_store.search_by_paragraph = MagicMock(return_value=para_results)

        results = retriever.retrieve(
            query="paragraph 5",
            client_id="client-001",
            document_id="doc-001",
        )

        # search_by_paragraph must have been called with correct args
        mock_vector_store.search_by_paragraph.assert_called_once_with(
            document_id="doc-001",
            paragraph_number=5,
            client_id="client-001",
        )
        # The returned results should be the paragraph search results
        assert results == para_results

    def test_retrieve_with_paragraph_ref_and_document_id_skips_full_pipeline(
        self, mock_vector_store, mock_embedding_service
    ):
        """
        When paragraph search succeeds, the embedding service should NOT be
        called (no embed_query), proving the full pipeline was skipped.
        """
        retriever = _build_retriever(mock_vector_store, mock_embedding_service)

        para_results = [
            _make_search_result("chunk-para-10", score=1.0, paragraph_start=10, paragraph_end=10),
        ]
        mock_vector_store.search_by_paragraph = MagicMock(return_value=para_results)

        # Track embed_query calls
        original_call_count = mock_embedding_service._call_count

        retriever.retrieve(
            query="paragraph 10",
            client_id="client-001",
            document_id="doc-001",
        )

        # embed_query must NOT have been called
        assert mock_embedding_service._call_count == original_call_count

    def test_retrieve_with_paragraph_ref_but_no_document_id_falls_through(
        self, mock_vector_store, mock_embedding_service
    ):
        """
        When a paragraph reference is detected but no document_id is provided,
        retrieve() must fall through to the normal retrieval pipeline because
        search_by_paragraph requires a document_id.
        """
        retriever = _build_retriever(mock_vector_store, mock_embedding_service)

        mock_vector_store.search_by_paragraph = MagicMock()

        results = retriever.retrieve(
            query="paragraph 5",
            client_id="client-001",
            document_id=None,
        )

        # search_by_paragraph must NOT have been called
        mock_vector_store.search_by_paragraph.assert_not_called()

        # The normal pipeline ran (returned whatever the mock store yields)
        assert isinstance(results, list)

    def test_retrieve_with_no_paragraph_ref_falls_through(
        self, mock_vector_store, mock_embedding_service
    ):
        """
        When the query has no paragraph reference, retrieve() must NOT call
        search_by_paragraph even if document_id is provided.
        """
        retriever = _build_retriever(mock_vector_store, mock_embedding_service)

        mock_vector_store.search_by_paragraph = MagicMock()

        results = retriever.retrieve(
            query="What is the termination clause?",
            client_id="client-001",
            document_id="doc-001",
        )

        mock_vector_store.search_by_paragraph.assert_not_called()
        assert isinstance(results, list)

    def test_retrieve_paragraph_search_empty_falls_through_to_normal_pipeline(
        self, mock_vector_store, mock_embedding_service
    ):
        """
        When search_by_paragraph returns an empty list (paragraph not found),
        retrieve() must fall through to the normal retrieval pipeline.
        """
        retriever = _build_retriever(mock_vector_store, mock_embedding_service)

        mock_vector_store.search_by_paragraph = MagicMock(return_value=[])

        original_call_count = mock_embedding_service._call_count

        results = retriever.retrieve(
            query="paragraph 999",
            client_id="client-001",
            document_id="doc-001",
        )

        # search_by_paragraph was called (paragraph detected + document_id present)
        mock_vector_store.search_by_paragraph.assert_called_once()

        # But since it returned empty, the full pipeline ran
        # (embed_query was called at least once for the normal path)
        assert mock_embedding_service._call_count > original_call_count

    def test_retrieve_paragraph_search_respects_top_k(
        self, mock_vector_store, mock_embedding_service
    ):
        """
        When paragraph search returns results, the output must be truncated
        to the requested top_k.
        """
        retriever = _build_retriever(mock_vector_store, mock_embedding_service)

        # Return 5 results from paragraph search
        para_results = [
            _make_search_result(f"chunk-para-{i}", score=1.0, paragraph_start=5, paragraph_end=5)
            for i in range(5)
        ]
        mock_vector_store.search_by_paragraph = MagicMock(return_value=para_results)

        results = retriever.retrieve(
            query="paragraph 5",
            client_id="client-001",
            document_id="doc-001",
            top_k=2,
        )

        assert len(results) <= 2


# ============================================================================
# Fix 5 -- Cache double-embed fix in QueryResultCache
# ============================================================================

class TestCacheDoubleEmbedFix:
    """
    QueryResultCache.get() and .set() now accept an optional query_embedding
    parameter to avoid redundant embed_query() calls.  In retrieve(), the
    query embedding is computed ONCE and reused for cache operations and the
    first search iteration.
    """

    @pytest.fixture
    def cache(self, mock_embedding_service):
        from execution.legal_rag.retriever import QueryResultCache
        return QueryResultCache(
            embedding_service=mock_embedding_service,
            similarity_threshold=0.92,
            max_size=100,
            ttl_seconds=3600,
        )

    # -- QueryResultCache.get() signature tests -----------------------------

    def test_cache_get_accepts_query_embedding_kwarg(self, cache, mock_embedding_service):
        """QueryResultCache.get() must accept query_embedding as a keyword argument."""
        embedding = mock_embedding_service.embed_query("test query")

        # Should not raise TypeError
        result = cache.get(
            "test query",
            client_id=None,
            document_id=None,
            query_embedding=embedding,
        )
        assert result is None  # empty cache

    def test_cache_get_without_query_embedding_still_works(self, cache):
        """QueryResultCache.get() without query_embedding must still work (backwards compat)."""
        result = cache.get("test query")
        assert result is None

    # -- QueryResultCache.set() signature tests -----------------------------

    def test_cache_set_accepts_query_embedding_kwarg(self, cache, mock_embedding_service):
        """QueryResultCache.set() must accept query_embedding as a keyword argument."""
        embedding = mock_embedding_service.embed_query("test query")
        results = [_make_search_result("c1", score=0.9)]

        # Should not raise TypeError
        cache.set(
            "test query",
            results,
            client_id=None,
            document_id=None,
            query_embedding=embedding,
        )
        assert cache.size == 1

    def test_cache_set_without_query_embedding_still_works(self, cache):
        """QueryResultCache.set() without query_embedding must still work."""
        results = [_make_search_result("c1", score=0.9)]
        cache.set("test query", results)
        assert cache.size == 1

    # -- Cache stores and retrieves with pre-computed embedding -------------

    def test_cache_roundtrip_with_precomputed_embedding(self, cache, mock_embedding_service):
        """
        When both get and set are called with the same pre-computed embedding,
        the cache should store and retrieve results correctly.
        """
        embedding = mock_embedding_service.embed_query("termination clause")
        results = [_make_search_result("c1", score=0.95)]

        # Store with pre-computed embedding
        cache.set(
            "termination clause",
            results,
            client_id="client-1",
            document_id="doc-1",
            query_embedding=embedding,
        )

        # Retrieve with the same pre-computed embedding
        cached = cache.get(
            "termination clause",
            client_id="client-1",
            document_id="doc-1",
            query_embedding=embedding,
        )

        assert cached is not None
        assert len(cached) == 1
        assert cached[0].chunk_id == "c1"

    def test_cache_get_with_embedding_avoids_embed_query(self, cache, mock_embedding_service):
        """
        When query_embedding is provided to get(), embed_query should NOT be
        called inside the cache (saving an API call).
        """
        embedding = mock_embedding_service.embed_query("test query")
        initial_count = mock_embedding_service._call_count

        cache.get(
            "test query",
            client_id=None,
            document_id=None,
            query_embedding=embedding,
        )

        # embed_query should not have been called again inside cache.get()
        assert mock_embedding_service._call_count == initial_count

    def test_cache_set_with_embedding_avoids_embed_query(self, cache, mock_embedding_service):
        """
        When query_embedding is provided to set(), embed_query should NOT be
        called inside the cache.
        """
        embedding = mock_embedding_service.embed_query("test query")
        initial_count = mock_embedding_service._call_count

        results = [_make_search_result("c1", score=0.9)]
        cache.set(
            "test query",
            results,
            client_id=None,
            document_id=None,
            query_embedding=embedding,
        )

        assert mock_embedding_service._call_count == initial_count

    # -- retrieve() embedding reuse tests -----------------------------------

    def test_retrieve_embeds_original_query_only_once(
        self, mock_vector_store, mock_embedding_service
    ):
        """
        In a simple retrieval (no query expansion/HyDE/multi-query), the
        original query should be embedded exactly ONCE, not twice.  The
        pre-computed embedding is reused for cache and vector search.
        """
        retriever = _build_retriever(
            mock_vector_store,
            mock_embedding_service,
            use_query_expansion=False,
            use_hyde=False,
            use_multi_query=False,
        )

        initial_count = mock_embedding_service._call_count

        retriever.retrieve(
            query="termination clause",
            client_id="client-001",
            use_cache=True,
        )

        # embed_query should be called exactly ONCE for the original query
        # (not once for cache.get + once for search + once for cache.set)
        calls_made = mock_embedding_service._call_count - initial_count
        assert calls_made == 1, (
            f"Expected embed_query to be called once, but it was called {calls_made} times"
        )

    def test_retrieve_reuses_embedding_for_first_search_query(
        self, mock_vector_store, mock_embedding_service
    ):
        """
        When iterating over search_queries, the first query (if it matches
        the original query) should reuse the pre-computed embedding instead
        of calling embed_query again.
        """
        retriever = _build_retriever(
            mock_vector_store,
            mock_embedding_service,
            use_query_expansion=False,
            use_hyde=False,
            use_multi_query=False,
        )

        # Wrap embed_query to track calls
        original_embed = mock_embedding_service.embed_query
        embed_calls = []

        def tracking_embed(text):
            embed_calls.append(text)
            return original_embed(text)

        mock_embedding_service.embed_query = tracking_embed

        retriever.retrieve(
            query="governing law clause",
            client_id="client-001",
            use_cache=False,
        )

        # The original query should only appear once in embed_calls
        original_query_count = embed_calls.count("governing law clause")
        assert original_query_count == 1, (
            f"Original query embedded {original_query_count} times, expected 1. "
            f"All calls: {embed_calls}"
        )


# ============================================================================
# Fix 6 -- Lazy DB connection checking in VectorStore._get_connection()
# ============================================================================

class TestLazyDBConnectionChecking:
    """
    _get_connection() no longer executes SELECT 1 to verify the connection.
    It only checks self._conn.closed -- if True, reconnects; if False,
    returns the existing connection as-is.
    """

    @pytest.fixture
    def store_with_open_conn(self):
        """
        Build a VectorStore in single-connection mode with a mock connection
        whose .closed property returns False (connection is open).
        """
        from execution.legal_rag.vector_store import VectorStore, VectorStoreConfig

        store = VectorStore.__new__(VectorStore)
        store.config = VectorStoreConfig(use_pooling=False)
        store._pool = None
        store._current_tenant = None
        store._connection_string = "postgresql://fake/test"

        mock_conn = MagicMock()
        mock_conn.closed = False
        store._conn = mock_conn

        return store, mock_conn

    @pytest.fixture
    def store_with_closed_conn(self):
        """
        Build a VectorStore in single-connection mode with a mock connection
        whose .closed property returns True (connection is dead).
        """
        from execution.legal_rag.vector_store import VectorStore, VectorStoreConfig

        store = VectorStore.__new__(VectorStore)
        store.config = VectorStoreConfig(use_pooling=False)
        store._pool = None
        store._current_tenant = None
        store._connection_string = "postgresql://fake/test"

        mock_conn = MagicMock()
        mock_conn.closed = True
        store._conn = mock_conn

        return store, mock_conn

    @pytest.fixture
    def store_with_no_conn(self):
        """
        Build a VectorStore in single-connection mode with _conn set to None
        (no connection yet established).
        """
        from execution.legal_rag.vector_store import VectorStore, VectorStoreConfig

        store = VectorStore.__new__(VectorStore)
        store.config = VectorStoreConfig(use_pooling=False)
        store._pool = None
        store._current_tenant = None
        store._connection_string = "postgresql://fake/test"
        store._conn = None

        return store

    def test_get_connection_returns_existing_conn_when_open(self, store_with_open_conn):
        """
        When self._conn is not None and self._conn.closed is False,
        _get_connection() must return the existing connection directly.
        """
        store, mock_conn = store_with_open_conn

        result = store._get_connection()

        assert result is mock_conn

    def test_get_connection_does_not_execute_select_1_when_open(self, store_with_open_conn):
        """
        When the connection is open, _get_connection() must NOT execute
        any SQL (no SELECT 1 ping).
        """
        store, mock_conn = store_with_open_conn

        store._get_connection()

        # cursor() should NOT be called -- no SQL executed
        mock_conn.cursor.assert_not_called()

    def test_get_connection_reconnects_when_conn_closed(self, store_with_closed_conn):
        """
        When self._conn.closed is True, _get_connection() must call
        self.connect() to re-establish the connection.
        """
        store, mock_conn = store_with_closed_conn

        with patch.object(store, "connect") as mock_connect:
            # After connect(), store._conn should be a fresh connection
            new_conn = MagicMock()
            new_conn.closed = False

            def fake_connect():
                store._conn = new_conn

            mock_connect.side_effect = fake_connect

            result = store._get_connection()

            mock_connect.assert_called_once()
            assert result is new_conn

    def test_get_connection_returns_none_when_no_conn_and_no_pool(self, store_with_no_conn):
        """
        When self._conn is None and self._pool is None, _get_connection()
        returns None (the current value of self._conn).  The caller
        (_ensure_connection) is responsible for calling connect() first.
        """
        store = store_with_no_conn

        result = store._get_connection()

        # _conn is None, _pool is None, so returns None
        assert result is None

    def test_get_connection_checks_closed_attribute_not_sql(self, store_with_open_conn):
        """
        Verify that _get_connection() accesses the .closed property on the
        connection object (Python-level check) rather than executing SQL.
        This is the essence of the lazy connection checking fix.
        """
        store, mock_conn = store_with_open_conn

        # Make .closed a property we can monitor via the mock
        # MagicMock already tracks attribute access, but let's be explicit
        type(mock_conn).closed = property(lambda self: False)

        result = store._get_connection()

        assert result is mock_conn
        # No cursor() or execute() calls should have been made
        mock_conn.cursor.assert_not_called()

    def test_get_connection_with_pool_uses_pool_getconn(self):
        """
        When _pool is set, _get_connection() should use pool.getconn()
        instead of the single connection path.
        """
        from execution.legal_rag.vector_store import VectorStore, VectorStoreConfig

        store = VectorStore.__new__(VectorStore)
        store.config = VectorStoreConfig(use_pooling=True)
        store._current_tenant = None
        store._connection_string = "postgresql://fake/test"

        mock_pool = MagicMock()
        pool_conn = MagicMock()
        mock_pool.getconn.return_value = pool_conn
        store._pool = mock_pool
        store._conn = None

        result = store._get_connection()

        mock_pool.getconn.assert_called_once()
        assert result is pool_conn

    def test_get_connection_single_mode_no_reconnect_when_open(self, store_with_open_conn):
        """
        In single-connection mode with an open connection, connect() must
        NOT be called -- the existing connection is returned as-is.
        """
        store, mock_conn = store_with_open_conn

        with patch.object(store, "connect") as mock_connect:
            store._get_connection()
            mock_connect.assert_not_called()
