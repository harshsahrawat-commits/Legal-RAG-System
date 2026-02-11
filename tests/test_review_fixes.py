"""
Tests for three targeted bug fixes applied to the Legal RAG System.

Fix 1: LLM None guard in retriever.py
    _expand_query(), _generate_hyde_document(), and _generate_query_variants()
    now handle None content from the OpenAI-compatible LLM API without raising
    AttributeError.

Fix 2: Smart reranking uses original_score from metadata
    _should_skip_reranking() reads metadata["original_score"] instead of the
    raw RRF score, which maxes out at ~0.01 and would never exceed the 0.85
    threshold.

Fix 3: Greek FTS index in initialize_schema
    initialize_schema() now creates both the English and Greek full-text search
    GIN indexes in a single migration step.

All external API calls are mocked -- no database, LLM, or embedding service
is required.
"""

from unittest.mock import patch, MagicMock
from dataclasses import replace as _replace

import pytest


# ============================================================================
# Helpers
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
# Fix 1 -- LLM None guard in retriever.py
# ============================================================================

class TestLLMNoneGuard:
    """
    When the LLM API returns response.choices[0].message.content == None,
    the retriever methods must gracefully fall back to the original query
    instead of raising AttributeError on NoneType.strip().
    """

    @pytest.fixture
    def retriever(self, mock_vector_store, mock_embedding_service):
        return _build_retriever(mock_vector_store, mock_embedding_service)

    # -- _expand_query ---------------------------------------------------

    def test_expand_query_with_none_content_returns_original(self, retriever):
        """_expand_query must return the original query when LLM content is None."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_llm_response(None)
        retriever._llm_client = mock_client

        result = retriever._expand_query("termination clause")

        assert result == "termination clause"

    def test_expand_query_with_none_content_does_not_raise(self, retriever):
        """_expand_query must not raise when LLM content is None."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_llm_response(None)
        retriever._llm_client = mock_client

        # Should complete without any exception
        try:
            retriever._expand_query("termination clause")
        except AttributeError:
            pytest.fail("_expand_query raised AttributeError on None content")

    def test_expand_query_with_valid_content_returns_expanded(self, retriever):
        """_expand_query should return the LLM expansion when content is present."""
        expanded_text = "termination clause, contract termination, Section 4.2"
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_llm_response(expanded_text)
        retriever._llm_client = mock_client

        result = retriever._expand_query("termination clause")

        assert result == expanded_text

    # -- _generate_hyde_document -----------------------------------------

    def test_generate_hyde_document_with_none_content_returns_original(self, retriever):
        """_generate_hyde_document must return the original query when LLM content is None."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_llm_response(None)
        retriever._llm_client = mock_client

        result = retriever._generate_hyde_document("what is the governing law?")

        assert result == "what is the governing law?"

    def test_generate_hyde_document_with_none_content_does_not_raise(self, retriever):
        """_generate_hyde_document must not raise when LLM content is None."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_llm_response(None)
        retriever._llm_client = mock_client

        try:
            retriever._generate_hyde_document("what is the governing law?")
        except AttributeError:
            pytest.fail("_generate_hyde_document raised AttributeError on None content")

    def test_generate_hyde_document_with_valid_content_returns_hyde(self, retriever):
        """_generate_hyde_document should return the hypothetical document text."""
        hyde_text = "IT IS ORDERED that this case is governed by Delaware law."
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_llm_response(hyde_text)
        retriever._llm_client = mock_client

        result = retriever._generate_hyde_document("what is the governing law?")

        assert result == hyde_text

    # -- _generate_query_variants ----------------------------------------

    def test_generate_query_variants_with_none_content_returns_original_list(self, retriever):
        """_generate_query_variants must return [original_query] when LLM content is None."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_llm_response(None)
        retriever._llm_client = mock_client

        result = retriever._generate_query_variants("contract termination")

        assert result == ["contract termination"]

    def test_generate_query_variants_with_none_content_does_not_raise(self, retriever):
        """_generate_query_variants must not raise when LLM content is None."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_llm_response(None)
        retriever._llm_client = mock_client

        try:
            retriever._generate_query_variants("contract termination")
        except AttributeError:
            pytest.fail("_generate_query_variants raised AttributeError on None content")

    def test_generate_query_variants_with_valid_content_returns_variants(self, retriever):
        """_generate_query_variants should return original + up to 3 variants."""
        variant_text = "how to end a contract\ncontract cancellation process\nearly exit from agreement"
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_llm_response(variant_text)
        retriever._llm_client = mock_client

        result = retriever._generate_query_variants("contract termination")

        # Original query is always first
        assert result[0] == "contract termination"
        # Up to 3 variants follow
        assert len(result) <= 4
        assert len(result) >= 2

    def test_generate_query_variants_with_empty_string_content_returns_original_list(self, retriever):
        """_generate_query_variants must return [original_query] when LLM content is empty string."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_llm_response("")
        retriever._llm_client = mock_client

        result = retriever._generate_query_variants("contract termination")

        # Empty string stripped yields no variants, so only original
        assert result == ["contract termination"]

    def test_expand_query_with_empty_string_content_returns_original(self, retriever):
        """_expand_query should return original query when LLM returns empty string."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_llm_response("")
        retriever._llm_client = mock_client

        result = retriever._expand_query("termination clause")

        # empty string stripped is falsy, so falls back to query
        assert result == "termination clause"


# ============================================================================
# Fix 2 -- Smart reranking uses original_score from metadata
# ============================================================================

class TestSmartRerankingOriginalScore:
    """
    After RRF, scores are ~0.01. The 0.85 threshold only makes sense
    against original cosine similarity scores preserved in
    metadata["original_score"]. _should_skip_reranking must read from
    metadata, not from the raw .score attribute.
    """

    @pytest.fixture
    def retriever(self, mock_vector_store, mock_embedding_service):
        return _build_retriever(
            mock_vector_store,
            mock_embedding_service,
            use_reranking=True,
            use_smart_reranking=True,
            smart_rerank_threshold=0.85,
            smart_rerank_gap=0.15,
        )

    def test_skip_when_original_score_high_and_gap_large(self, retriever):
        """
        RRF scores are low (0.009, 0.008) but original_score is high (0.92, 0.60).
        The method should use original_score and return True (skip reranking).
        """
        results = [
            _make_search_result("c1", score=0.009, metadata={"original_score": 0.92}),
            _make_search_result("c2", score=0.008, metadata={"original_score": 0.60}),
        ]

        assert retriever._should_skip_reranking(results) is True

    def test_no_skip_when_original_score_low(self, retriever):
        """
        RRF scores are low (0.009, 0.008) and original_score is also low (0.50, 0.48).
        Should return False (do not skip reranking).
        """
        results = [
            _make_search_result("c1", score=0.009, metadata={"original_score": 0.50}),
            _make_search_result("c2", score=0.008, metadata={"original_score": 0.48}),
        ]

        assert retriever._should_skip_reranking(results) is False

    def test_no_skip_when_gap_too_small(self, retriever):
        """
        Original scores are both high but very close -- gap is below threshold.
        Should return False (do not skip).
        """
        results = [
            _make_search_result("c1", score=0.009, metadata={"original_score": 0.90}),
            _make_search_result("c2", score=0.008, metadata={"original_score": 0.88}),
        ]

        assert retriever._should_skip_reranking(results) is False

    def test_falls_back_to_score_when_no_original_score_in_metadata(self, retriever):
        """
        If metadata lacks original_score, the method falls back to result.score.
        With RRF scores (0.009), this should NOT skip (below threshold).
        """
        results = [
            _make_search_result("c1", score=0.009, metadata={}),
            _make_search_result("c2", score=0.008, metadata={}),
        ]

        assert retriever._should_skip_reranking(results) is False

    def test_rrf_produces_original_score_in_metadata(self, retriever):
        """
        _reciprocal_rank_fusion should store the original vector score
        as metadata["original_score"] so _should_skip_reranking can use it.
        """
        results_in = [
            _make_search_result("c1", score=0.92, metadata={"level": 0}),
            _make_search_result("c2", score=0.75, metadata={"level": 0}),
        ]

        fused = retriever._reciprocal_rank_fusion(results_in, [])

        assert len(fused) == 2
        # Every fused result must carry original_score in metadata
        for r in fused:
            assert "original_score" in r.metadata
        # The original_score should match the input scores
        scores_by_id = {r.chunk_id: r.metadata["original_score"] for r in fused}
        assert scores_by_id["c1"] == pytest.approx(0.92)
        assert scores_by_id["c2"] == pytest.approx(0.75)

    def test_rrf_score_much_lower_than_original_score(self, retriever):
        """
        After RRF, the .score attribute should be much smaller than the
        original cosine similarity score stored in metadata.
        """
        results_in = [
            _make_search_result("c1", score=0.92, metadata={"level": 0}),
        ]

        fused = retriever._reciprocal_rank_fusion(results_in, [])

        assert len(fused) == 1
        # RRF score = vector_weight / (rrf_k + rank + 1) = 0.6 / (60+0+1) ~ 0.0098
        assert fused[0].score < 0.02
        assert fused[0].metadata["original_score"] == pytest.approx(0.92)

    def test_skip_reranking_disabled_returns_false(self, mock_vector_store, mock_embedding_service):
        """When use_smart_reranking=False, _should_skip_reranking always returns False."""
        retriever = _build_retriever(
            mock_vector_store,
            mock_embedding_service,
            use_smart_reranking=False,
        )
        results = [
            _make_search_result("c1", score=0.009, metadata={"original_score": 0.99}),
            _make_search_result("c2", score=0.008, metadata={"original_score": 0.50}),
        ]

        assert retriever._should_skip_reranking(results) is False


# ============================================================================
# Fix 3 -- Greek FTS index in initialize_schema
# ============================================================================

class TestGreekFTSIndex:
    """
    initialize_schema() must create both English and Greek full-text search
    indexes so that Greek tenants can use keyword_search without a separate
    migration step.
    """

    @pytest.fixture
    def store_and_cursor(self):
        """
        Build a VectorStore with fully mocked database connection and return
        both the store and the mock cursor for SQL inspection.
        """
        from execution.legal_rag.vector_store import VectorStore, VectorStoreConfig

        store = VectorStore.__new__(VectorStore)
        store.config = VectorStoreConfig()
        store._pool = None
        store._current_tenant = None
        store._connection_string = "postgresql://fake/test"

        mock_cursor = MagicMock()
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_conn.closed = False

        store._conn = mock_conn

        return store, mock_cursor

    def test_schema_contains_english_fts_index(self, store_and_cursor):
        """initialize_schema SQL must include an English FTS GIN index."""
        store, mock_cursor = store_and_cursor
        store.initialize_schema()

        executed_sql = mock_cursor.execute.call_args[0][0]
        assert "to_tsvector('english'" in executed_sql

    def test_schema_contains_greek_fts_index(self, store_and_cursor):
        """initialize_schema SQL must include a Greek FTS GIN index."""
        store, mock_cursor = store_and_cursor
        store.initialize_schema()

        executed_sql = mock_cursor.execute.call_args[0][0]
        assert "to_tsvector('greek'" in executed_sql

    def test_schema_contains_both_fts_indexes(self, store_and_cursor):
        """Both English and Greek FTS indexes must be present in the same SQL."""
        store, mock_cursor = store_and_cursor
        store.initialize_schema()

        executed_sql = mock_cursor.execute.call_args[0][0]
        assert "to_tsvector('english'" in executed_sql
        assert "to_tsvector('greek'" in executed_sql

    def test_greek_index_name_is_idx_chunks_fts_greek(self, store_and_cursor):
        """The Greek FTS index must use the canonical name idx_chunks_fts_greek."""
        store, mock_cursor = store_and_cursor
        store.initialize_schema()

        executed_sql = mock_cursor.execute.call_args[0][0]
        assert "idx_chunks_fts_greek" in executed_sql

    def test_english_index_name_is_idx_chunks_content_fts(self, store_and_cursor):
        """The English FTS index must use the canonical name idx_chunks_content_fts."""
        store, mock_cursor = store_and_cursor
        store.initialize_schema()

        executed_sql = mock_cursor.execute.call_args[0][0]
        assert "idx_chunks_content_fts" in executed_sql

    def test_both_indexes_use_gin(self, store_and_cursor):
        """Both FTS indexes must use the GIN index type."""
        store, mock_cursor = store_and_cursor
        store.initialize_schema()

        executed_sql = mock_cursor.execute.call_args[0][0]
        # Count GIN occurrences related to tsvector
        assert executed_sql.count("USING GIN (to_tsvector(") >= 2

    def test_schema_calls_commit(self, store_and_cursor):
        """initialize_schema must commit after executing the DDL."""
        store, mock_cursor = store_and_cursor
        mock_conn = store._conn

        store.initialize_schema()

        mock_conn.commit.assert_called_once()
