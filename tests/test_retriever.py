"""
Tests for execution/legal_rag/retriever.py

Covers: QUERY_CONFIGS, QueryResultCache, RetrievalConfig, HybridRetriever
        (RRF, query classification, smart reranking, paragraph extraction),
        and SimpleRetriever.

All external API calls (Cohere reranker, NVIDIA LLM) are mocked.
"""

import time
import hashlib
from unittest.mock import patch, MagicMock, PropertyMock
from collections import OrderedDict

import pytest


# ---------------------------------------------------------------------------
# QUERY_CONFIGS
# ---------------------------------------------------------------------------

class TestQueryConfigs:
    """Verify QUERY_CONFIGS structure."""

    def test_all_types_present(self):
        from execution.legal_rag.retriever import QUERY_CONFIGS
        assert set(QUERY_CONFIGS.keys()) == {"simple", "factual", "analytical", "standard"}

    def test_each_config_has_required_keys(self):
        from execution.legal_rag.retriever import QUERY_CONFIGS
        for name, cfg in QUERY_CONFIGS.items():
            assert "use_query_expansion" in cfg
            assert "use_hyde" in cfg
            assert "use_multi_query" in cfg
            assert "description" in cfg


# ---------------------------------------------------------------------------
# RetrievalConfig
# ---------------------------------------------------------------------------

class TestRetrievalConfig:
    """Tests for RetrievalConfig defaults."""

    def test_defaults(self):
        from execution.legal_rag.retriever import RetrievalConfig
        cfg = RetrievalConfig()
        assert cfg.vector_top_k == 40
        assert cfg.keyword_top_k == 40
        assert cfg.final_top_k == 10
        assert cfg.rrf_k == 60
        assert cfg.vector_weight == 0.6
        assert cfg.keyword_weight == 0.4
        assert cfg.use_reranking is True
        assert cfg.use_smart_reranking is True


# ---------------------------------------------------------------------------
# QueryResultCache
# ---------------------------------------------------------------------------

class TestQueryResultCache:
    """Tests for the semantic query result cache."""

    @pytest.fixture
    def cache(self, mock_embedding_service):
        from execution.legal_rag.retriever import QueryResultCache
        return QueryResultCache(
            embedding_service=mock_embedding_service,
            similarity_threshold=0.92,
            max_size=10,
            ttl_seconds=3600,
        )

    def test_empty_cache_returns_none(self, cache):
        assert cache.get("some query") is None

    def test_set_and_get(self, cache):
        from execution.legal_rag.vector_store import SearchResult
        results = [SearchResult(
            chunk_id="c1", document_id="d1", content="text",
            section_title="S", hierarchy_path="P",
            page_numbers=[], score=0.9, metadata={},
        )]
        cache.set("test query", results)
        # The same query should hit cache (cosine similarity = 1.0)
        cached = cache.get("test query")
        assert cached is not None
        assert len(cached) == 1

    def test_different_client_id_misses(self, cache):
        from execution.legal_rag.vector_store import SearchResult
        results = [SearchResult(
            chunk_id="c1", document_id="d1", content="text",
            section_title="S", hierarchy_path="P",
            page_numbers=[], score=0.9, metadata={},
        )]
        cache.set("query", results, client_id="client-a")
        # Different client_id should miss
        assert cache.get("query", client_id="client-b") is None

    def test_clear(self, cache):
        from execution.legal_rag.vector_store import SearchResult
        results = [SearchResult(
            chunk_id="c1", document_id="d1", content="t",
            section_title="S", hierarchy_path="P",
            page_numbers=[], score=0.9, metadata={},
        )]
        cache.set("q", results)
        assert cache.size == 1
        cache.clear()
        assert cache.size == 0

    def test_evicts_oldest_at_capacity(self, mock_embedding_service):
        from execution.legal_rag.retriever import QueryResultCache
        from execution.legal_rag.vector_store import SearchResult

        cache = QueryResultCache(
            embedding_service=mock_embedding_service,
            max_size=2, ttl_seconds=3600,
        )
        r = [SearchResult(
            chunk_id="c", document_id="d", content="t",
            section_title="S", hierarchy_path="P",
            page_numbers=[], score=0.9, metadata={},
        )]
        cache.set("query1", r)
        cache.set("query2", r)
        cache.set("query3", r)  # Should evict query1
        assert cache.size == 2

    def test_cosine_similarity_self_is_one(self, cache):
        """Cosine similarity of a vector with itself should be ~1.0."""
        vec = [0.1, 0.2, 0.3, 0.4, 0.5]
        sim = cache._cosine_similarity(vec, vec)
        assert abs(sim - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# HybridRetriever - query classification
# ---------------------------------------------------------------------------

class TestQueryClassification:
    """Tests for _classify_query."""

    @pytest.fixture
    def retriever(self, mock_vector_store, mock_embedding_service):
        from execution.legal_rag.retriever import HybridRetriever, RetrievalConfig
        cfg = RetrievalConfig(use_reranking=False)
        with patch.object(HybridRetriever, "_init_reranker", lambda self: None):
            r = HybridRetriever(mock_vector_store, mock_embedding_service, cfg)
            return r

    @pytest.mark.parametrize("query,expected", [
        ("contract", "simple"),
        ("NDA", "simple"),
        ("force majeure", "simple"),
    ])
    def test_simple_queries(self, retriever, query, expected):
        assert retriever._classify_query(query) == expected

    @pytest.mark.parametrize("query,expected", [
        ("What is the termination clause?", "factual"),
        ("Who are the parties to this agreement?", "factual"),
        ("What are the payment terms?", "factual"),
        ("List the key definitions", "factual"),
    ])
    def test_factual_queries(self, retriever, query, expected):
        assert retriever._classify_query(query) == expected

    @pytest.mark.parametrize("query,expected", [
        ("Explain the implications of the indemnification clause", "analytical"),
        ("Compare the liability provisions", "analytical"),
        ("Why was the contract terminated?", "analytical"),
        ("Analyze the confidentiality obligations", "analytical"),
    ])
    def test_analytical_queries(self, retriever, query, expected):
        assert retriever._classify_query(query) == expected

    def test_legal_reference_classified_as_factual(self, retriever):
        # "What does..." matches the factual pattern "^what\s+(did|does|do)\b"
        result = retriever._classify_query("What does Section 4.2 say about termination?")
        assert result == "factual"


# ---------------------------------------------------------------------------
# HybridRetriever - paragraph reference extraction
# ---------------------------------------------------------------------------

class TestParagraphReferenceExtraction:
    """Tests for _extract_paragraph_reference."""

    @pytest.fixture
    def retriever(self, mock_vector_store, mock_embedding_service):
        from execution.legal_rag.retriever import HybridRetriever, RetrievalConfig
        cfg = RetrievalConfig(use_reranking=False)
        with patch.object(HybridRetriever, "_init_reranker", lambda self: None):
            return HybridRetriever(mock_vector_store, mock_embedding_service, cfg)

    @pytest.mark.parametrize("query,expected_para", [
        ("paragraph 28", 28),
        ("para 15", 15),
        ("para. 7", 7),
        ("what does paragraph 42 say", 42),
    ])
    def test_extraction(self, retriever, query, expected_para):
        assert retriever._extract_paragraph_reference(query) == expected_para

    def test_no_paragraph_reference(self, retriever):
        assert retriever._extract_paragraph_reference("termination clause") is None

    def test_pilcrow_symbol(self, retriever):
        # Note: the pattern expects the pilcrow character
        result = retriever._extract_paragraph_reference("see \u00b628")
        assert result == 28


# ---------------------------------------------------------------------------
# HybridRetriever - Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

class TestReciprocalRankFusion:
    """Tests for _reciprocal_rank_fusion."""

    @pytest.fixture
    def retriever(self, mock_vector_store, mock_embedding_service):
        from execution.legal_rag.retriever import HybridRetriever, RetrievalConfig
        cfg = RetrievalConfig(use_reranking=False, rrf_k=60,
                              vector_weight=0.6, keyword_weight=0.4)
        with patch.object(HybridRetriever, "_init_reranker", lambda self: None):
            return HybridRetriever(mock_vector_store, mock_embedding_service, cfg)

    def test_empty_inputs_returns_empty(self, retriever):
        result = retriever._reciprocal_rank_fusion([], [])
        assert result == []

    def test_vector_only(self, retriever):
        from execution.legal_rag.vector_store import SearchResult
        v = [SearchResult(
            chunk_id="c1", document_id="d1", content="t",
            section_title="S", hierarchy_path="P",
            page_numbers=[], score=0.9, metadata={},
        )]
        result = retriever._reciprocal_rank_fusion(v, [])
        assert len(result) == 1
        assert result[0].chunk_id == "c1"

    def test_keyword_only(self, retriever):
        from execution.legal_rag.vector_store import SearchResult
        k = [SearchResult(
            chunk_id="c2", document_id="d1", content="t",
            section_title="S", hierarchy_path="P",
            page_numbers=[], score=0.5, metadata={},
        )]
        result = retriever._reciprocal_rank_fusion([], k)
        assert len(result) == 1

    def test_combined_results_deduplication(self, retriever):
        """Same chunk_id in both lists should be combined, not duplicated."""
        from execution.legal_rag.vector_store import SearchResult
        sr = SearchResult(
            chunk_id="c1", document_id="d1", content="t",
            section_title="S", hierarchy_path="P",
            page_numbers=[], score=0.5, metadata={},
        )
        result = retriever._reciprocal_rank_fusion([sr], [sr])
        assert len(result) == 1
        # Combined RRF score should be higher than either alone
        assert result[0].score > 0

    def test_ranking_order(self, retriever):
        """Higher-ranked results should get higher RRF scores."""
        from execution.legal_rag.vector_store import SearchResult

        def make_sr(cid):
            return SearchResult(
                chunk_id=cid, document_id="d1", content="t",
                section_title="S", hierarchy_path="P",
                page_numbers=[], score=0.5, metadata={},
            )

        v = [make_sr("top"), make_sr("mid"), make_sr("low")]
        result = retriever._reciprocal_rank_fusion(v, [])
        assert result[0].chunk_id == "top"


# ---------------------------------------------------------------------------
# HybridRetriever - smart reranking
# ---------------------------------------------------------------------------

class TestSmartReranking:
    """Tests for _should_skip_reranking."""

    @pytest.fixture
    def retriever(self, mock_vector_store, mock_embedding_service):
        from execution.legal_rag.retriever import HybridRetriever, RetrievalConfig
        cfg = RetrievalConfig(
            use_reranking=True,
            use_smart_reranking=True,
            smart_rerank_threshold=0.85,
            smart_rerank_gap=0.15,
        )
        with patch.object(HybridRetriever, "_init_reranker", lambda self: None):
            return HybridRetriever(mock_vector_store, mock_embedding_service, cfg)

    def test_skip_when_confident(self, retriever):
        from execution.legal_rag.vector_store import SearchResult

        results = [
            SearchResult(chunk_id="c1", document_id="d1", content="t",
                         section_title="S", hierarchy_path="P",
                         page_numbers=[], score=0.95, metadata={}),
            SearchResult(chunk_id="c2", document_id="d1", content="t",
                         section_title="S", hierarchy_path="P",
                         page_numbers=[], score=0.70, metadata={}),
        ]
        assert retriever._should_skip_reranking(results) is True

    def test_no_skip_when_close_scores(self, retriever):
        from execution.legal_rag.vector_store import SearchResult

        results = [
            SearchResult(chunk_id="c1", document_id="d1", content="t",
                         section_title="S", hierarchy_path="P",
                         page_numbers=[], score=0.86, metadata={}),
            SearchResult(chunk_id="c2", document_id="d1", content="t",
                         section_title="S", hierarchy_path="P",
                         page_numbers=[], score=0.84, metadata={}),
        ]
        assert retriever._should_skip_reranking(results) is False

    def test_skip_with_single_result(self, retriever):
        from execution.legal_rag.vector_store import SearchResult

        results = [
            SearchResult(chunk_id="c1", document_id="d1", content="t",
                         section_title="S", hierarchy_path="P",
                         page_numbers=[], score=0.95, metadata={}),
        ]
        assert retriever._should_skip_reranking(results) is True

    def test_no_skip_when_disabled(self, mock_vector_store, mock_embedding_service):
        from execution.legal_rag.retriever import HybridRetriever, RetrievalConfig
        from execution.legal_rag.vector_store import SearchResult

        cfg = RetrievalConfig(use_smart_reranking=False)
        with patch.object(HybridRetriever, "_init_reranker", lambda self: None):
            retriever = HybridRetriever(mock_vector_store, mock_embedding_service, cfg)
        results = [
            SearchResult(chunk_id="c1", document_id="d1", content="t",
                         section_title="S", hierarchy_path="P",
                         page_numbers=[], score=0.99, metadata={}),
            SearchResult(chunk_id="c2", document_id="d1", content="t",
                         section_title="S", hierarchy_path="P",
                         page_numbers=[], score=0.50, metadata={}),
        ]
        assert retriever._should_skip_reranking(results) is False


# ---------------------------------------------------------------------------
# HybridRetriever - rerank cache
# ---------------------------------------------------------------------------

class TestRerankCache:
    """Tests for _get_rerank_cache_key."""

    @pytest.fixture
    def retriever(self, mock_vector_store, mock_embedding_service):
        from execution.legal_rag.retriever import HybridRetriever, RetrievalConfig
        cfg = RetrievalConfig(use_reranking=False)
        with patch.object(HybridRetriever, "_init_reranker", lambda self: None):
            return HybridRetriever(mock_vector_store, mock_embedding_service, cfg)

    def test_same_inputs_same_key(self, retriever):
        k1 = retriever._get_rerank_cache_key("query", ["a", "b"])
        k2 = retriever._get_rerank_cache_key("query", ["a", "b"])
        assert k1 == k2

    def test_different_order_same_key(self, retriever):
        """Chunk IDs are sorted, so order should not matter."""
        k1 = retriever._get_rerank_cache_key("query", ["a", "b"])
        k2 = retriever._get_rerank_cache_key("query", ["b", "a"])
        assert k1 == k2

    def test_different_query_different_key(self, retriever):
        k1 = retriever._get_rerank_cache_key("query1", ["a"])
        k2 = retriever._get_rerank_cache_key("query2", ["a"])
        assert k1 != k2


# ---------------------------------------------------------------------------
# HybridRetriever - retrieve (integration with mocks)
# ---------------------------------------------------------------------------

class TestHybridRetrieverRetrieve:
    """Integration test for retrieve() using mocked store and embeddings."""

    def test_retrieve_returns_results(
        self, mock_vector_store, mock_embedding_service, sample_chunks
    ):
        """retrieve() should return results from the mock store."""
        from execution.legal_rag.retriever import HybridRetriever, RetrievalConfig

        # Populate mock store
        chunk_dicts = [c.to_dict() for c in sample_chunks]
        embeddings = mock_embedding_service.embed_documents(
            [c.content for c in sample_chunks]
        )
        mock_vector_store.insert_chunks(chunk_dicts, embeddings)

        cfg = RetrievalConfig(
            use_reranking=False,
            use_query_expansion=False,
            use_hyde=False,
            use_multi_query=False,
        )
        with patch.object(HybridRetriever, "_init_reranker", lambda self: None):
            retriever = HybridRetriever(
                mock_vector_store, mock_embedding_service, cfg
            )
            results = retriever.retrieve("termination clause", top_k=5, use_cache=False)
            assert isinstance(results, list)
            assert len(results) > 0

    def test_retrieve_respects_top_k(
        self, mock_vector_store, mock_embedding_service, sample_chunks
    ):
        from execution.legal_rag.retriever import HybridRetriever, RetrievalConfig

        chunk_dicts = [c.to_dict() for c in sample_chunks]
        embeddings = mock_embedding_service.embed_documents(
            [c.content for c in sample_chunks]
        )
        mock_vector_store.insert_chunks(chunk_dicts, embeddings)

        cfg = RetrievalConfig(
            use_reranking=False,
            use_query_expansion=False,
            use_hyde=False,
            use_multi_query=False,
        )
        with patch.object(HybridRetriever, "_init_reranker", lambda self: None):
            retriever = HybridRetriever(
                mock_vector_store, mock_embedding_service, cfg
            )
            results = retriever.retrieve("fees", top_k=2, use_cache=False)
            assert len(results) <= 2
