"""
Dimension 2: EFFICIENCY

Tests embedding batching (token-aware), caching behavior, and smart reranking
cost optimization.
"""

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch
from collections import OrderedDict

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.conftest import skip_no_creds

pytestmark = pytest.mark.efficiency


# =============================================================================
# 2A. Embedding Batching (unit)
# =============================================================================

class TestEmbeddingBatching:
    """Verify token-aware batch splitting in BaseEmbeddingService."""

    def _make_service(self, batch_size=128, max_tokens=100000, chars_per_token=4.0):
        """Create a minimal BaseEmbeddingService subclass for testing _create_batches."""
        from execution.legal_rag.embeddings import BaseEmbeddingService, EmbeddingConfig
        config = EmbeddingConfig(
            batch_size=batch_size,
            max_tokens_per_batch=max_tokens,
            chars_per_token=chars_per_token,
        )

        class _TestService(BaseEmbeddingService):
            _provider_name = "Test"
            _env_var_name = "TEST_KEY"
            def _init_client(self):
                self._client = True  # Skip real init

        with patch.dict("os.environ", {"TEST_KEY": "fake"}):
            return _TestService(config)

    def test_batch_split_respects_item_limit(self):
        service = self._make_service(batch_size=3, max_tokens=999999)
        batches = service._create_batches(["text"] * 10)
        for batch in batches:
            assert len(batch) <= 3

    def test_batch_split_respects_token_limit(self):
        # 50 chars / 1.0 cpt = 50 tokens each. Max 100 tokens per batch -> max 2 texts.
        service = self._make_service(batch_size=999, max_tokens=100, chars_per_token=1.0)
        texts = ["x" * 50] * 5
        batches = service._create_batches(texts)
        assert len(batches) >= 3, f"Expected >=3 batches, got {len(batches)}"
        for batch in batches:
            assert len(batch) <= 2

    def test_empty_input_returns_empty(self):
        service = self._make_service()
        assert service._create_batches([]) == []

    def test_single_large_text_gets_own_batch(self):
        service = self._make_service(batch_size=10, max_tokens=10, chars_per_token=1.0)
        batches = service._create_batches(["x" * 100])
        assert len(batches) == 1
        assert len(batches[0]) == 1

    def test_mixed_sizes_batch_correctly(self):
        # Small texts should group, large texts force new batches
        service = self._make_service(batch_size=100, max_tokens=50, chars_per_token=1.0)
        texts = ["short"] * 3 + ["x" * 45] + ["short"] * 3
        batches = service._create_batches(texts)
        assert len(batches) >= 2
        total_texts = sum(len(b) for b in batches)
        assert total_texts == 7, "No texts should be dropped"


# =============================================================================
# 2B. Caching (unit)
# =============================================================================

class TestQueryResultCaching:
    """Verify the QueryResultCache behavior."""

    def _make_cache(self, max_size=100, ttl_seconds=3600):
        from execution.legal_rag.retriever import QueryResultCache
        from tests.conftest import MockEmbeddingService
        mock_emb = MockEmbeddingService()
        return QueryResultCache(mock_emb, max_size=max_size, ttl_seconds=ttl_seconds)

    def _mock_result(self, chunk_id="c1"):
        from execution.legal_rag.vector_store import SearchResult
        return SearchResult(
            chunk_id=chunk_id, document_id="d1", content="test",
            section_title="S1", hierarchy_path="Doc/S1",
            page_numbers=[1], score=0.8, metadata={},
        )

    def test_cache_set_and_get(self):
        cache = self._make_cache()
        result = self._mock_result()
        cache.set("test query about contracts", [result])
        assert cache.size == 1

    def test_cache_eviction_at_capacity(self):
        cache = self._make_cache(max_size=2)
        r = self._mock_result()
        cache.set("query one about law", [r])
        cache.set("query two about contracts", [r])
        assert cache.size == 2
        cache.set("query three about torts", [r])
        assert cache.size <= 2, "Cache should evict oldest when at capacity"

    def test_cache_clear(self):
        cache = self._make_cache()
        r = self._mock_result()
        cache.set("query about law", [r])
        cache.clear()
        assert cache.size == 0


class TestRerankCacheBounded:
    """Verify the rerank cache stays within its max size."""

    def test_rerank_cache_stays_bounded(self):
        from execution.legal_rag.retriever import HybridRetriever, RetrievalConfig
        from tests.conftest import MockEmbeddingService, MockVectorStore

        store = MockVectorStore()
        emb = MockEmbeddingService()
        retriever = HybridRetriever(store, emb, config=RetrievalConfig(use_reranking=False))
        retriever._rerank_cache_max_size = 5

        # Simulate adding entries beyond max
        for i in range(10):
            retriever._rerank_cache[f"key_{i}"] = [MagicMock()]
            while len(retriever._rerank_cache) > retriever._rerank_cache_max_size:
                retriever._rerank_cache.popitem(last=False)

        assert len(retriever._rerank_cache) <= 5


# =============================================================================
# 2C. Smart Reranking (unit)
# =============================================================================

class TestSmartReranking:
    """Verify smart reranking skips expensive API calls when appropriate."""

    def _make_retriever(self, threshold=0.85, gap=0.15, use_smart=True):
        from execution.legal_rag.retriever import HybridRetriever, RetrievalConfig
        from tests.conftest import MockEmbeddingService, MockVectorStore
        config = RetrievalConfig(
            use_reranking=True,
            use_smart_reranking=use_smart,
            smart_rerank_threshold=threshold,
            smart_rerank_gap=gap,
        )
        return HybridRetriever(MockVectorStore(), MockEmbeddingService(), config=config)

    def _make_result(self, score):
        from execution.legal_rag.vector_store import SearchResult
        return SearchResult(
            chunk_id="c1", document_id="d1", content="test",
            section_title="S1", hierarchy_path="Doc/S1",
            page_numbers=[1], score=score, metadata={},
        )

    def test_high_confidence_skips_rerank(self):
        retriever = self._make_retriever()
        results = [self._make_result(0.95), self._make_result(0.70)]
        assert retriever._should_skip_reranking(results) is True

    def test_close_scores_triggers_rerank(self):
        retriever = self._make_retriever()
        results = [self._make_result(0.86), self._make_result(0.84)]
        assert retriever._should_skip_reranking(results) is False

    def test_low_confidence_triggers_rerank(self):
        retriever = self._make_retriever()
        results = [self._make_result(0.60), self._make_result(0.40)]
        assert retriever._should_skip_reranking(results) is False

    def test_single_result_skips_rerank(self):
        retriever = self._make_retriever()
        results = [self._make_result(0.90)]
        assert retriever._should_skip_reranking(results) is True

    def test_smart_reranking_disabled(self):
        retriever = self._make_retriever(use_smart=False)
        results = [self._make_result(0.99), self._make_result(0.10)]
        assert retriever._should_skip_reranking(results) is False
