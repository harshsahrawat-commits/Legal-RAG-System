"""
Tests for execution/legal_rag/embeddings.py

Covers: EmbeddingConfig, EmbeddingService, VoyageEmbeddingService,
        LocalEmbeddingService, get_embedding_service() factory,
        caching behaviour, and batch processing.

All external API calls are mocked.
"""

import json
import hashlib
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


# ---------------------------------------------------------------------------
# EmbeddingConfig
# ---------------------------------------------------------------------------

class TestEmbeddingConfig:
    """Tests for EmbeddingConfig dataclass."""

    def test_defaults(self):
        from execution.legal_rag.embeddings import EmbeddingConfig
        cfg = EmbeddingConfig()
        assert cfg.provider == "voyage"
        assert cfg.model == "voyage-law-2"
        assert cfg.dimensions == 1024
        assert cfg.batch_size == 128
        assert cfg.use_cache is True
        assert cfg.cache_dir is None


# ---------------------------------------------------------------------------
# EmbeddingService (Cohere-based)
# ---------------------------------------------------------------------------

class TestEmbeddingService:
    """Tests for the Cohere-based EmbeddingService."""

    def test_embed_documents_empty_list(self, monkeypatch):
        """embed_documents([]) should return [] without calling the API."""
        from execution.legal_rag.embeddings import EmbeddingService, EmbeddingConfig

        monkeypatch.setenv("COHERE_API_KEY", "fake-key")
        mock_cohere = MagicMock()
        with patch.dict("sys.modules", {"cohere": mock_cohere}):
            svc = EmbeddingService(EmbeddingConfig(use_cache=False))
            result = svc.embed_documents([])
            assert result == []

    def test_embed_query_raises_without_client(self, monkeypatch):
        """If no API key, _client is None and embed_query should raise."""
        from execution.legal_rag.embeddings import EmbeddingService, EmbeddingConfig

        monkeypatch.delenv("COHERE_API_KEY", raising=False)
        # Patch cohere import so it doesn't blow up
        with patch.dict("sys.modules", {"cohere": MagicMock()}):
            svc = EmbeddingService.__new__(EmbeddingService)
            svc.config = EmbeddingConfig(use_cache=False)
            svc._client = None
            svc._cache = {}
            svc._cache_path = None

            with pytest.raises(RuntimeError, match="Cohere client not initialized"):
                svc.embed_query("test query")

    def test_embed_documents_raises_without_client(self, monkeypatch):
        from execution.legal_rag.embeddings import EmbeddingService, EmbeddingConfig

        monkeypatch.delenv("COHERE_API_KEY", raising=False)
        with patch.dict("sys.modules", {"cohere": MagicMock()}):
            svc = EmbeddingService.__new__(EmbeddingService)
            svc.config = EmbeddingConfig(use_cache=False)
            svc._client = None
            svc._cache = {}
            svc._cache_path = None

            with pytest.raises(RuntimeError, match="Cohere client not initialized"):
                svc.embed_documents(["text"])

    def test_dimensions_property(self, monkeypatch):
        from execution.legal_rag.embeddings import EmbeddingService, EmbeddingConfig

        monkeypatch.setenv("COHERE_API_KEY", "fake-key")
        mock_cohere = MagicMock()
        with patch.dict("sys.modules", {"cohere": mock_cohere}):
            svc = EmbeddingService(EmbeddingConfig(dimensions=512))
            assert svc.dimensions == 512


# ---------------------------------------------------------------------------
# EmbeddingService caching
# ---------------------------------------------------------------------------

class TestEmbeddingServiceCaching:
    """Tests for the caching layer in EmbeddingService."""

    def test_cache_key_deterministic(self):
        from execution.legal_rag.embeddings import EmbeddingService

        svc = EmbeddingService.__new__(EmbeddingService)
        svc.config = MagicMock()
        svc.config.model = "test-model"
        key1 = svc._get_cache_key("hello", "query")
        key2 = svc._get_cache_key("hello", "query")
        assert key1 == key2

    def test_cache_key_differs_by_input_type(self):
        from execution.legal_rag.embeddings import EmbeddingService

        svc = EmbeddingService.__new__(EmbeddingService)
        svc.config = MagicMock()
        svc.config.model = "test-model"
        key_q = svc._get_cache_key("hello", "query")
        key_d = svc._get_cache_key("hello", "search_document")
        assert key_q != key_d

    def test_memory_cache_set_and_get(self):
        from execution.legal_rag.embeddings import EmbeddingService, EmbeddingConfig

        svc = EmbeddingService.__new__(EmbeddingService)
        svc.config = EmbeddingConfig(use_cache=True)
        svc._cache = {}
        svc._cache_path = None

        svc._set_cached("key1", [1.0, 2.0, 3.0])
        result = svc._get_cached("key1")
        assert result == [1.0, 2.0, 3.0]

    def test_cache_disabled_returns_none(self):
        from execution.legal_rag.embeddings import EmbeddingService, EmbeddingConfig

        svc = EmbeddingService.__new__(EmbeddingService)
        svc.config = EmbeddingConfig(use_cache=False)
        svc._cache = {"key1": [1.0]}
        svc._cache_path = None

        assert svc._get_cached("key1") is None

    def test_file_cache_write_and_read(self, tmp_path):
        from execution.legal_rag.embeddings import EmbeddingService, EmbeddingConfig

        svc = EmbeddingService.__new__(EmbeddingService)
        svc.config = EmbeddingConfig(use_cache=True, cache_dir=str(tmp_path))
        svc._cache = {}
        svc._cache_path = tmp_path

        embedding = [0.1, 0.2, 0.3]
        svc._set_cached("mykey", embedding)

        # Clear memory cache to force file read
        svc._cache = {}
        result = svc._get_cached("mykey")
        assert result == embedding


# ---------------------------------------------------------------------------
# VoyageEmbeddingService
# ---------------------------------------------------------------------------

class TestVoyageEmbeddingService:
    """Tests for VoyageEmbeddingService."""

    def test_embed_documents_empty(self, monkeypatch):
        from execution.legal_rag.embeddings import VoyageEmbeddingService, EmbeddingConfig

        monkeypatch.setenv("VOYAGE_API_KEY", "fake-key")
        mock_voyage = MagicMock()
        with patch.dict("sys.modules", {"voyageai": mock_voyage}):
            svc = VoyageEmbeddingService(EmbeddingConfig(use_cache=False))
            assert svc.embed_documents([]) == []

    def test_embed_query_raises_without_client(self):
        from execution.legal_rag.embeddings import VoyageEmbeddingService, EmbeddingConfig

        svc = VoyageEmbeddingService.__new__(VoyageEmbeddingService)
        svc.config = EmbeddingConfig(use_cache=False)
        svc._client = None
        svc._cache = {}
        svc._cache_path = None

        with pytest.raises(RuntimeError, match="Voyage AI client not initialized"):
            svc.embed_query("test")

    def test_dimensions_property(self, monkeypatch):
        from execution.legal_rag.embeddings import VoyageEmbeddingService, EmbeddingConfig

        monkeypatch.setenv("VOYAGE_API_KEY", "fake-key")
        mock_voyage = MagicMock()
        with patch.dict("sys.modules", {"voyageai": mock_voyage}):
            svc = VoyageEmbeddingService(EmbeddingConfig(dimensions=768))
            assert svc.dimensions == 768


# ---------------------------------------------------------------------------
# get_embedding_service factory
# ---------------------------------------------------------------------------

class TestGetEmbeddingService:
    """Tests for the get_embedding_service factory function."""

    def test_returns_voyage_by_default(self, monkeypatch):
        from execution.legal_rag.embeddings import get_embedding_service, VoyageEmbeddingService

        monkeypatch.setenv("VOYAGE_API_KEY", "fake-key")
        mock_voyage = MagicMock()
        with patch.dict("sys.modules", {"voyageai": mock_voyage}):
            svc = get_embedding_service(provider="voyage")
            assert isinstance(svc, VoyageEmbeddingService)

    def test_returns_cohere_when_specified(self, monkeypatch):
        from execution.legal_rag.embeddings import get_embedding_service, EmbeddingService

        monkeypatch.setenv("COHERE_API_KEY", "fake-key")
        mock_cohere = MagicMock()
        with patch.dict("sys.modules", {"cohere": mock_cohere}):
            svc = get_embedding_service(provider="cohere")
            assert isinstance(svc, EmbeddingService)

    def test_returns_local_when_use_local_true(self):
        """When use_local=True, should return LocalEmbeddingService."""
        from execution.legal_rag.embeddings import get_embedding_service

        with patch("execution.legal_rag.embeddings.LocalEmbeddingService") as MockLocal:
            MockLocal.return_value = MagicMock()
            svc = get_embedding_service(use_local=True)
            MockLocal.assert_called_once()


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

class TestBatchProcessing:
    """Tests for batch embedding with caching."""

    def test_embed_batch_uses_cache(self, monkeypatch):
        """Cached texts should not be re-embedded via API."""
        from execution.legal_rag.embeddings import EmbeddingService, EmbeddingConfig

        monkeypatch.setenv("COHERE_API_KEY", "fake-key")

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.embeddings = [[0.1] * 1024]
        mock_client.embed.return_value = mock_response

        mock_cohere = MagicMock()
        mock_cohere.Client.return_value = mock_client
        with patch.dict("sys.modules", {"cohere": mock_cohere}):
            svc = EmbeddingService(EmbeddingConfig(use_cache=True))

            # First call: API is called
            result1 = svc._embed_batch(["hello"], input_type="search_document")
            assert mock_client.embed.call_count == 1

            # Second call: should use cache, API not called again
            result2 = svc._embed_batch(["hello"], input_type="search_document")
            assert mock_client.embed.call_count == 1
            assert result1 == result2
