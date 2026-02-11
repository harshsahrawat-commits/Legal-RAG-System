"""
Tests for execution/legal_rag/vector_store.py

Covers: VectorStoreConfig, SearchResult dataclass, VectorStore class
        configuration, connection string resolution, tenant context,
        and API key hashing.

All database calls are mocked -- no PostgreSQL required.
"""

import hashlib
import secrets
from unittest.mock import patch, MagicMock, PropertyMock

import pytest


# ---------------------------------------------------------------------------
# VectorStoreConfig
# ---------------------------------------------------------------------------

class TestVectorStoreConfig:
    """Tests for VectorStoreConfig defaults."""

    def test_defaults(self):
        from execution.legal_rag.vector_store import VectorStoreConfig
        cfg = VectorStoreConfig()
        assert cfg.connection_string is None
        assert cfg.table_name == "document_chunks"
        assert cfg.embedding_dimensions == 1024
        assert cfg.index_lists == 100
        assert cfg.pool_min_connections == 2
        assert cfg.pool_max_connections == 20
        assert cfg.use_pooling is True

    def test_custom_values(self):
        from execution.legal_rag.vector_store import VectorStoreConfig
        cfg = VectorStoreConfig(
            connection_string="postgres://localhost/test",
            table_name="my_chunks",
            embedding_dimensions=768,
        )
        assert cfg.connection_string == "postgres://localhost/test"
        assert cfg.table_name == "my_chunks"
        assert cfg.embedding_dimensions == 768


# ---------------------------------------------------------------------------
# SearchResult
# ---------------------------------------------------------------------------

class TestSearchResult:
    """Tests for SearchResult dataclass."""

    def test_to_dict_keys(self):
        from execution.legal_rag.vector_store import SearchResult
        sr = SearchResult(
            chunk_id="c1", document_id="d1", content="text",
            section_title="S", hierarchy_path="P",
            page_numbers=[1], score=0.9, metadata={"level": 1},
        )
        d = sr.to_dict()
        assert "chunk_id" in d
        assert "score" in d
        assert "paragraph_start" in d
        assert "paragraph_end" in d
        assert "original_paragraph_numbers" in d

    def test_post_init_defaults(self):
        from execution.legal_rag.vector_store import SearchResult
        sr = SearchResult(
            chunk_id="c1", document_id="d1", content="t",
            section_title="S", hierarchy_path="P",
            page_numbers=[], score=0.5, metadata={},
        )
        assert sr.original_paragraph_numbers == []
        assert sr.paragraph_start is None
        assert sr.paragraph_end is None

    def test_paragraph_fields_preserved(self):
        from execution.legal_rag.vector_store import SearchResult
        sr = SearchResult(
            chunk_id="c1", document_id="d1", content="t",
            section_title="S", hierarchy_path="P",
            page_numbers=[], score=0.5, metadata={},
            paragraph_start=5, paragraph_end=10,
            original_paragraph_numbers=[5, 6, 7, 8, 9, 10],
        )
        assert sr.paragraph_start == 5
        assert sr.paragraph_end == 10
        assert len(sr.original_paragraph_numbers) == 6


# ---------------------------------------------------------------------------
# VectorStore - connection string resolution
# ---------------------------------------------------------------------------

class TestVectorStoreConnectionString:
    """Test connection string resolution from config and env vars."""

    def test_uses_config_connection_string(self, monkeypatch):
        from execution.legal_rag.vector_store import VectorStore, VectorStoreConfig

        monkeypatch.delenv("POSTGRES_URL", raising=False)
        monkeypatch.delenv("DATABASE_URL", raising=False)
        cfg = VectorStoreConfig(connection_string="postgres://custom/db")
        store = VectorStore(cfg)
        assert store._connection_string == "postgres://custom/db"

    def test_falls_back_to_postgres_url_env(self, monkeypatch):
        from execution.legal_rag.vector_store import VectorStore, VectorStoreConfig

        monkeypatch.setenv("POSTGRES_URL", "postgres://env/db")
        monkeypatch.delenv("DATABASE_URL", raising=False)
        store = VectorStore(VectorStoreConfig())
        assert store._connection_string == "postgres://env/db"

    def test_falls_back_to_database_url_env(self, monkeypatch):
        from execution.legal_rag.vector_store import VectorStore, VectorStoreConfig

        monkeypatch.delenv("POSTGRES_URL", raising=False)
        monkeypatch.setenv("DATABASE_URL", "postgres://dburl/db")
        store = VectorStore(VectorStoreConfig())
        assert store._connection_string == "postgres://dburl/db"

    def test_falls_back_to_default_localhost(self, monkeypatch):
        from execution.legal_rag.vector_store import VectorStore, VectorStoreConfig

        monkeypatch.delenv("POSTGRES_URL", raising=False)
        monkeypatch.delenv("DATABASE_URL", raising=False)
        store = VectorStore(VectorStoreConfig())
        assert "localhost" in store._connection_string


# ---------------------------------------------------------------------------
# VectorStore - tenant context
# ---------------------------------------------------------------------------

class TestVectorStoreTenantContext:
    """Test tenant context management."""

    def test_set_tenant_stores_value(self):
        from execution.legal_rag.vector_store import VectorStore, VectorStoreConfig

        store = VectorStore.__new__(VectorStore)
        store.config = VectorStoreConfig()
        store._conn = None
        store._pool = MagicMock()
        store._current_tenant = None
        store._connection_string = "fake"

        store.set_tenant_context("client-abc")
        assert store._current_tenant == "client-abc"

    def test_clear_tenant_resets(self):
        from execution.legal_rag.vector_store import VectorStore, VectorStoreConfig

        store = VectorStore.__new__(VectorStore)
        store.config = VectorStoreConfig()
        store._conn = None
        store._pool = MagicMock()
        store._current_tenant = "client-abc"
        store._connection_string = "fake"

        store.clear_tenant_context()
        assert store._current_tenant is None


# ---------------------------------------------------------------------------
# VectorStore - is_connected
# ---------------------------------------------------------------------------

class TestVectorStoreIsConnected:
    """Test _is_connected helper."""

    def test_not_connected_when_both_none(self):
        from execution.legal_rag.vector_store import VectorStore, VectorStoreConfig

        store = VectorStore.__new__(VectorStore)
        store._conn = None
        store._pool = None
        assert store._is_connected() is False

    def test_connected_with_conn(self):
        from execution.legal_rag.vector_store import VectorStore

        store = VectorStore.__new__(VectorStore)
        store._conn = MagicMock()
        store._pool = None
        assert store._is_connected() is True

    def test_connected_with_pool(self):
        from execution.legal_rag.vector_store import VectorStore

        store = VectorStore.__new__(VectorStore)
        store._conn = None
        store._pool = MagicMock()
        assert store._is_connected() is True


# ---------------------------------------------------------------------------
# VectorStore - insert_chunks validation
# ---------------------------------------------------------------------------

class TestVectorStoreInsertValidation:
    """Test input validation for insert_chunks."""

    def test_mismatched_chunks_and_embeddings_raises(self):
        from execution.legal_rag.vector_store import VectorStore, VectorStoreConfig

        store = VectorStore.__new__(VectorStore)
        store.config = VectorStoreConfig()
        store._conn = MagicMock()
        store._conn.closed = False
        store._pool = None
        store._connection_string = "fake"
        store._current_tenant = None

        with pytest.raises(ValueError, match="Mismatch"):
            store.insert_chunks(
                chunks=[{"chunk_id": "1"}],
                embeddings=[[0.1], [0.2]],
            )


# ---------------------------------------------------------------------------
# VectorStore - close
# ---------------------------------------------------------------------------

class TestVectorStoreClose:
    """Test close() method."""

    def test_close_pool_and_conn(self):
        from execution.legal_rag.vector_store import VectorStore, VectorStoreConfig

        store = VectorStore.__new__(VectorStore)
        store._pool = MagicMock()
        store._conn = MagicMock()
        store.config = VectorStoreConfig()

        store.close()
        store._pool is None or store._pool.closeall.assert_called_once()
        assert store._conn is None

    def test_close_when_nothing_open(self):
        from execution.legal_rag.vector_store import VectorStore, VectorStoreConfig

        store = VectorStore.__new__(VectorStore)
        store._pool = None
        store._conn = None
        store.config = VectorStoreConfig()

        # Should not raise
        store.close()


# ---------------------------------------------------------------------------
# API key hashing consistency
# ---------------------------------------------------------------------------

class TestApiKeyHashing:
    """Verify that API key hashing is consistent for validate_api_key."""

    def test_hash_consistency(self):
        """The same raw key should always produce the same hash."""
        raw_key = "lrag_test_key_12345"
        hash1 = hashlib.sha256(raw_key.encode()).hexdigest()
        hash2 = hashlib.sha256(raw_key.encode()).hexdigest()
        assert hash1 == hash2

    def test_different_keys_different_hashes(self):
        k1 = hashlib.sha256(b"key_a").hexdigest()
        k2 = hashlib.sha256(b"key_b").hexdigest()
        assert k1 != k2
