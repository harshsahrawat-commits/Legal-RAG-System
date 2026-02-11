"""
Tests for batch 4 of codebase review fixes applied to the Legal RAG System.

Fix 10: Cache key collision fix (retriever.py)
    _make_cache_key() now samples embedding[::32] (every 32nd element) instead
    of embedding[:8] (first 8 elements). This spreads sampling across the full
    vector for fewer hash collisions.

Fix 11: MockVectorStore.delete_document fix (conftest.py)
    delete_document was returning the deletion flag before cleaning up associated
    chunks. The return is now placed AFTER the chunk cleanup loop.

Fix 12: Token estimation min 1 (chunker.py)
    _estimate_tokens() now returns max(1, len(text) // chars_per_token) instead
    of len(text) // chars_per_token, preventing a zero result for very short text.

Fix 13: websearch_to_tsquery (vector_store.py)
    keyword_search() now uses websearch_to_tsquery instead of plainto_tsquery,
    providing better morphological matching for Greek text.

All external API calls are mocked -- no database, LLM, or embedding service
is required.
"""

import hashlib
from unittest.mock import patch, MagicMock

import pytest


# ============================================================================
# Helpers
# ============================================================================

def _build_cache(mock_embedding_service, **kwargs):
    """Construct a QueryResultCache with the given embedding service."""
    from execution.legal_rag.retriever import QueryResultCache

    return QueryResultCache(
        embedding_service=mock_embedding_service,
        **kwargs,
    )


def _build_chunker(language="en", **config_overrides):
    """
    Construct a LegalChunker for the specified language.

    Accepts optional ChunkConfig overrides.
    """
    from execution.legal_rag.chunker import LegalChunker, ChunkConfig
    from execution.legal_rag.language_config import TenantLanguageConfig

    cfg = ChunkConfig(**config_overrides) if config_overrides else None
    lang_cfg = TenantLanguageConfig.for_language(language)
    return LegalChunker(config=cfg, language_config=lang_cfg)


def _build_vector_store_with_mock_cursor():
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
    # Make cursor work as context manager
    mock_cursor.fetchall.return_value = []

    mock_conn = MagicMock()
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    mock_conn.closed = False

    store._conn = mock_conn

    return store, mock_cursor


# ============================================================================
# Fix 10 -- Cache key collision fix in _make_cache_key
# ============================================================================

class TestCacheKeyCollisionFix:
    """
    _make_cache_key() now samples every 32nd element (embedding[::32]) instead
    of only the first 8 elements (embedding[:8]).  This means two embeddings
    that share the same first 8 values but differ later will produce different
    cache keys, reducing collision risk.
    """

    @pytest.fixture
    def cache(self, mock_embedding_service):
        """Return a QueryResultCache instance."""
        return _build_cache(mock_embedding_service)

    def test_different_embeddings_with_shared_prefix_produce_different_keys(self, cache):
        """Two 1024-d embeddings sharing the first 8 elements but differing later
        must produce DIFFERENT cache keys under the new ::32 sampling scheme."""
        base = [0.5] * 1024
        emb_a = base[:]
        emb_b = base[:]

        # Alter elements beyond index 8 but at positions that ARE sampled by ::32
        # Sampled indices: 0, 32, 64, 96, ...
        emb_b[32] = 0.999
        emb_b[64] = 0.001

        key_a = cache._make_cache_key(emb_a)
        key_b = cache._make_cache_key(emb_b)

        assert key_a != key_b, (
            "Embeddings that differ at index 32 and 64 should produce different "
            "cache keys when sampling every 32nd element."
        )

    def test_same_embedding_produces_same_cache_key(self, cache):
        """Determinism: the same embedding must always produce the same key."""
        emb = [float(i) / 1000.0 for i in range(1024)]

        key_1 = cache._make_cache_key(emb)
        key_2 = cache._make_cache_key(emb)

        assert key_1 == key_2

    def test_short_embedding_under_32_elements_still_works(self, cache):
        """Embeddings shorter than 32 elements must not raise and must produce
        a valid hex cache key."""
        short_emb = [0.1, 0.2, 0.3, 0.4, 0.5]

        key = cache._make_cache_key(short_emb)

        assert isinstance(key, str)
        assert len(key) == 16
        # Must be valid hexadecimal
        int(key, 16)

    def test_cache_key_is_hex_string_of_length_16(self, cache):
        """_make_cache_key must return a 16-character hexadecimal string
        (first 16 chars of a SHA-256 hex digest)."""
        emb = [0.42] * 1024

        key = cache._make_cache_key(emb)

        assert isinstance(key, str)
        assert len(key) == 16
        # Validate hex format
        int(key, 16)

    def test_sampling_stride_is_32(self, cache):
        """Verify the key is derived from every 32nd element, not the first 8."""
        emb = [0.0] * 1024
        # The sampled subset should be emb[::32] = indices 0, 32, 64, ..., 992
        sampled = emb[::32]
        expected_key = hashlib.sha256(str(sampled).encode()).hexdigest()[:16]

        actual_key = cache._make_cache_key(emb)

        assert actual_key == expected_key

    def test_identical_first_8_but_different_later_gives_different_keys(self, cache):
        """A more explicit test: embeddings identical in [:8] but different at
        index 32 must produce different cache keys."""
        emb_a = [0.0] * 1024
        emb_b = [0.0] * 1024
        # First 8 elements are identical
        assert emb_a[:8] == emb_b[:8]
        # Change an element that the ::32 stride samples
        emb_b[32] = 1.0

        key_a = cache._make_cache_key(emb_a)
        key_b = cache._make_cache_key(emb_b)

        assert key_a != key_b

    def test_short_embedding_exactly_32_elements(self, cache):
        """An embedding with exactly 32 elements should still work. The stride
        samples indices 0 (the only sampled element since 32 is not < 32+1
        with step 32 would give [0])."""
        emb = [float(i) for i in range(32)]

        key = cache._make_cache_key(emb)

        assert isinstance(key, str)
        assert len(key) == 16

    def test_empty_embedding_produces_valid_key(self, cache):
        """Even an empty embedding should not crash and should produce a key."""
        emb = []

        key = cache._make_cache_key(emb)

        assert isinstance(key, str)
        assert len(key) == 16


# ============================================================================
# Fix 11 -- MockVectorStore.delete_document fix
# ============================================================================

class TestMockVectorStoreDeleteDocument:
    """
    MockVectorStore.delete_document was returning the `deleted` flag BEFORE
    cleaning up associated chunks. Now the return is placed AFTER chunk cleanup,
    ensuring the mock faithfully simulates cascade deletion.
    """

    def test_delete_existing_document_returns_true(self, mock_vector_store):
        """delete_document must return True when the document exists."""
        mock_vector_store.insert_document("doc-1", "Contract A", "contract")

        result = mock_vector_store.delete_document("doc-1")

        assert result is True

    def test_delete_nonexistent_document_returns_false(self, mock_vector_store):
        """delete_document must return False when the document does not exist."""
        result = mock_vector_store.delete_document("no-such-doc")

        assert result is False

    def test_associated_chunks_removed_after_deletion(self, mock_vector_store):
        """After deleting a document, all associated chunks must be removed
        from the internal _chunks dict."""
        mock_vector_store.insert_document("doc-1", "Contract A", "contract")
        # Insert chunks referencing doc-1
        chunks = [
            {"chunk_id": "c1", "document_id": "doc-1", "content": "Section 1 text"},
            {"chunk_id": "c2", "document_id": "doc-1", "content": "Section 2 text"},
        ]
        embeddings = [[0.1] * 10, [0.2] * 10]
        mock_vector_store.insert_chunks(chunks, embeddings)

        # Also insert a chunk for a DIFFERENT document (should NOT be removed)
        mock_vector_store.insert_document("doc-2", "Contract B", "contract")
        other_chunks = [
            {"chunk_id": "c3", "document_id": "doc-2", "content": "Other doc text"},
        ]
        mock_vector_store.insert_chunks(other_chunks, [[0.3] * 10])

        # Delete doc-1
        mock_vector_store.delete_document("doc-1")

        # doc-1 chunks must be gone
        assert "c1" not in mock_vector_store._chunks
        assert "c2" not in mock_vector_store._chunks
        # doc-2 chunk must remain
        assert "c3" in mock_vector_store._chunks

    def test_document_no_longer_in_list_after_deletion(self, mock_vector_store):
        """After deletion, the document must not appear in list_documents."""
        mock_vector_store.insert_document("doc-1", "Contract A", "contract")
        mock_vector_store.insert_document("doc-2", "Contract B", "contract")

        mock_vector_store.delete_document("doc-1")

        docs = mock_vector_store.list_documents()
        doc_ids = [d["id"] for d in docs]
        assert "doc-1" not in doc_ids
        assert "doc-2" in doc_ids

    def test_delete_returns_true_and_chunks_are_cleaned(self, mock_vector_store):
        """Verify both the return value and chunk cleanup happen atomically --
        the return value reflects success AND chunks are cleaned up."""
        mock_vector_store.insert_document("doc-1", "Contract A", "contract")
        chunks = [
            {"chunk_id": "c1", "document_id": "doc-1", "content": "Text"},
        ]
        mock_vector_store.insert_chunks(chunks, [[0.1] * 10])

        result = mock_vector_store.delete_document("doc-1")

        # Both conditions must hold simultaneously
        assert result is True
        assert len(mock_vector_store._chunks) == 0

    def test_delete_with_no_chunks_returns_true(self, mock_vector_store):
        """Deleting a document that has no associated chunks should still
        return True and not error."""
        mock_vector_store.insert_document("doc-lonely", "No Chunks", "memo")

        result = mock_vector_store.delete_document("doc-lonely")

        assert result is True
        assert "doc-lonely" not in mock_vector_store._documents


# ============================================================================
# Fix 12 -- Token estimation min 1
# ============================================================================

class TestTokenEstimationMin1:
    """
    _estimate_tokens() now returns max(1, len(text) // chars_per_token)
    instead of raw integer division, ensuring the result is always >= 1
    for any non-negative input.  This prevents downstream division-by-zero
    when token counts are used as denominators.
    """

    @pytest.fixture
    def english_chunker(self):
        """LegalChunker with English config (chars_per_token = 4)."""
        return _build_chunker(language="en")

    @pytest.fixture
    def greek_chunker(self):
        """LegalChunker with Greek config (chars_per_token = 3)."""
        return _build_chunker(language="el")

    def test_empty_string_returns_1_not_0(self, english_chunker):
        """An empty string has 0 characters; len('') // 4 = 0, but max(1, 0) = 1."""
        result = english_chunker._estimate_tokens("")

        assert result == 1

    def test_single_character_returns_1_not_0(self, english_chunker):
        """A single character: len('x') // 4 = 0, but max(1, 0) = 1."""
        result = english_chunker._estimate_tokens("x")

        assert result == 1

    def test_short_string_under_chars_per_token_returns_1(self, english_chunker):
        """'abc' has 3 chars; 3 // 4 = 0, but max(1, 0) = 1."""
        result = english_chunker._estimate_tokens("abc")

        assert result == 1

    def test_normal_text_returns_expected_count(self, english_chunker):
        """Normal-length text should return len(text) // chars_per_token."""
        text = "This is a normal sentence with enough characters to test."
        expected = max(1, len(text) // 4)

        result = english_chunker._estimate_tokens(text)

        assert result == expected
        assert result > 1

    def test_very_long_text_returns_proportionally_large_count(self, english_chunker):
        """Very long text should return a proportionally large token estimate."""
        text = "word " * 10000  # 50000 characters
        expected = max(1, len(text) // 4)

        result = english_chunker._estimate_tokens(text)

        assert result == expected
        assert result > 10000

    def test_exactly_chars_per_token_length_returns_1(self, english_chunker):
        """Text with exactly chars_per_token characters: 4 // 4 = 1."""
        result = english_chunker._estimate_tokens("abcd")

        assert result == 1

    def test_greek_empty_string_returns_1(self, greek_chunker):
        """Greek config (chars_per_token=3): empty string still returns 1."""
        result = greek_chunker._estimate_tokens("")

        assert result == 1

    def test_greek_two_chars_returns_1(self, greek_chunker):
        """Greek config: 2 chars // 3 = 0, but max(1, 0) = 1."""
        result = greek_chunker._estimate_tokens("ab")

        assert result == 1

    def test_greek_normal_text_returns_expected(self, greek_chunker):
        """Greek config: normal text returns len(text) // 3."""
        text = "Some standard text for testing purposes."
        expected = max(1, len(text) // 3)

        result = greek_chunker._estimate_tokens(text)

        assert result == expected

    @pytest.mark.parametrize("length", [0, 1, 2, 3, 4, 5, 10, 100, 1000])
    def test_result_always_at_least_1_english(self, english_chunker, length):
        """For any text length, the result must be >= 1 (English config)."""
        text = "a" * length

        result = english_chunker._estimate_tokens(text)

        assert result >= 1

    @pytest.mark.parametrize("length", [0, 1, 2, 3, 4, 5, 10, 100, 1000])
    def test_result_always_at_least_1_greek(self, greek_chunker, length):
        """For any text length, the result must be >= 1 (Greek config)."""
        text = "a" * length

        result = greek_chunker._estimate_tokens(text)

        assert result >= 1


# ============================================================================
# Fix 13 -- websearch_to_tsquery in keyword_search
# ============================================================================

class TestWebsearchToTsquery:
    """
    keyword_search() now uses websearch_to_tsquery instead of plainto_tsquery.
    websearch_to_tsquery provides better morphological matching and supports
    operators like quoted phrases and the minus sign, improving Greek FTS.
    """

    @pytest.fixture
    def store_and_cursor(self):
        """Build a VectorStore with mocked database and return (store, cursor)."""
        return _build_vector_store_with_mock_cursor()

    def test_sql_uses_websearch_to_tsquery(self, store_and_cursor):
        """keyword_search SQL must contain websearch_to_tsquery, not plainto_tsquery."""
        store, mock_cursor = store_and_cursor

        store.keyword_search(query="termination clause", top_k=5)

        executed_sql = mock_cursor.execute.call_args[0][0]
        assert "websearch_to_tsquery" in executed_sql
        assert "plainto_tsquery" not in executed_sql

    def test_fts_language_parameter_passed_correctly_english(self, store_and_cursor):
        """keyword_search must pass the fts_language to websearch_to_tsquery
        as a positional parameter."""
        store, mock_cursor = store_and_cursor

        store.keyword_search(
            query="contract breach",
            top_k=10,
            fts_language="english",
        )

        executed_params = mock_cursor.execute.call_args[0][1]
        # fts_language appears multiple times in the params
        # (for ts_rank, websearch_to_tsquery in SELECT and WHERE)
        assert "english" in executed_params

    def test_fts_language_parameter_passed_correctly_greek(self, store_and_cursor):
        """keyword_search must pass 'greek' when fts_language='greek'."""
        store, mock_cursor = store_and_cursor

        store.keyword_search(
            query="contract breach",
            top_k=10,
            fts_language="greek",
        )

        executed_params = mock_cursor.execute.call_args[0][1]
        assert "greek" in executed_params

    def test_empty_query_returns_empty_results(self, store_and_cursor):
        """keyword_search with an empty query should return empty results
        (the DB returns no rows for an empty tsquery)."""
        store, mock_cursor = store_and_cursor
        mock_cursor.fetchall.return_value = []

        results = store.keyword_search(query="", top_k=10)

        assert results == []

    def test_sql_contains_websearch_to_tsquery_in_both_rank_and_where(self, store_and_cursor):
        """websearch_to_tsquery must appear twice: once in ts_rank() for scoring
        and once in the WHERE clause for filtering."""
        store, mock_cursor = store_and_cursor

        store.keyword_search(query="section 4.2", top_k=5)

        executed_sql = mock_cursor.execute.call_args[0][0]
        occurrences = executed_sql.count("websearch_to_tsquery")
        assert occurrences == 2, (
            f"Expected websearch_to_tsquery to appear twice (rank + filter), "
            f"but found {occurrences} occurrence(s)."
        )

    def test_invalid_fts_language_falls_back_to_english(self, store_and_cursor):
        """If an invalid fts_language is supplied, keyword_search must fall back
        to 'english' and still use websearch_to_tsquery."""
        store, mock_cursor = store_and_cursor

        store.keyword_search(
            query="test",
            top_k=5,
            fts_language="klingon",  # not a valid FTS config
        )

        executed_params = mock_cursor.execute.call_args[0][1]
        # Should have fallen back to 'english'
        assert "english" in executed_params
        # Should still use websearch_to_tsquery
        executed_sql = mock_cursor.execute.call_args[0][0]
        assert "websearch_to_tsquery" in executed_sql

    def test_client_id_filter_appended_to_sql(self, store_and_cursor):
        """When client_id is provided, keyword_search must add a client_id filter
        to the SQL while still using websearch_to_tsquery."""
        store, mock_cursor = store_and_cursor

        store.keyword_search(
            query="governing law",
            top_k=5,
            client_id="00000000-0000-0000-0000-000000000001",
        )

        executed_sql = mock_cursor.execute.call_args[0][0]
        assert "client_id" in executed_sql
        assert "websearch_to_tsquery" in executed_sql

    def test_document_id_filter_appended_to_sql(self, store_and_cursor):
        """When document_id is provided, keyword_search must add a document_id filter."""
        store, mock_cursor = store_and_cursor

        store.keyword_search(
            query="liability",
            top_k=5,
            document_id="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        )

        executed_sql = mock_cursor.execute.call_args[0][0]
        assert "document_id" in executed_sql
        assert "websearch_to_tsquery" in executed_sql
