"""
Tests for execution/legal_rag/quotas.py

Covers: TenantQuota, QUOTA_TIERS, QuotaUsage, QuotaExceededError,
        QuotaManager (tier lookup, usage caching, document/query quota
        checks, recording, quota status), and get_quota_manager factory.
"""

import pytest


# ---------------------------------------------------------------------------
# TenantQuota and QUOTA_TIERS
# ---------------------------------------------------------------------------

class TestTenantQuota:
    """Tests for TenantQuota defaults and tier definitions."""

    def test_default_quota_values(self):
        from execution.legal_rag.quotas import TenantQuota
        q = TenantQuota()
        assert q.max_documents == 100
        assert q.max_queries_per_day == 1000
        assert q.max_total_chunks == 50000
        assert q.max_document_size_mb == 50

    def test_all_tiers_defined(self):
        from execution.legal_rag.quotas import QUOTA_TIERS
        assert set(QUOTA_TIERS.keys()) == {"demo", "default", "premium", "enterprise"}

    def test_demo_tier_is_smallest(self):
        from execution.legal_rag.quotas import QUOTA_TIERS
        demo = QUOTA_TIERS["demo"]
        default = QUOTA_TIERS["default"]
        assert demo.max_documents < default.max_documents
        assert demo.max_queries_per_day < default.max_queries_per_day

    def test_enterprise_tier_is_largest(self):
        from execution.legal_rag.quotas import QUOTA_TIERS
        ent = QUOTA_TIERS["enterprise"]
        prem = QUOTA_TIERS["premium"]
        assert ent.max_documents >= prem.max_documents
        assert ent.max_queries_per_day >= prem.max_queries_per_day

    @pytest.mark.parametrize("tier", ["demo", "default", "premium", "enterprise"])
    def test_tier_has_positive_limits(self, tier):
        from execution.legal_rag.quotas import QUOTA_TIERS
        q = QUOTA_TIERS[tier]
        assert q.max_documents > 0
        assert q.max_queries_per_day > 0
        assert q.max_total_chunks > 0
        assert q.max_document_size_mb > 0


# ---------------------------------------------------------------------------
# QuotaUsage
# ---------------------------------------------------------------------------

class TestQuotaUsage:
    """Tests for QuotaUsage defaults."""

    def test_defaults(self):
        from execution.legal_rag.quotas import QuotaUsage
        u = QuotaUsage()
        assert u.document_count == 0
        assert u.chunk_count == 0
        assert u.queries_today == 0
        assert u.queries_this_hour == 0
        assert u.storage_mb == 0


# ---------------------------------------------------------------------------
# QuotaExceededError
# ---------------------------------------------------------------------------

class TestQuotaExceededError:
    """Tests for the QuotaExceededError exception class."""

    def test_attributes_stored(self):
        from execution.legal_rag.quotas import QuotaExceededError
        err = QuotaExceededError(
            "Limit reached", quota_type="documents",
            current=100, limit=100,
        )
        assert err.quota_type == "documents"
        assert err.current == 100
        assert err.limit == 100
        assert "Limit reached" in str(err)

    def test_is_exception(self):
        from execution.legal_rag.quotas import QuotaExceededError
        assert issubclass(QuotaExceededError, Exception)


# ---------------------------------------------------------------------------
# QuotaManager - tier lookup
# ---------------------------------------------------------------------------

class TestQuotaManagerGetQuota:
    """Tests for QuotaManager.get_quota."""

    def test_known_tier(self):
        from execution.legal_rag.quotas import QuotaManager
        manager = QuotaManager()
        q = manager.get_quota("premium")
        assert q.max_documents == 1000

    def test_unknown_tier_returns_default(self):
        from execution.legal_rag.quotas import QuotaManager
        manager = QuotaManager()
        q = manager.get_quota("nonexistent_tier")
        default = manager.get_quota("default")
        assert q.max_documents == default.max_documents


# ---------------------------------------------------------------------------
# QuotaManager - usage without store
# ---------------------------------------------------------------------------

class TestQuotaManagerUsageNoStore:
    """Tests for get_usage when no vector store is attached."""

    def test_returns_empty_usage(self):
        from execution.legal_rag.quotas import QuotaManager, QuotaUsage
        manager = QuotaManager(vector_store=None)
        usage = manager.get_usage("client-123")
        assert isinstance(usage, QuotaUsage)
        assert usage.document_count == 0

    def test_returns_cached_usage(self):
        from execution.legal_rag.quotas import QuotaManager, QuotaUsage
        manager = QuotaManager(vector_store=None)
        manager._usage_cache["client-1"] = QuotaUsage(document_count=5)
        usage = manager.get_usage("client-1")
        assert usage.document_count == 5


# ---------------------------------------------------------------------------
# QuotaManager - check_document_quota
# ---------------------------------------------------------------------------

class TestCheckDocumentQuota:
    """Tests for check_document_quota enforcement."""

    def test_allows_within_limits(self):
        from execution.legal_rag.quotas import QuotaManager, QuotaUsage
        manager = QuotaManager(vector_store=None)
        manager._usage_cache["c1"] = QuotaUsage(document_count=0, chunk_count=0)
        assert manager.check_document_quota("c1", tier="default", new_chunks=10) is True

    def test_raises_on_document_limit(self):
        from execution.legal_rag.quotas import QuotaManager, QuotaUsage, QuotaExceededError
        manager = QuotaManager(vector_store=None)
        manager._usage_cache["c1"] = QuotaUsage(document_count=100)
        with pytest.raises(QuotaExceededError) as exc_info:
            manager.check_document_quota("c1", tier="default")
        assert exc_info.value.quota_type == "documents"

    def test_raises_on_chunks_per_document(self):
        from execution.legal_rag.quotas import QuotaManager, QuotaUsage, QuotaExceededError
        manager = QuotaManager(vector_store=None)
        manager._usage_cache["c1"] = QuotaUsage(document_count=0)
        with pytest.raises(QuotaExceededError) as exc_info:
            manager.check_document_quota("c1", tier="demo", new_chunks=9999)
        assert exc_info.value.quota_type == "chunks_per_document"

    def test_raises_on_total_chunks_exceeded(self):
        from execution.legal_rag.quotas import QuotaManager, QuotaUsage, QuotaExceededError
        manager = QuotaManager(vector_store=None)
        manager._usage_cache["c1"] = QuotaUsage(
            document_count=0, chunk_count=4999
        )
        with pytest.raises(QuotaExceededError) as exc_info:
            manager.check_document_quota("c1", tier="demo", new_chunks=100)
        assert exc_info.value.quota_type == "total_chunks"

    @pytest.mark.parametrize("tier,max_docs", [
        ("demo", 10),
        ("default", 100),
        ("premium", 1000),
        ("enterprise", 10000),
    ])
    def test_tier_specific_document_limits(self, tier, max_docs):
        from execution.legal_rag.quotas import QuotaManager, QuotaUsage, QuotaExceededError
        manager = QuotaManager(vector_store=None)
        manager._usage_cache["c1"] = QuotaUsage(document_count=max_docs)
        with pytest.raises(QuotaExceededError):
            manager.check_document_quota("c1", tier=tier)


# ---------------------------------------------------------------------------
# QuotaManager - check_query_quota
# ---------------------------------------------------------------------------

class TestCheckQueryQuota:
    """Tests for check_query_quota enforcement."""

    def test_allows_within_limits(self):
        from execution.legal_rag.quotas import QuotaManager, QuotaUsage
        manager = QuotaManager(vector_store=None)
        manager._usage_cache["c1"] = QuotaUsage(queries_today=0)
        assert manager.check_query_quota("c1", tier="default") is True

    def test_raises_on_daily_limit(self):
        from execution.legal_rag.quotas import QuotaManager, QuotaUsage, QuotaExceededError
        manager = QuotaManager(vector_store=None)
        manager._usage_cache["c1"] = QuotaUsage(queries_today=100)
        with pytest.raises(QuotaExceededError) as exc_info:
            manager.check_query_quota("c1", tier="demo")
        assert exc_info.value.quota_type == "queries_per_day"


# ---------------------------------------------------------------------------
# QuotaManager - recording
# ---------------------------------------------------------------------------

class TestQuotaManagerRecording:
    """Tests for record_document_upload and record_query."""

    def test_record_document_upload_increments(self):
        from execution.legal_rag.quotas import QuotaManager, QuotaUsage
        manager = QuotaManager(vector_store=None)
        manager._usage_cache["c1"] = QuotaUsage(document_count=5, chunk_count=100)
        manager.record_document_upload("c1", chunk_count=50)
        assert manager._usage_cache["c1"].document_count == 6
        assert manager._usage_cache["c1"].chunk_count == 150

    def test_record_query_increments(self):
        from execution.legal_rag.quotas import QuotaManager, QuotaUsage
        manager = QuotaManager(vector_store=None)
        manager._usage_cache["c1"] = QuotaUsage(queries_today=10)
        manager.record_query("c1")
        assert manager._usage_cache["c1"].queries_today == 11

    def test_record_upload_no_cache_entry_is_noop(self):
        """If client not in cache, recording should not crash."""
        from execution.legal_rag.quotas import QuotaManager
        manager = QuotaManager(vector_store=None)
        # Should not raise
        manager.record_document_upload("unknown-client", chunk_count=10)
        manager.record_query("unknown-client")


# ---------------------------------------------------------------------------
# QuotaManager - quota status
# ---------------------------------------------------------------------------

class TestQuotaStatus:
    """Tests for get_quota_status."""

    def test_status_structure(self):
        from execution.legal_rag.quotas import QuotaManager, QuotaUsage
        manager = QuotaManager(vector_store=None)
        manager._usage_cache["c1"] = QuotaUsage(
            document_count=5, chunk_count=1000, queries_today=50
        )
        status = manager.get_quota_status("c1", tier="default")
        assert "tier" in status
        assert "documents" in status
        assert "chunks" in status
        assert "queries_today" in status
        assert status["tier"] == "default"

    def test_status_percentages(self):
        from execution.legal_rag.quotas import QuotaManager, QuotaUsage
        manager = QuotaManager(vector_store=None)
        manager._usage_cache["c1"] = QuotaUsage(document_count=50)
        status = manager.get_quota_status("c1", tier="default")
        # 50 out of 100 = 50%
        assert status["documents"]["percentage"] == 50.0

    def test_status_remaining(self):
        from execution.legal_rag.quotas import QuotaManager, QuotaUsage
        manager = QuotaManager(vector_store=None)
        manager._usage_cache["c1"] = QuotaUsage(document_count=30)
        status = manager.get_quota_status("c1", tier="default")
        assert status["documents"]["remaining"] == 70


# ---------------------------------------------------------------------------
# get_quota_manager factory
# ---------------------------------------------------------------------------

class TestGetQuotaManager:
    """Tests for the module-level factory function."""

    def test_returns_manager(self):
        from execution.legal_rag.quotas import get_quota_manager, QuotaManager
        m = get_quota_manager()
        assert isinstance(m, QuotaManager)

    def test_attaches_store_if_provided(self):
        from execution.legal_rag.quotas import get_quota_manager
        from unittest.mock import MagicMock
        mock_store = MagicMock()
        m = get_quota_manager(vector_store=mock_store)
        assert m.store is mock_store
