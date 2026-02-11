"""
Dimension 1: FUNCTIONALITY

Tests API correctness, authentication, rate limiting, multi-tenant isolation,
language routing, and quota enforcement.
"""

import sys
import time
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.conftest import (
    ENGLISH_CLIENT_ID, GREEK_CLIENT_ID, FAKE_CLIENT_ID, skip_no_creds,
)

pytestmark = pytest.mark.functionality


# =============================================================================
# 1A. API Endpoint Correctness (unit, mocked services)
# =============================================================================

class TestAPIEndpoints:
    """Verify API endpoints return correct status codes and shapes."""

    @pytest.fixture(autouse=True)
    def _mock_services(self):
        """Patch ServiceContainer so no real DB/API is needed."""
        mock_store = MagicMock()
        mock_store.validate_api_key.return_value = {
            "client_id": ENGLISH_CLIENT_ID,
            "tier": "default",
            "name": "test-key",
        }
        mock_store.list_documents.return_value = []
        mock_store.get_tenant_config.return_value = None

        with patch("execution.legal_rag.api._container") as mock_container:
            mock_container.get_store.return_value = mock_store
            mock_container.get_services.return_value = {
                "retriever": MagicMock(),
                "citation_extractor": MagicMock(),
                "parser": MagicMock(),
                "chunker": MagicMock(),
                "embeddings": MagicMock(),
            }
            mock_container.get_llm_client.return_value = MagicMock()
            self._mock_store = mock_store
            yield

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from execution.legal_rag.api import app
        return TestClient(app)

    def test_health_returns_200(self, client):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_query_empty_string_422(self, client):
        resp = client.post(
            "/api/v1/query",
            json={"query": ""},
            headers={"x-api-key": "test-key"},
        )
        assert resp.status_code == 422

    def test_query_too_long_422(self, client):
        resp = client.post(
            "/api/v1/query",
            json={"query": "x" * 2001},
            headers={"x-api-key": "test-key"},
        )
        assert resp.status_code == 422

    def test_list_documents_returns_array(self, client):
        resp = client.get(
            "/api/v1/documents",
            headers={"x-api-key": "test-key"},
        )
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)


# =============================================================================
# 1B. Authentication & Rate Limiting (unit)
# =============================================================================

class TestAuthentication:
    """Verify API key validation and rate limiting."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from execution.legal_rag.api import app
        return TestClient(app)

    def test_missing_api_key_422(self, client):
        resp = client.post("/api/v1/query", json={"query": "test"})
        assert resp.status_code == 422

    def test_invalid_api_key_401(self, client):
        mock_store = MagicMock()
        mock_store.validate_api_key.return_value = None
        with patch("execution.legal_rag.api._container") as mc:
            mc.get_store.return_value = mock_store
            resp = client.post(
                "/api/v1/query",
                json={"query": "test"},
                headers={"x-api-key": "invalid-key-12345"},
            )
            assert resp.status_code == 401


class TestRateLimiting:
    """Verify the RateLimiter class enforces limits."""

    def test_rate_limit_enforcement(self):
        from execution.legal_rag.api import RateLimiter

        limiter = RateLimiter(max_requests=5, window_seconds=1)
        key = "test-key-rate"

        for i in range(5):
            assert limiter.is_allowed(key), f"Request {i+1} should be allowed"

        assert not limiter.is_allowed(key), "6th request should be blocked"

    def test_rate_limit_resets_after_window(self):
        from execution.legal_rag.api import RateLimiter

        limiter = RateLimiter(max_requests=2, window_seconds=0.5)
        key = "test-key-reset"

        assert limiter.is_allowed(key)
        assert limiter.is_allowed(key)
        assert not limiter.is_allowed(key)

        time.sleep(0.6)
        assert limiter.is_allowed(key), "Should be allowed after window reset"


# =============================================================================
# 1C. Multi-Tenant Isolation (integration)
# =============================================================================

@skip_no_creds
@pytest.mark.integration
class TestMultiTenantIsolation:
    """Verify that tenants cannot see each other's data."""

    def test_english_tenant_cannot_see_greek_data(self, live_store, en_embeddings):
        query_emb = en_embeddings.embed_query("termination clause")
        en_results = live_store.search(query_emb, top_k=5, client_id=ENGLISH_CLIENT_ID)
        el_results = live_store.search(query_emb, top_k=5, client_id=GREEK_CLIENT_ID)

        en_doc_ids = {r.document_id for r in en_results}
        el_doc_ids = {r.document_id for r in el_results}

        assert en_doc_ids.isdisjoint(el_doc_ids), (
            f"EN and EL tenants returned overlapping docs: {en_doc_ids & el_doc_ids}"
        )

    def test_fake_tenant_gets_zero_results(self, live_store, en_embeddings):
        query_emb = en_embeddings.embed_query("contract")
        results = live_store.search(query_emb, top_k=10, client_id=FAKE_CLIENT_ID)
        assert len(results) == 0, f"Fake tenant got {len(results)} results, expected 0"

    def test_list_documents_isolation(self, live_store):
        en_docs = live_store.list_documents(client_id=ENGLISH_CLIENT_ID)
        el_docs = live_store.list_documents(client_id=GREEK_CLIENT_ID)

        en_ids = {str(d["id"]) for d in en_docs}
        el_ids = {str(d["id"]) for d in el_docs}

        assert en_ids.isdisjoint(el_ids), "Tenants should have no overlapping document IDs"
        assert len(en_ids) > 0, "English tenant should have documents"
        assert len(el_ids) > 0, "Greek tenant should have documents"

    def test_keyword_search_isolation(self, live_store):
        results = live_store.keyword_search("defendant", top_k=5, client_id=FAKE_CLIENT_ID)
        assert len(results) == 0, f"Fake tenant keyword search got {len(results)} results"


# =============================================================================
# 1D. Language Routing (unit + integration)
# =============================================================================

class TestLanguageRouting:
    """Verify language configuration and FTS routing."""

    def test_english_config_defaults(self):
        from execution.legal_rag.language_config import TenantLanguageConfig
        config = TenantLanguageConfig.for_language("en")
        assert config.embedding_model == "voyage-law-2"
        assert config.fts_language == "english"
        assert config.chars_per_token == 4

    def test_greek_config_defaults(self):
        from execution.legal_rag.language_config import TenantLanguageConfig
        config = TenantLanguageConfig.for_language("el")
        assert config.embedding_model == "voyage-multilingual-2"
        assert config.fts_language == "greek"
        assert config.chars_per_token == 3

    @skip_no_creds
    @pytest.mark.integration
    def test_greek_fts_returns_results(self, live_store):
        results = live_store.keyword_search(
            "νόμος", top_k=5, client_id=GREEK_CLIENT_ID, fts_language="greek",
        )
        assert len(results) > 0, "Greek FTS for 'νόμος' should return results"

    @skip_no_creds
    @pytest.mark.integration
    def test_english_fts_returns_results(self, live_store):
        results = live_store.keyword_search(
            "defendant", top_k=5, client_id=ENGLISH_CLIENT_ID, fts_language="english",
        )
        assert len(results) > 0, "English FTS for 'defendant' should return results"


# =============================================================================
# 1E. Quota Enforcement (unit)
# =============================================================================

class TestQuotaEnforcement:
    """Verify quota limits are enforced per tier."""

    def test_demo_tier_limits(self):
        from execution.legal_rag.quotas import QuotaManager, QUOTA_TIERS
        tier = QUOTA_TIERS["demo"]
        assert tier.max_documents == 10
        assert tier.max_total_chunks == 5000
        assert tier.max_queries_per_day == 100

    def test_enterprise_tier_higher(self):
        from execution.legal_rag.quotas import QUOTA_TIERS
        demo = QUOTA_TIERS["demo"]
        enterprise = QUOTA_TIERS["enterprise"]
        assert enterprise.max_documents > demo.max_documents
        assert enterprise.max_total_chunks > demo.max_total_chunks
