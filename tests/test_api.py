"""Tests for the FastAPI backend endpoints."""

import os
import json
import hashlib
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

# Set up environment before importing FastAPI app
os.environ.setdefault("NVIDIA_API_KEY", "test-key")
os.environ.setdefault("COHERE_API_KEY", "test-key")

from fastapi.testclient import TestClient

from execution.legal_rag.api_models import (
    QueryRequest, QueryResponse, DocumentInfo,
    TenantConfigUpdate, TenantConfigResponse, HealthResponse,
)
from execution.legal_rag.language_config import TenantLanguageConfig


# ---------------------------------------------------------------------------
# Mock the ServiceContainer so no real DB or API is needed
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_store():
    """Create a mock VectorStore."""
    store = MagicMock()
    store.validate_api_key.return_value = {
        "client_id": "00000000-0000-0000-0000-000000000001",
        "tier": "default",
        "name": "Test Key",
    }
    store.get_tenant_config.return_value = None
    store.list_documents.return_value = [
        {
            "id": "doc-001",
            "title": "Test Contract",
            "document_type": "contract",
            "jurisdiction": "Delaware",
            "page_count": 10,
            "created_at": "2024-01-01T00:00:00",
        }
    ]
    store.connect.return_value = None
    store.initialize_schema.return_value = None
    store.initialize_auth_schema.return_value = None
    store.initialize_tenant_config_schema.return_value = None
    store.log_audit.return_value = None
    store.delete_document.return_value = True
    store.set_tenant_config.return_value = None
    store.create_greek_fts_index.return_value = None
    return store


@pytest.fixture
def client(mock_store):
    """Create a TestClient with mocked services."""
    from execution.legal_rag import api

    # Replace the container's store
    api._container._store = mock_store
    api._container._services = {}

    # We need to mock the services too so they don't try to load real models
    mock_services = {
        "parser": MagicMock(),
        "chunker": MagicMock(),
        "embeddings": MagicMock(),
        "retriever": MagicMock(),
        "citation_extractor": MagicMock(),
    }

    # Make retriever.retrieve return empty list by default
    mock_services["retriever"].retrieve.return_value = []
    mock_services["citation_extractor"].extract.return_value = []

    # Cache mock services for English
    api._container._services["en"] = mock_services

    return TestClient(api.app)


VALID_API_KEY = "lrag_test_key_12345"


class TestHealthEndpoint:
    def test_health_check(self, client):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["version"] == "0.2.0"
        assert data["database"] == "connected"


class TestAuthMiddleware:
    def test_missing_api_key(self, client, mock_store):
        response = client.get("/api/v1/documents")
        assert response.status_code == 422  # Missing required header

    def test_invalid_api_key(self, client, mock_store):
        mock_store.validate_api_key.return_value = None
        response = client.get(
            "/api/v1/documents",
            headers={"x-api-key": "invalid_key"},
        )
        assert response.status_code == 401

    def test_valid_api_key(self, client, mock_store):
        response = client.get(
            "/api/v1/documents",
            headers={"x-api-key": VALID_API_KEY},
        )
        assert response.status_code == 200


class TestDocumentsEndpoint:
    def test_list_documents(self, client, mock_store):
        response = client.get(
            "/api/v1/documents",
            headers={"x-api-key": VALID_API_KEY},
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["title"] == "Test Contract"

    def test_delete_document(self, client, mock_store):
        response = client.delete(
            "/api/v1/documents/doc-001",
            headers={"x-api-key": VALID_API_KEY},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "deleted"
        mock_store.delete_document.assert_called_once_with(
            "doc-001", client_id="00000000-0000-0000-0000-000000000001"
        )


class TestQueryEndpoint:
    def test_query_no_results(self, client, mock_store):
        response = client.post(
            "/api/v1/query",
            headers={"x-api-key": VALID_API_KEY},
            json={"query": "What is the termination clause?"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "No relevant information" in data["answer"]
        assert data["sources"] == []

    def test_query_with_results(self, client, mock_store):
        from execution.legal_rag.vector_store import SearchResult
        from execution.legal_rag.citation import CitedContent, Citation
        from execution.legal_rag import api

        # Setup mock results
        mock_result = SearchResult(
            chunk_id="chunk-001",
            document_id="doc-001",
            content="The contract may be terminated with 60 days notice.",
            section_title="Termination",
            hierarchy_path="Doc/Termination/part_0",
            page_numbers=[5],
            score=0.92,
            metadata={"level": 2, "context_before": "", "context_after": ""},
        )

        mock_citation = Citation(
            document_title="Test Contract",
            section="Section 4.2",
            page_numbers=[5],
            hierarchy_path="Doc/Termination/part_0",
            chunk_id="chunk-001",
            document_id="doc-001",
            relevance_score=0.92,
        )

        mock_cited = CitedContent(
            content="The contract may be terminated with 60 days notice.",
            citation=mock_citation,
            context_before="",
            context_after="",
        )

        services = api._container._services["en"]
        services["retriever"].retrieve.return_value = [mock_result]
        services["citation_extractor"].extract.return_value = [mock_cited]

        # Mock the LLM call (OpenAI is imported locally inside the endpoint)
        with patch("openai.OpenAI") as mock_openai:
            mock_llm = MagicMock()
            mock_openai.return_value = mock_llm
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "The contract can be terminated with 60 days notice [1]."
            mock_llm.chat.completions.create.return_value = mock_response

            response = client.post(
                "/api/v1/query",
                headers={"x-api-key": VALID_API_KEY},
                json={"query": "What is the termination clause?"},
            )

        assert response.status_code == 200
        data = response.json()
        assert len(data["sources"]) == 1
        assert data["sources"][0]["document_title"] == "Test Contract"
        assert "content" in data["sources"][0]
        assert data["sources"][0]["content"] == "The contract may be terminated with 60 days notice."

    def test_query_validation(self, client, mock_store):
        # Empty query
        response = client.post(
            "/api/v1/query",
            headers={"x-api-key": VALID_API_KEY},
            json={"query": ""},
        )
        assert response.status_code == 422


class TestConfigEndpoint:
    def test_get_config_default(self, client, mock_store):
        response = client.get(
            "/api/v1/config",
            headers={"x-api-key": VALID_API_KEY},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["language"] == "en"
        assert data["fts_language"] == "english"

    def test_update_config_to_greek(self, client, mock_store):
        response = client.put(
            "/api/v1/config",
            headers={"x-api-key": VALID_API_KEY},
            json={"language": "el"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["language"] == "el"
        assert data["fts_language"] == "greek"
        assert data["embedding_model"] == "voyage-multilingual-2"
        mock_store.set_tenant_config.assert_called_once()
        mock_store.create_greek_fts_index.assert_called_once()

    def test_update_config_invalid_language(self, client, mock_store):
        response = client.put(
            "/api/v1/config",
            headers={"x-api-key": VALID_API_KEY},
            json={"language": "xx"},
        )
        assert response.status_code == 422
