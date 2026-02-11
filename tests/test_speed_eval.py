"""
Dimension 5: SPEED

Latency benchmarks (P50/P95) for queries, embeddings, database search,
and end-to-end API response time. Compares English vs Greek.
"""

import sys
import time
import statistics
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.conftest import (
    ENGLISH_CLIENT_ID, GREEK_CLIENT_ID, GREEK_API_KEY, skip_no_creds,
)

pytestmark = [pytest.mark.benchmark, pytest.mark.integration]


# =============================================================================
# Latency SLAs (milliseconds)
# =============================================================================

LATENCY_SLA = {
    "simple": {"p50": 3000, "p95": 6000},
    "factual": {"p50": 5000, "p95": 10000},
    "analytical": {"p50": 8000, "p95": 15000},
}


def _measure_latencies(fn, n_runs=3):
    """Run fn() n_runs times and return list of elapsed times in ms."""
    latencies = []
    for _ in range(n_runs):
        start = time.perf_counter()
        fn()
        latencies.append((time.perf_counter() - start) * 1000)
    return latencies


def _percentile(data, p):
    """Calculate the p-th percentile of data."""
    if not data:
        return 0
    sorted_data = sorted(data)
    idx = int(len(sorted_data) * p / 100)
    idx = min(idx, len(sorted_data) - 1)
    return sorted_data[idx]


# =============================================================================
# Query Latency Benchmarks
# =============================================================================

@skip_no_creds
class TestQueryLatency:
    """Measure and assert query latencies by type."""

    def test_simple_query_latency(self, en_retriever):
        queries = ["contract", "NDA", "breach"]
        all_latencies = []
        for q in queries:
            lats = _measure_latencies(
                lambda q=q: en_retriever.retrieve(q, client_id=ENGLISH_CLIENT_ID, top_k=5, use_cache=False),
                n_runs=2,
            )
            all_latencies.extend(lats)

        p50 = statistics.median(all_latencies)
        sla = LATENCY_SLA["simple"]
        assert p50 < sla["p50"], f"Simple P50={p50:.0f}ms exceeds SLA {sla['p50']}ms"

    def test_factual_query_latency(self, en_retriever):
        queries = ["What is the termination clause?", "Who are the parties?"]
        all_latencies = []
        for q in queries:
            lats = _measure_latencies(
                lambda q=q: en_retriever.retrieve(q, client_id=ENGLISH_CLIENT_ID, top_k=5, use_cache=False),
                n_runs=2,
            )
            all_latencies.extend(lats)

        p50 = statistics.median(all_latencies)
        sla = LATENCY_SLA["factual"]
        assert p50 < sla["p50"], f"Factual P50={p50:.0f}ms exceeds SLA {sla['p50']}ms"

    def test_analytical_query_latency(self, en_retriever):
        lats = _measure_latencies(
            lambda: en_retriever.retrieve(
                "Explain the implications of the indemnification clause",
                client_id=ENGLISH_CLIENT_ID, top_k=5, use_cache=False,
            ),
            n_runs=2,
        )
        p50 = statistics.median(lats)
        sla = LATENCY_SLA["analytical"]
        assert p50 < sla["p50"], f"Analytical P50={p50:.0f}ms exceeds SLA {sla['p50']}ms"


# =============================================================================
# Embedding Throughput
# =============================================================================

@skip_no_creds
class TestEmbeddingThroughput:
    """Measure embedding generation speed."""

    def test_batch_embedding_throughput(self, en_embeddings):
        texts = ["This is a test legal document about contract termination and liability."] * 50
        start = time.perf_counter()
        embeddings = en_embeddings.embed_documents(texts)
        elapsed = time.perf_counter() - start

        assert len(embeddings) == 50
        throughput = 50 / elapsed
        assert throughput > 3, f"Throughput {throughput:.1f} texts/sec too low (expected >3)"

    def test_single_query_embedding_speed(self, en_embeddings):
        start = time.perf_counter()
        emb = en_embeddings.embed_query("termination clause")
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert len(emb) == 1024
        assert elapsed_ms < 1000, f"Query embedding took {elapsed_ms:.0f}ms (expected < 1000ms)"


# =============================================================================
# Database Search Latency
# =============================================================================

@skip_no_creds
class TestDatabaseSearchLatency:
    """Measure raw database search performance."""

    def test_vector_search_under_1s(self, live_store, en_embeddings):
        q_emb = en_embeddings.embed_query("termination clause")
        start = time.perf_counter()
        results = live_store.search(q_emb, top_k=10, client_id=ENGLISH_CLIENT_ID)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 1000, f"EN vector search took {elapsed_ms:.0f}ms (SLA: 1000ms)"

    def test_keyword_search_under_500ms(self, live_store):
        start = time.perf_counter()
        results = live_store.keyword_search("defendant", top_k=10, client_id=ENGLISH_CLIENT_ID)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 500, f"EN keyword search took {elapsed_ms:.0f}ms (SLA: 500ms)"

    def test_greek_vector_search_under_2s(self, live_store, el_embeddings):
        q_emb = el_embeddings.embed_query("νόμος ενοικιοστασίου")
        start = time.perf_counter()
        results = live_store.search(q_emb, top_k=10, client_id=GREEK_CLIENT_ID)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 2000, f"EL vector search on ~94K chunks took {elapsed_ms:.0f}ms (SLA: 2000ms)"


# =============================================================================
# End-to-End & Comparison
# =============================================================================

@skip_no_creds
class TestEndToEndLatency:
    """Measure full pipeline latency and compare EN vs EL."""

    def test_english_vs_greek_latency_ratio(self, en_retriever, el_retriever):
        en_start = time.perf_counter()
        en_retriever.retrieve("termination", client_id=ENGLISH_CLIENT_ID, top_k=5, use_cache=False)
        en_ms = (time.perf_counter() - en_start) * 1000

        el_start = time.perf_counter()
        el_retriever.retrieve("τερματισμός", client_id=GREEK_CLIENT_ID, top_k=5, use_cache=False)
        el_ms = (time.perf_counter() - el_start) * 1000

        ratio = el_ms / max(en_ms, 1)
        assert ratio < 5.0, (
            f"Greek is {ratio:.1f}x slower than English ({el_ms:.0f}ms vs {en_ms:.0f}ms). "
            f"Expected < 5x."
        )

    def test_api_e2e_under_30s(self, live_api_client):
        start = time.perf_counter()
        response = live_api_client.post(
            "/api/v1/query",
            headers={"x-api-key": GREEK_API_KEY},
            json={"query": "What is the termination clause?", "top_k": 3},
        )
        elapsed = time.perf_counter() - start

        assert response.status_code == 200, f"API returned {response.status_code}: {response.text[:200]}"
        assert elapsed < 30.0, f"End-to-end API query took {elapsed:.1f}s (SLA: 30s)"
