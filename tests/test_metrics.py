"""
Tests for execution/legal_rag/metrics.py

Covers: QueryMetrics, SystemMetrics (aggregation properties),
        MetricsCollector singleton, QueryTracker context manager,
        and the get_metrics_collector factory.
"""

import time
from datetime import datetime, timedelta
from collections import defaultdict

import pytest


# ---------------------------------------------------------------------------
# SystemMetrics aggregation properties
# ---------------------------------------------------------------------------

class TestSystemMetrics:
    """Tests for SystemMetrics computed properties."""

    def test_avg_latency_zero_queries(self):
        from execution.legal_rag.metrics import SystemMetrics
        m = SystemMetrics()
        assert m.avg_latency_ms == 0

    def test_avg_latency_with_queries(self):
        from execution.legal_rag.metrics import SystemMetrics
        m = SystemMetrics(total_queries=4, total_latency_ms=400.0)
        assert m.avg_latency_ms == 100.0

    def test_p95_latency_empty(self):
        from execution.legal_rag.metrics import SystemMetrics
        m = SystemMetrics()
        assert m.p95_latency_ms == 0

    def test_p95_latency(self):
        from execution.legal_rag.metrics import SystemMetrics
        m = SystemMetrics(latencies=list(range(1, 101)))  # 1..100
        p95 = m.p95_latency_ms
        assert p95 >= 95

    def test_p99_latency(self):
        from execution.legal_rag.metrics import SystemMetrics
        m = SystemMetrics(latencies=list(range(1, 101)))
        p99 = m.p99_latency_ms
        assert p99 >= 99

    def test_cache_hit_rate_zero(self):
        from execution.legal_rag.metrics import SystemMetrics
        m = SystemMetrics()
        assert m.cache_hit_rate == 0

    def test_cache_hit_rate_calculation(self):
        from execution.legal_rag.metrics import SystemMetrics
        m = SystemMetrics(cache_hits=3, cache_misses=7)
        assert abs(m.cache_hit_rate - 0.3) < 1e-6

    def test_rerank_skip_rate_zero(self):
        from execution.legal_rag.metrics import SystemMetrics
        m = SystemMetrics()
        assert m.rerank_skip_rate == 0

    def test_rerank_skip_rate_calculation(self):
        from execution.legal_rag.metrics import SystemMetrics
        m = SystemMetrics(rerank_calls=6, rerank_skipped=4)
        assert abs(m.rerank_skip_rate - 0.4) < 1e-6

    def test_error_rate_zero(self):
        from execution.legal_rag.metrics import SystemMetrics
        m = SystemMetrics()
        assert m.error_rate == 0

    def test_error_rate_calculation(self):
        from execution.legal_rag.metrics import SystemMetrics
        m = SystemMetrics(total_queries=10, failed_queries=2)
        assert abs(m.error_rate - 0.2) < 1e-6

    def test_to_dict_structure(self):
        from execution.legal_rag.metrics import SystemMetrics
        m = SystemMetrics(total_queries=5, successful_queries=4, failed_queries=1)
        d = m.to_dict()
        assert "queries" in d
        assert "latency_ms" in d
        assert "cache" in d
        assert "reranking" in d
        assert "ingestion" in d
        assert "errors" in d

    def test_min_latency_infinity_renders_as_zero(self):
        from execution.legal_rag.metrics import SystemMetrics
        m = SystemMetrics()  # min_latency_ms defaults to inf
        d = m.to_dict()
        assert d["latency_ms"]["min"] == 0


# ---------------------------------------------------------------------------
# MetricsCollector singleton
# ---------------------------------------------------------------------------

class TestMetricsCollectorSingleton:
    """Test that MetricsCollector uses singleton pattern."""

    def test_singleton_returns_same_instance(self):
        from execution.legal_rag.metrics import MetricsCollector
        a = MetricsCollector()
        b = MetricsCollector()
        assert a is b

    def test_reset_clears_state(self):
        from execution.legal_rag.metrics import MetricsCollector
        collector = MetricsCollector()
        collector.metrics.total_queries = 99
        collector.reset()
        assert collector.metrics.total_queries == 0


# ---------------------------------------------------------------------------
# QueryTracker context manager
# ---------------------------------------------------------------------------

class TestQueryTracker:
    """Tests for the QueryTracker context manager."""

    def test_successful_query_tracking(self):
        from execution.legal_rag.metrics import MetricsCollector
        collector = MetricsCollector()
        collector.reset()

        with collector.track_query("client-1", "test query") as tracker:
            tracker.set_results(5, cache_hit=False, rerank_skipped=True)

        m = collector.get_metrics()
        assert m.total_queries == 1
        assert m.successful_queries == 1
        assert m.failed_queries == 0
        assert m.rerank_skipped == 1

    def test_failed_query_tracking(self):
        from execution.legal_rag.metrics import MetricsCollector
        collector = MetricsCollector()
        collector.reset()

        with pytest.raises(ValueError):
            with collector.track_query("client-1", "bad query") as tracker:
                raise ValueError("test error")

        m = collector.get_metrics()
        assert m.total_queries == 1
        assert m.failed_queries == 1
        assert m.errors_by_type["ValueError"] == 1

    def test_latency_tracked(self):
        from execution.legal_rag.metrics import MetricsCollector
        collector = MetricsCollector()
        collector.reset()

        with collector.track_query("client-1", "query") as tracker:
            time.sleep(0.01)  # Small delay to ensure measurable latency
            tracker.set_results(1)

        m = collector.get_metrics()
        assert m.total_latency_ms > 0
        assert m.min_latency_ms > 0
        assert m.max_latency_ms > 0

    def test_cache_hit_tracked(self):
        from execution.legal_rag.metrics import MetricsCollector
        collector = MetricsCollector()
        collector.reset()

        with collector.track_query("c1", "q") as t:
            t.set_results(1, cache_hit=True)

        assert collector.get_metrics().cache_hits == 1

    def test_per_tenant_tracking(self):
        from execution.legal_rag.metrics import MetricsCollector
        collector = MetricsCollector()
        collector.reset()

        with collector.track_query("client-a", "q1") as t:
            t.set_results(1)
        with collector.track_query("client-a", "q2") as t:
            t.set_results(1)
        with collector.track_query("client-b", "q3") as t:
            t.set_results(1)

        m = collector.get_metrics()
        assert m.queries_by_tenant["client-a"] == 2
        assert m.queries_by_tenant["client-b"] == 1

    def test_query_text_truncated(self):
        """Long query text should be truncated to 200 chars."""
        from execution.legal_rag.metrics import MetricsCollector
        collector = MetricsCollector()
        collector.reset()

        long_query = "x" * 500
        with collector.track_query("c1", long_query) as t:
            t.set_results(0)

        recent = collector.get_recent_queries(1)
        assert len(recent[0].query_text) == 200


# ---------------------------------------------------------------------------
# MetricsCollector - record_ingestion
# ---------------------------------------------------------------------------

class TestRecordIngestion:
    """Tests for record_ingestion."""

    def test_ingestion_tracked(self):
        from execution.legal_rag.metrics import MetricsCollector
        collector = MetricsCollector()
        collector.reset()

        collector.record_ingestion("client-1", "doc-1", chunks_count=50, duration_ms=1500)
        m = collector.get_metrics()
        assert m.documents_ingested == 1
        assert m.chunks_created == 50
        assert m.total_ingestion_time_ms == 1500
        assert m.documents_by_tenant["client-1"] == 1


# ---------------------------------------------------------------------------
# MetricsCollector - cache recording
# ---------------------------------------------------------------------------

class TestCacheRecording:
    """Tests for record_cache_hit / record_cache_miss."""

    def test_cache_hit_recorded(self):
        from execution.legal_rag.metrics import MetricsCollector
        collector = MetricsCollector()
        collector.reset()

        collector.record_cache_hit()
        collector.record_cache_hit()
        collector.record_cache_miss()
        m = collector.get_metrics()
        assert m.cache_hits == 2
        assert m.cache_misses == 1


# ---------------------------------------------------------------------------
# MetricsCollector - helper methods
# ---------------------------------------------------------------------------

class TestMetricsCollectorHelpers:
    """Tests for get_recent_queries, get_uptime, get_tenant_summary."""

    def test_get_recent_queries(self):
        from execution.legal_rag.metrics import MetricsCollector
        collector = MetricsCollector()
        collector.reset()

        for i in range(5):
            with collector.track_query("c1", f"query_{i}") as t:
                t.set_results(1)

        recent = collector.get_recent_queries(3)
        assert len(recent) == 3
        assert recent[-1].query_text == "query_4"

    def test_get_uptime(self):
        from execution.legal_rag.metrics import MetricsCollector
        collector = MetricsCollector()
        uptime = collector.get_uptime()
        assert isinstance(uptime, timedelta)
        assert uptime.total_seconds() >= 0

    def test_get_tenant_summary(self):
        from execution.legal_rag.metrics import MetricsCollector
        collector = MetricsCollector()
        collector.reset()

        collector.record_ingestion("c1", "d1", 10, 100)
        with collector.track_query("c1", "q") as t:
            t.set_results(1)

        summary = collector.get_tenant_summary()
        assert "queries_by_tenant" in summary
        assert "documents_by_tenant" in summary
        assert summary["queries_by_tenant"]["c1"] == 1
        assert summary["documents_by_tenant"]["c1"] == 1


# ---------------------------------------------------------------------------
# get_metrics_collector factory
# ---------------------------------------------------------------------------

class TestGetMetricsCollector:
    """Tests for the module-level factory function."""

    def test_returns_collector(self):
        from execution.legal_rag.metrics import get_metrics_collector, MetricsCollector
        c = get_metrics_collector()
        assert isinstance(c, MetricsCollector)

    def test_returns_same_instance(self):
        from execution.legal_rag.metrics import get_metrics_collector
        a = get_metrics_collector()
        b = get_metrics_collector()
        assert a is b
