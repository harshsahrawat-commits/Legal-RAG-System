"""
Metrics Collection for Legal RAG System

Tracks performance, usage, and health metrics for monitoring and optimization.
"""

import time
import logging
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class QueryMetrics:
    """Metrics for a single query."""
    query_id: str
    client_id: str
    query_text: str
    start_time: float
    end_time: float = 0
    latency_ms: float = 0
    results_count: int = 0
    cache_hit: bool = False
    rerank_skipped: bool = False
    error: Optional[str] = None


@dataclass
class SystemMetrics:
    """Aggregated system metrics."""
    # Query metrics
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0

    # Latency tracking (in ms)
    total_latency_ms: float = 0
    min_latency_ms: float = float('inf')
    max_latency_ms: float = 0
    latencies: list = field(default_factory=list)

    # Cache metrics
    cache_hits: int = 0
    cache_misses: int = 0

    # Reranking metrics
    rerank_calls: int = 0
    rerank_skipped: int = 0

    # Ingestion metrics
    documents_ingested: int = 0
    chunks_created: int = 0
    total_ingestion_time_ms: float = 0

    # Error tracking
    errors_by_type: dict = field(default_factory=lambda: defaultdict(int))

    # Per-tenant tracking
    queries_by_tenant: dict = field(default_factory=lambda: defaultdict(int))
    documents_by_tenant: dict = field(default_factory=lambda: defaultdict(int))

    @property
    def avg_latency_ms(self) -> float:
        """Calculate average query latency."""
        if self.total_queries == 0:
            return 0
        return self.total_latency_ms / self.total_queries

    @property
    def p95_latency_ms(self) -> float:
        """Calculate 95th percentile latency."""
        if not self.latencies:
            return 0
        sorted_latencies = sorted(self.latencies)
        index = int(len(sorted_latencies) * 0.95)
        return sorted_latencies[min(index, len(sorted_latencies) - 1)]

    @property
    def p99_latency_ms(self) -> float:
        """Calculate 99th percentile latency."""
        if not self.latencies:
            return 0
        sorted_latencies = sorted(self.latencies)
        index = int(len(sorted_latencies) * 0.99)
        return sorted_latencies[min(index, len(sorted_latencies) - 1)]

    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0
        return self.cache_hits / total

    @property
    def rerank_skip_rate(self) -> float:
        """Calculate rerank skip rate (cost savings)."""
        total = self.rerank_calls + self.rerank_skipped
        if total == 0:
            return 0
        return self.rerank_skipped / total

    @property
    def error_rate(self) -> float:
        """Calculate error rate."""
        if self.total_queries == 0:
            return 0
        return self.failed_queries / self.total_queries

    def to_dict(self) -> dict:
        """Convert to dictionary for display."""
        return {
            "queries": {
                "total": self.total_queries,
                "successful": self.successful_queries,
                "failed": self.failed_queries,
                "error_rate": f"{self.error_rate:.2%}",
            },
            "latency_ms": {
                "avg": round(self.avg_latency_ms, 2),
                "min": round(self.min_latency_ms, 2) if self.min_latency_ms != float('inf') else 0,
                "max": round(self.max_latency_ms, 2),
                "p95": round(self.p95_latency_ms, 2),
                "p99": round(self.p99_latency_ms, 2),
            },
            "cache": {
                "hits": self.cache_hits,
                "misses": self.cache_misses,
                "hit_rate": f"{self.cache_hit_rate:.2%}",
            },
            "reranking": {
                "calls": self.rerank_calls,
                "skipped": self.rerank_skipped,
                "skip_rate": f"{self.rerank_skip_rate:.2%}",
                "cost_savings": f"{self.rerank_skip_rate:.0%}",
            },
            "ingestion": {
                "documents": self.documents_ingested,
                "chunks": self.chunks_created,
                "avg_time_ms": round(
                    self.total_ingestion_time_ms / max(self.documents_ingested, 1), 2
                ),
            },
            "errors": dict(self.errors_by_type),
        }


class MetricsCollector:
    """
    Collects and aggregates system metrics.

    Usage:
        collector = MetricsCollector()

        # Track a query
        with collector.track_query(client_id, query_text) as tracker:
            results = retriever.retrieve(query_text)
            tracker.set_results(len(results), cache_hit=False)

        # Get metrics
        metrics = collector.get_metrics()
    """

    _instance = None

    def __new__(cls):
        """Singleton pattern for global metrics collection."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.metrics = SystemMetrics()
        self._query_history: list[QueryMetrics] = []
        self._max_history = 1000  # Keep last 1000 queries
        self._start_time = datetime.now()
        self._initialized = True

    def reset(self):
        """Reset all metrics (for testing)."""
        self.metrics = SystemMetrics()
        self._query_history = []
        self._start_time = datetime.now()

    class QueryTracker:
        """Context manager for tracking query metrics."""

        def __init__(self, collector: 'MetricsCollector', client_id: str, query_text: str):
            self.collector = collector
            self.query = QueryMetrics(
                query_id=f"q_{int(time.time() * 1000)}",
                client_id=client_id,
                query_text=query_text[:200],  # Truncate for storage
                start_time=time.time(),
            )

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.query.end_time = time.time()
            self.query.latency_ms = (self.query.end_time - self.query.start_time) * 1000

            if exc_type:
                self.query.error = str(exc_val)
                self.collector._record_error(exc_type.__name__)

            self.collector._record_query(self.query)
            return False  # Don't suppress exceptions

        def set_results(
            self,
            count: int,
            cache_hit: bool = False,
            rerank_skipped: bool = False
        ):
            """Set query result metadata."""
            self.query.results_count = count
            self.query.cache_hit = cache_hit
            self.query.rerank_skipped = rerank_skipped

    def track_query(self, client_id: str, query_text: str) -> QueryTracker:
        """
        Create a query tracker context manager.

        Usage:
            with collector.track_query(client_id, query) as tracker:
                results = do_search()
                tracker.set_results(len(results))
        """
        return self.QueryTracker(self, client_id, query_text)

    def _record_query(self, query: QueryMetrics):
        """Record completed query metrics."""
        self.metrics.total_queries += 1

        if query.error:
            self.metrics.failed_queries += 1
        else:
            self.metrics.successful_queries += 1

        # Latency tracking
        self.metrics.total_latency_ms += query.latency_ms
        self.metrics.min_latency_ms = min(self.metrics.min_latency_ms, query.latency_ms)
        self.metrics.max_latency_ms = max(self.metrics.max_latency_ms, query.latency_ms)
        self.metrics.latencies.append(query.latency_ms)

        # Keep latencies list bounded
        if len(self.metrics.latencies) > self._max_history:
            self.metrics.latencies = self.metrics.latencies[-self._max_history:]

        # Cache tracking
        if query.cache_hit:
            self.metrics.cache_hits += 1
        else:
            self.metrics.cache_misses += 1

        # Rerank tracking
        if query.rerank_skipped:
            self.metrics.rerank_skipped += 1
        else:
            self.metrics.rerank_calls += 1

        # Per-tenant tracking
        self.metrics.queries_by_tenant[query.client_id] += 1

        # Query history
        self._query_history.append(query)
        if len(self._query_history) > self._max_history:
            self._query_history = self._query_history[-self._max_history:]

    def _record_error(self, error_type: str):
        """Record an error by type."""
        self.metrics.errors_by_type[error_type] += 1

    def record_ingestion(
        self,
        client_id: str,
        document_id: str,
        chunks_count: int,
        duration_ms: float
    ):
        """Record document ingestion metrics."""
        self.metrics.documents_ingested += 1
        self.metrics.chunks_created += chunks_count
        self.metrics.total_ingestion_time_ms += duration_ms
        self.metrics.documents_by_tenant[client_id] += 1

    def record_cache_hit(self):
        """Record a cache hit (for embedding cache, etc.)."""
        self.metrics.cache_hits += 1

    def record_cache_miss(self):
        """Record a cache miss."""
        self.metrics.cache_misses += 1

    def get_metrics(self) -> SystemMetrics:
        """Get current metrics."""
        return self.metrics

    def get_metrics_dict(self) -> dict:
        """Get metrics as a dictionary."""
        return self.metrics.to_dict()

    def get_recent_queries(self, limit: int = 10) -> list[QueryMetrics]:
        """Get most recent queries."""
        return self._query_history[-limit:]

    def get_uptime(self) -> timedelta:
        """Get system uptime."""
        return datetime.now() - self._start_time

    def get_tenant_summary(self) -> dict:
        """Get per-tenant summary."""
        return {
            "queries_by_tenant": dict(self.metrics.queries_by_tenant),
            "documents_by_tenant": dict(self.metrics.documents_by_tenant),
        }


# Global metrics collector instance
_collector = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    global _collector
    if _collector is None:
        _collector = MetricsCollector()
    return _collector


# CLI for testing
if __name__ == "__main__":
    import random

    collector = get_metrics_collector()

    # Simulate some queries
    for i in range(20):
        client = f"client_{random.randint(1, 3)}"
        query = f"Test query {i}"

        with collector.track_query(client, query) as tracker:
            # Simulate work
            time.sleep(random.uniform(0.01, 0.1))
            tracker.set_results(
                count=random.randint(1, 10),
                cache_hit=random.random() > 0.7,
                rerank_skipped=random.random() > 0.6,
            )

    # Simulate an error
    try:
        with collector.track_query("client_1", "error query") as tracker:
            raise ValueError("Simulated error")
    except ValueError:
        pass

    # Print metrics
    import json
    print("\n=== System Metrics ===")
    print(json.dumps(collector.get_metrics_dict(), indent=2))

    print(f"\n=== Uptime: {collector.get_uptime()} ===")

    print("\n=== Tenant Summary ===")
    print(json.dumps(collector.get_tenant_summary(), indent=2))
