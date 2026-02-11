"""
Tenant Quota Management for Legal RAG System

Enforces usage limits per client based on subscription tier.
Prevents abuse and enables tiered pricing.
"""

import logging
from dataclasses import dataclass
from typing import Optional
from datetime import date

logger = logging.getLogger(__name__)


@dataclass
class TenantQuota:
    """Quota limits for a tenant."""
    # Document limits
    max_documents: int = 100
    max_chunks_per_document: int = 1000
    max_total_chunks: int = 50000

    # Query limits
    max_queries_per_day: int = 1000
    max_queries_per_hour: int = 100

    # Size limits
    max_document_size_mb: int = 50
    max_total_storage_mb: int = 5000


# Predefined tiers
QUOTA_TIERS = {
    "demo": TenantQuota(
        max_documents=10,
        max_chunks_per_document=500,
        max_total_chunks=5000,
        max_queries_per_day=100,
        max_queries_per_hour=20,
        max_document_size_mb=10,
        max_total_storage_mb=100,
    ),
    "default": TenantQuota(
        max_documents=100,
        max_chunks_per_document=1000,
        max_total_chunks=50000,
        max_queries_per_day=1000,
        max_queries_per_hour=100,
        max_document_size_mb=50,
        max_total_storage_mb=5000,
    ),
    "premium": TenantQuota(
        max_documents=1000,
        max_chunks_per_document=2000,
        max_total_chunks=500000,
        max_queries_per_day=10000,
        max_queries_per_hour=500,
        max_document_size_mb=100,
        max_total_storage_mb=50000,
    ),
    "enterprise": TenantQuota(
        max_documents=10000,
        max_chunks_per_document=5000,
        max_total_chunks=5000000,
        max_queries_per_day=100000,
        max_queries_per_hour=5000,
        max_document_size_mb=200,
        max_total_storage_mb=500000,
    ),
}


@dataclass
class QuotaUsage:
    """Current usage for a tenant."""
    document_count: int = 0
    chunk_count: int = 0
    queries_today: int = 0
    queries_this_hour: int = 0
    storage_mb: float = 0


class QuotaExceededError(Exception):
    """Raised when a quota limit is exceeded."""

    def __init__(self, message: str, quota_type: str, current: int, limit: int):
        super().__init__(message)
        self.quota_type = quota_type
        self.current = current
        self.limit = limit


class QuotaManager:
    """
    Manages tenant quotas and enforces limits.

    Usage:
        manager = QuotaManager(vector_store)

        # Check before document upload
        manager.check_document_quota(client_id, tier="default")

        # Check before query
        manager.check_query_quota(client_id, tier="default")

        # Record usage
        manager.record_document_upload(client_id, chunk_count=150)
        manager.record_query(client_id)
    """

    def __init__(self, vector_store=None):
        """
        Initialize quota manager.

        Args:
            vector_store: VectorStore instance for querying current usage
        """
        self.store = vector_store
        self._usage_cache: dict[str, QuotaUsage] = {}

    def get_quota(self, tier: str = "default") -> TenantQuota:
        """Get quota limits for a tier."""
        return QUOTA_TIERS.get(tier, QUOTA_TIERS["default"])

    def get_usage(self, client_id: str) -> QuotaUsage:
        """
        Get current usage for a tenant.

        Queries the database for actual counts.
        """
        if not self.store:
            return self._usage_cache.get(client_id, QuotaUsage())

        try:
            # Get document count
            docs = self.store.list_documents(client_id=client_id)
            doc_count = len(docs)

            # Get chunk count (from database)
            chunk_count = self._get_chunk_count(client_id)

            # Get daily query count (from usage_daily table)
            queries_today = self._get_queries_today(client_id)

            usage = QuotaUsage(
                document_count=doc_count,
                chunk_count=chunk_count,
                queries_today=queries_today,
            )

            self._usage_cache[client_id] = usage
            return usage

        except Exception as e:
            logger.warning(f"Failed to get usage for {client_id}: {e}")
            return self._usage_cache.get(client_id, QuotaUsage())

    def _get_chunk_count(self, client_id: str) -> int:
        """Get total chunk count for a client."""
        if not self.store:
            return 0

        try:
            with self.store.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT COUNT(*) FROM document_chunks WHERE client_id = %s::uuid",
                        (client_id,)
                    )
                    row = cur.fetchone()
                    return row["count"] if row else 0
        except Exception as e:
            logger.warning(f"Failed to get chunk count for {client_id}: {e}")
            return 0

    def _get_queries_today(self, client_id: str) -> int:
        """Get query count for today from usage_daily table."""
        if not self.store:
            return 0

        try:
            with self.store.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT query_count FROM usage_daily
                        WHERE client_id = %s::uuid AND date = CURRENT_DATE
                        """,
                        (client_id,)
                    )
                    row = cur.fetchone()
                    return row["query_count"] if row else 0
        except Exception as e:
            logger.warning(f"Failed to get query count for {client_id}: {e}")
            return 0

    def check_document_quota(
        self,
        client_id: str,
        tier: str = "default",
        new_chunks: int = 0,
    ) -> bool:
        """
        Check if document upload is allowed.

        Args:
            client_id: Client identifier
            tier: Subscription tier
            new_chunks: Number of chunks in new document

        Returns:
            True if allowed

        Raises:
            QuotaExceededError: If quota would be exceeded
        """
        quota = self.get_quota(tier)
        usage = self.get_usage(client_id)

        # Check document count
        if usage.document_count >= quota.max_documents:
            raise QuotaExceededError(
                f"Document limit reached ({quota.max_documents} documents)",
                quota_type="documents",
                current=usage.document_count,
                limit=quota.max_documents,
            )

        # Check chunk count
        if new_chunks > quota.max_chunks_per_document:
            raise QuotaExceededError(
                f"Document too large ({new_chunks} chunks, max {quota.max_chunks_per_document})",
                quota_type="chunks_per_document",
                current=new_chunks,
                limit=quota.max_chunks_per_document,
            )

        # Check total chunks
        if usage.chunk_count + new_chunks > quota.max_total_chunks:
            raise QuotaExceededError(
                f"Total chunk limit would be exceeded",
                quota_type="total_chunks",
                current=usage.chunk_count,
                limit=quota.max_total_chunks,
            )

        return True

    def check_query_quota(
        self,
        client_id: str,
        tier: str = "default",
    ) -> bool:
        """
        Check if query is allowed.

        Args:
            client_id: Client identifier
            tier: Subscription tier

        Returns:
            True if allowed

        Raises:
            QuotaExceededError: If quota would be exceeded
        """
        quota = self.get_quota(tier)
        usage = self.get_usage(client_id)

        # Check daily limit
        if usage.queries_today >= quota.max_queries_per_day:
            raise QuotaExceededError(
                f"Daily query limit reached ({quota.max_queries_per_day} queries/day)",
                quota_type="queries_per_day",
                current=usage.queries_today,
                limit=quota.max_queries_per_day,
            )

        return True

    def record_document_upload(
        self,
        client_id: str,
        chunk_count: int,
    ):
        """Record a document upload for quota tracking."""
        if client_id in self._usage_cache:
            self._usage_cache[client_id].document_count += 1
            self._usage_cache[client_id].chunk_count += chunk_count

        # Update usage_daily table
        if self.store:
            try:
                with self.store.get_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            INSERT INTO usage_daily (client_id, date, document_count)
                            VALUES (%s::uuid, CURRENT_DATE, 1)
                            ON CONFLICT (client_id, date)
                            DO UPDATE SET document_count = usage_daily.document_count + 1
                            """,
                            (client_id,)
                        )
                        conn.commit()
            except Exception as e:
                logger.warning(f"Failed to update usage: {e}")

    def record_query(self, client_id: str):
        """Record a query for quota tracking."""
        if client_id in self._usage_cache:
            self._usage_cache[client_id].queries_today += 1

        # Update usage_daily table
        if self.store:
            try:
                with self.store.get_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            INSERT INTO usage_daily (client_id, date, query_count)
                            VALUES (%s::uuid, CURRENT_DATE, 1)
                            ON CONFLICT (client_id, date)
                            DO UPDATE SET query_count = usage_daily.query_count + 1
                            """,
                            (client_id,)
                        )
                        conn.commit()
            except Exception as e:
                logger.warning(f"Failed to update usage: {e}")

    def get_quota_status(self, client_id: str, tier: str = "default") -> dict:
        """
        Get detailed quota status for a tenant.

        Returns a dictionary showing current usage vs limits.
        """
        quota = self.get_quota(tier)
        usage = self.get_usage(client_id)

        return {
            "tier": tier,
            "documents": {
                "used": usage.document_count,
                "limit": quota.max_documents,
                "remaining": quota.max_documents - usage.document_count,
                "percentage": round(usage.document_count / quota.max_documents * 100, 1),
            },
            "chunks": {
                "used": usage.chunk_count,
                "limit": quota.max_total_chunks,
                "remaining": quota.max_total_chunks - usage.chunk_count,
                "percentage": round(usage.chunk_count / quota.max_total_chunks * 100, 1),
            },
            "queries_today": {
                "used": usage.queries_today,
                "limit": quota.max_queries_per_day,
                "remaining": quota.max_queries_per_day - usage.queries_today,
                "percentage": round(usage.queries_today / quota.max_queries_per_day * 100, 1),
            },
        }


# Global quota manager instance
_manager = None


def get_quota_manager(vector_store=None) -> QuotaManager:
    """Get the global quota manager instance."""
    global _manager
    if _manager is None:
        _manager = QuotaManager(vector_store)
    elif vector_store and _manager.store is None:
        _manager.store = vector_store
    return _manager


# CLI for testing
if __name__ == "__main__":
    import json

    manager = QuotaManager()

    # Simulate usage
    client_id = "test-client-123"

    print("=== Quota Tiers ===")
    for tier_name, tier in QUOTA_TIERS.items():
        print(f"\n{tier_name.upper()}:")
        print(f"  Documents: {tier.max_documents}")
        print(f"  Queries/day: {tier.max_queries_per_day}")
        print(f"  Total chunks: {tier.max_total_chunks}")

    print("\n=== Checking Quotas ===")

    # Test document quota
    try:
        manager.check_document_quota(client_id, tier="demo", new_chunks=100)
        print("Document upload: ALLOWED")
    except QuotaExceededError as e:
        print(f"Document upload: DENIED - {e}")

    # Test query quota
    try:
        manager.check_query_quota(client_id, tier="demo")
        print("Query: ALLOWED")
    except QuotaExceededError as e:
        print(f"Query: DENIED - {e}")

    # Simulate hitting limit
    print("\n=== Simulating Quota Limit ===")
    manager._usage_cache[client_id] = QuotaUsage(document_count=10)

    try:
        manager.check_document_quota(client_id, tier="demo")
    except QuotaExceededError as e:
        print(f"Expected error: {e}")
        print(f"  Type: {e.quota_type}")
        print(f"  Current: {e.current}")
        print(f"  Limit: {e.limit}")
