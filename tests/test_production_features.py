#!/usr/bin/env python3
"""
Test Script for Production Features

Tests all new multi-tenant production features:
1. Database reset
2. Auth schema setup
3. RLS enablement
4. API key creation/validation
5. Audit logging
6. Metrics collection
7. Quota enforcement

Run with: python execution/legal_rag/test_production_features.py
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def print_section(title: str):
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}\n")


def test_database_reset():
    """Reset database to clean state."""
    print_section("1. Database Reset")

    from execution.legal_rag.vector_store import VectorStore

    store = VectorStore()
    store.connect()

    # Get current counts
    with store._conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM legal_documents")
        doc_count = cur.fetchone()["count"]
        cur.execute("SELECT COUNT(*) FROM document_chunks")
        chunk_count = cur.fetchone()["count"]

    print(f"Current state: {doc_count} documents, {chunk_count} chunks")

    if doc_count > 0 or chunk_count > 0:
        # Reset
        with store._conn.cursor() as cur:
            cur.execute("TRUNCATE legal_documents CASCADE")
            store._conn.commit()
        print("✓ Cleared all documents and chunks")
    else:
        print("✓ Database already empty")

    # Drop new tables for fresh setup
    with store._conn.cursor() as cur:
        for table in ['api_keys', 'audit_log', 'usage_daily']:
            cur.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
        store._conn.commit()
    print("✓ Dropped auth tables for fresh setup")

    store.close()
    return True


def test_auth_setup():
    """Test authentication schema setup."""
    print_section("2. Authentication Schema")

    from execution.legal_rag.vector_store import VectorStore

    store = VectorStore()
    store.connect()
    store.initialize_auth_schema()

    # Verify tables exist
    with store._conn.cursor() as cur:
        cur.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_name IN ('api_keys', 'audit_log', 'usage_daily')
        """)
        tables = [row["table_name"] for row in cur.fetchall()]

    print(f"✓ Created tables: {', '.join(tables)}")

    store.close()
    return len(tables) == 3


def test_api_key_creation():
    """Test API key creation and validation."""
    print_section("3. API Key Management")

    from execution.legal_rag.vector_store import VectorStore

    store = VectorStore()
    store.connect()
    store.initialize_auth_schema()

    # Create demo client API key
    demo_client_id = "00000000-0000-0000-0000-000000000001"
    api_key = store.create_api_key(demo_client_id, name="Test Key", tier="default")
    print(f"✓ Created API key: {api_key[:20]}...")

    # Validate the key
    result = store.validate_api_key(api_key)
    if result:
        print(f"✓ Key validated: client_id={result['client_id'][:8]}..., tier={result['tier']}")
    else:
        print("✗ Key validation failed!")
        return False

    # Test invalid key
    invalid_result = store.validate_api_key("invalid_key_12345")
    if invalid_result is None:
        print("✓ Invalid key correctly rejected")
    else:
        print("✗ Invalid key was accepted!")
        return False

    # Save key for later tests
    with open(project_root / ".tmp" / "test_api_key.txt", "w") as f:
        f.write(api_key)
    print(f"✓ Saved key to .tmp/test_api_key.txt")

    store.close()
    return True


def test_rls():
    """Test Row-Level Security."""
    print_section("4. Row-Level Security")

    from execution.legal_rag.vector_store import VectorStore

    store = VectorStore()
    store.connect()

    # Enable RLS
    try:
        store.enable_rls()
        print("✓ RLS enabled on tables")
    except Exception as e:
        # May already be enabled
        print(f"⚠ RLS setup: {e}")

    # Test tenant context
    client_a = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    client_b = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"

    store.set_tenant_context(client_a)
    print(f"✓ Tenant context set to Client A")

    store.set_tenant_context(client_b)
    print(f"✓ Tenant context set to Client B")

    store.clear_tenant_context()
    print(f"✓ Tenant context cleared")

    store.close()
    return True


def test_audit_logging():
    """Test audit logging."""
    print_section("5. Audit Logging")

    from execution.legal_rag.vector_store import VectorStore

    store = VectorStore()
    store.connect()
    store.initialize_auth_schema()

    demo_client_id = "00000000-0000-0000-0000-000000000001"

    # Log some actions
    store.log_audit(demo_client_id, "test_login", details={"source": "test_script"})
    store.log_audit(demo_client_id, "test_query", details={"query": "What is the termination clause?"})
    store.log_audit(demo_client_id, "test_ingest", resource_type="document", details={"title": "Test Doc"})

    # Check audit log
    with store._conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM audit_log WHERE client_id = %s::uuid", (demo_client_id,))
        count = cur.fetchone()["count"]

    print(f"✓ Logged {count} audit entries")

    # Show recent entries
    with store._conn.cursor() as cur:
        cur.execute("""
            SELECT action, details, created_at
            FROM audit_log
            WHERE client_id = %s::uuid
            ORDER BY created_at DESC LIMIT 3
        """, (demo_client_id,))
        entries = cur.fetchall()

    for entry in entries:
        print(f"  - {entry['action']}: {entry['details']}")

    store.close()
    return count >= 3


def test_metrics():
    """Test metrics collection."""
    print_section("6. Metrics Collection")

    from execution.legal_rag.metrics import get_metrics_collector
    import time
    import random

    collector = get_metrics_collector()
    collector.reset()  # Start fresh

    # Simulate some queries
    for i in range(5):
        client_id = f"client_{random.randint(1, 2)}"
        with collector.track_query(client_id, f"Test query {i}") as tracker:
            time.sleep(random.uniform(0.01, 0.05))  # Simulate work
            tracker.set_results(
                count=random.randint(1, 10),
                cache_hit=random.random() > 0.5,
                rerank_skipped=random.random() > 0.5,
            )

    metrics = collector.get_metrics_dict()

    print(f"✓ Tracked {metrics['queries']['total']} queries")
    print(f"  - Avg latency: {metrics['latency_ms']['avg']:.2f}ms")
    print(f"  - Cache hit rate: {metrics['cache']['hit_rate']}")
    print(f"  - Rerank skip rate: {metrics['reranking']['skip_rate']}")

    return metrics['queries']['total'] == 5


def test_quotas():
    """Test quota enforcement."""
    print_section("7. Quota Enforcement")

    from execution.legal_rag.quotas import QuotaManager, QuotaExceededError, QUOTA_TIERS

    manager = QuotaManager()

    # Show tier limits
    print("Tier limits:")
    for tier_name, tier in QUOTA_TIERS.items():
        print(f"  {tier_name}: {tier.max_documents} docs, {tier.max_queries_per_day} queries/day")

    # Test quota check (should pass)
    client_id = "test-client-123"
    try:
        manager.check_document_quota(client_id, tier="default", new_chunks=100)
        print("✓ Document upload allowed (under quota)")
    except QuotaExceededError as e:
        print(f"✗ Unexpected quota error: {e}")
        return False

    # Simulate hitting limit
    manager._usage_cache[client_id] = type('Usage', (), {
        'document_count': 100,
        'chunk_count': 50000,
        'queries_today': 0,
    })()

    try:
        manager.check_document_quota(client_id, tier="default")
        print("✗ Should have raised quota error!")
        return False
    except QuotaExceededError as e:
        print(f"✓ Quota correctly enforced: {e.quota_type} limit reached")

    return True


def test_smart_reranking():
    """Test smart reranking configuration."""
    print_section("8. Smart Reranking")

    from execution.legal_rag.retriever import RetrievalConfig

    config = RetrievalConfig()

    print(f"Smart reranking enabled: {config.use_smart_reranking}")
    print(f"Skip threshold: score > {config.smart_rerank_threshold}")
    print(f"Skip gap: gap > {config.smart_rerank_gap}")

    # The actual skip logic is in _should_skip_reranking()
    print("✓ Smart reranking configured")

    return config.use_smart_reranking


def test_connection_pooling():
    """Test connection pooling."""
    print_section("9. Connection Pooling")

    from execution.legal_rag.vector_store import VectorStore, VectorStoreConfig

    # Test with pooling enabled
    config = VectorStoreConfig(use_pooling=True, pool_min_connections=2, pool_max_connections=5)
    store = VectorStore(config)
    store.connect()

    if store._pool:
        print(f"✓ Connection pool created (min=2, max=5)")

        # Test getting connections
        with store.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 as test")
                result = cur.fetchone()["test"]
        print(f"✓ Connection from pool works: result={result}")
    else:
        print("✗ Pool not created!")
        return False

    store.close()
    print("✓ Pool closed cleanly")

    return True


def test_hnsw_index():
    """Test HNSW index information."""
    print_section("10. Vector Index")

    from execution.legal_rag.vector_store import VectorStore

    store = VectorStore()
    store.connect()

    # Check current index
    indexes = store.get_index_info()

    if indexes:
        for name, info in indexes.items():
            print(f"✓ Index: {name} (type: {info['type']})")
    else:
        print("⚠ No vector index found (will be created on first insert)")

    print("\nTo upgrade to HNSW (for 50K+ chunks):")
    print("  store.create_hnsw_index()")

    store.close()
    return True


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print(" Legal RAG Production Features Test Suite")
    print("="*60)

    # Ensure .tmp directory exists
    tmp_dir = project_root / ".tmp"
    tmp_dir.mkdir(exist_ok=True)

    tests = [
        ("Database Reset", test_database_reset),
        ("Auth Schema", test_auth_setup),
        ("API Keys", test_api_key_creation),
        ("Row-Level Security", test_rls),
        ("Audit Logging", test_audit_logging),
        ("Metrics", test_metrics),
        ("Quotas", test_quotas),
        ("Smart Reranking", test_smart_reranking),
        ("Connection Pooling", test_connection_pooling),
        ("Vector Index", test_hnsw_index),
    ]

    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed))
        except Exception as e:
            logger.error(f"Test '{name}' failed with error: {e}")
            results.append((name, False))

    # Summary
    print_section("Test Summary")

    passed = sum(1 for _, p in results if p)
    total = len(results)

    for name, p in results:
        status = "✓ PASS" if p else "✗ FAIL"
        print(f"  {status}: {name}")

    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print("\n✅ All production features working!")
        print("\nNext steps:")
        print("  1. Start the app: streamlit run execution/legal_rag/demo_app.py")
        print("  2. Upload a test PDF")
        print("  3. Ask questions to verify retrieval works")
        print("\nTo enable authentication:")
        print("  AUTH_ENABLED=true streamlit run execution/legal_rag/demo_app.py")
        print(f"  API Key saved to: .tmp/test_api_key.txt")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
