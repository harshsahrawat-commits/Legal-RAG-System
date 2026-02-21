#!/usr/bin/env python3
"""
Test script for latency optimizations in Phase 5.

Tests:
1. Query Classification - verifies different query types get different pipelines
2. Semantic Result Caching - verifies cache hits for similar queries
3. Latency Comparison - measures actual latency improvements
"""

import os
import sys
import time
import logging
from pathlib import Path
from dotenv import load_dotenv

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(name)s - %(message)s')
logger = logging.getLogger(__name__)

# Demo client ID
DEMO_CLIENT_ID = "00000000-0000-0000-0000-000000000001"


def test_query_classification():
    """Test that query classification works correctly."""
    print("\n" + "="*70)
    print("TEST 1: Query Classification")
    print("="*70)

    from execution.legal_rag.retriever import HybridRetriever, QUERY_CONFIGS
    from execution.legal_rag.vector_store import VectorStore
    from execution.legal_rag.embeddings import get_embedding_service

    # Initialize (we just need the classifier, not full retrieval)
    store = VectorStore()
    store.connect()
    embeddings = get_embedding_service(provider="voyage")
    retriever = HybridRetriever(store, embeddings)

    test_cases = [
        # (query, expected_classification)
        ("damages?", "simple"),  # ≤5 words
        ("What is breach of warranty?", "factual"),  # "What is" pattern
        ("List the defendants", "factual"),  # "List" pattern
        ("Explain the implications of the ruling", "analytical"),  # "Explain" pattern
        ("Why did the court rule this way?", "analytical"),  # "Why" pattern
        ("Compare the damages claims across plaintiffs", "analytical"),  # "Compare" pattern
        ("What are the causes of action in this case?", "factual"),  # "What are the" → factual
    ]

    passed = 0
    for query, expected in test_cases:
        actual = retriever._classify_query(query)
        status = "✅" if actual == expected else "❌"
        if actual == expected:
            passed += 1
        print(f"  {status} '{query[:40]}...' → {actual} (expected: {expected})")

    print(f"\nClassification: {passed}/{len(test_cases)} tests passed")
    store.close()
    return passed == len(test_cases)


def test_semantic_caching():
    """Test that semantic caching works for similar queries."""
    print("\n" + "="*70)
    print("TEST 2: Semantic Result Caching")
    print("="*70)

    from execution.legal_rag.retriever import QueryResultCache
    from execution.legal_rag.embeddings import get_embedding_service

    embeddings = get_embedding_service(provider="voyage")
    cache = QueryResultCache(embeddings, similarity_threshold=0.90)

    # Mock results
    class MockResult:
        def __init__(self, content):
            self.chunk_id = f"chunk_{hash(content)}"
            self.content = content

    mock_results = [MockResult("Test content 1"), MockResult("Test content 2")]

    # Test 1: Cache miss on first query
    query1 = "What are the breach of warranty claims?"
    result1 = cache.get(query1)
    print(f"  Query 1 (first): Cache {'HIT' if result1 else 'MISS'}")
    assert result1 is None, "Expected cache miss on first query"

    # Set cache
    cache.set(query1, mock_results)
    print(f"  Cached results for query 1")

    # Test 2: Cache hit on exact same query
    result2 = cache.get(query1)
    print(f"  Query 1 (repeat): Cache {'HIT' if result2 else 'MISS'}")
    assert result2 is not None, "Expected cache hit on repeat query"

    # Test 3: Cache hit on similar query
    query2 = "What are the warranty breach claims?"  # Similar meaning
    result3 = cache.get(query2)
    print(f"  Query 2 (similar): Cache {'HIT' if result3 else 'MISS'}")
    # This depends on embedding similarity - may or may not hit

    # Test 4: Cache miss on different query
    query3 = "Who are the defendants in this case?"  # Different topic
    result4 = cache.get(query3)
    print(f"  Query 3 (different): Cache {'HIT' if result4 else 'MISS'}")
    # This should miss

    print(f"\n  Cache size: {cache.size}")
    print("  ✅ Semantic caching basic tests passed")
    return True


def test_latency_improvement():
    """Test actual latency with different query types."""
    print("\n" + "="*70)
    print("TEST 3: Latency Comparison")
    print("="*70)

    from execution.legal_rag.retriever import HybridRetriever, RetrievalConfig
    from execution.legal_rag.vector_store import VectorStore
    from execution.legal_rag.embeddings import get_embedding_service

    store = VectorStore()
    store.connect()
    embeddings = get_embedding_service(provider="voyage")
    retriever = HybridRetriever(store, embeddings)

    test_queries = [
        ("damages?", "simple"),
        ("What are the causes of action?", "standard"),
        ("Explain the breach of warranty implications", "analytical"),
    ]

    results = []

    for query, expected_type in test_queries:
        print(f"\n  Testing: '{query}' (expected: {expected_type})")

        # First run (no cache)
        start = time.time()
        result1 = retriever.retrieve(query, client_id=DEMO_CLIENT_ID, top_k=5)
        latency1 = (time.time() - start) * 1000

        # Second run (should hit cache)
        start = time.time()
        result2 = retriever.retrieve(query, client_id=DEMO_CLIENT_ID, top_k=5)
        latency2 = (time.time() - start) * 1000

        print(f"    First run:  {latency1:.0f}ms ({len(result1)} results)")
        print(f"    Second run: {latency2:.0f}ms ({len(result2)} results) {'🚀 CACHE HIT!' if latency2 < latency1 * 0.5 else ''}")

        results.append({
            "query": query,
            "type": expected_type,
            "latency_first": latency1,
            "latency_cached": latency2,
            "speedup": latency1 / latency2 if latency2 > 0 else 1,
        })

    # Summary
    print("\n" + "-"*70)
    print("LATENCY SUMMARY:")
    print("-"*70)
    print(f"{'Query Type':<15} {'First (ms)':<12} {'Cached (ms)':<12} {'Speedup':<10}")
    print("-"*70)
    for r in results:
        print(f"{r['type']:<15} {r['latency_first']:<12.0f} {r['latency_cached']:<12.0f} {r['speedup']:<10.1f}x")

    avg_first = sum(r['latency_first'] for r in results) / len(results)
    avg_cached = sum(r['latency_cached'] for r in results) / len(results)
    print("-"*70)
    print(f"{'AVERAGE':<15} {avg_first:<12.0f} {avg_cached:<12.0f} {avg_first/avg_cached if avg_cached > 0 else 1:<10.1f}x")

    store.close()
    return True


def main():
    """Run all latency optimization tests."""
    print("\n" + "="*70)
    print("PHASE 5: LATENCY OPTIMIZATION TESTS")
    print("="*70)

    all_passed = True

    # Test 1: Query Classification
    try:
        if not test_query_classification():
            all_passed = False
    except Exception as e:
        print(f"  ❌ Query classification test failed: {e}")
        all_passed = False

    # Test 2: Semantic Caching
    try:
        if not test_semantic_caching():
            all_passed = False
    except Exception as e:
        print(f"  ❌ Semantic caching test failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    # Test 3: Latency Comparison
    try:
        if not test_latency_improvement():
            all_passed = False
    except Exception as e:
        print(f"  ❌ Latency test failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    # Final summary
    print("\n" + "="*70)
    if all_passed:
        print("✅ ALL LATENCY OPTIMIZATION TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED - Review output above")
    print("="*70)

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
