#!/usr/bin/env python3
"""
Diagnostic script to analyze score behavior through the retrieval pipeline.
Checks raw vector scores, RRF scores, and reranked scores.
"""

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(name)s - %(message)s')
logger = logging.getLogger(__name__)

def diagnose_retrieval():
    """Run diagnostic queries and analyze scores at each stage."""
    from execution.legal_rag.embeddings import get_embedding_service
    from execution.legal_rag.vector_store import VectorStore
    from execution.legal_rag.retriever import HybridRetriever, RetrievalConfig

    # Initialize
    store = VectorStore()
    store.connect()
    embeddings = get_embedding_service(provider="voyage")

    # Create retriever with detailed config
    config = RetrievalConfig(
        use_reranking=True,
        use_smart_reranking=False,  # Force reranking for diagnostics
        use_query_expansion=False,  # Simplify for diagnostics
        use_hyde=False,
        use_multi_query=False,
    )
    retriever = HybridRetriever(store, embeddings, config)

    test_queries = [
        "When was the case filed?",
        "What are the causes of action?",
        "breach of warranty claims",
        "paragraph 28",
    ]

    print("\n" + "="*80)
    print("RETRIEVAL SCORE DIAGNOSTICS")
    print("="*80)

    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"QUERY: {query}")
        print("="*80)

        # Step 1: Raw vector search only
        query_embedding = embeddings.embed_query(query)
        raw_results = store.search(query_embedding, top_k=5)

        print("\n📊 STAGE 1: Raw Vector Search Scores")
        for i, r in enumerate(raw_results[:5]):
            print(f"   {i+1}. score={r.score:.6f} | {r.section_title[:40]}...")

        # Step 2: RRF fusion scores
        keyword_results = store.keyword_search(query, top_k=5)
        fused = retriever._reciprocal_rank_fusion(raw_results[:5], keyword_results[:5])

        print("\n📊 STAGE 2: RRF Fusion Scores")
        for i, r in enumerate(fused[:5]):
            print(f"   {i+1}. score={r.score:.6f} | {r.section_title[:40]}...")

        # Step 3: Reranked scores
        if retriever._reranker:
            print("\n📊 STAGE 3: Cohere Reranked Scores")
            reranked = retriever._rerank(query, fused[:10])
            for i, r in enumerate(reranked[:5]):
                print(f"   {i+1}. score={r.score:.4f} | {r.section_title[:40]}...")
        else:
            print("\n⚠️  Reranker not initialized (COHERE_API_KEY missing?)")

        # Step 4: Full pipeline
        print("\n📊 STAGE 4: Full Pipeline (with all enhancements)")
        full_config = RetrievalConfig(use_reranking=True)
        full_retriever = HybridRetriever(store, embeddings, full_config)
        full_results = full_retriever.retrieve(query, top_k=5)
        for i, r in enumerate(full_results[:5]):
            rerank_score = r.metadata.get('rerank_score', 'N/A')
            print(f"   {i+1}. final={r.score:.4f} | rerank={rerank_score} | {r.section_title[:35]}...")

    store.close()
    print("\n" + "="*80)
    print("DIAGNOSTICS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    diagnose_retrieval()
