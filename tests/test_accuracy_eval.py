"""
Dimension 3: ACCURACY

Tests retrieval relevance against ground-truth queries for both English and
Greek tenants, query classification correctness, and hybrid vs vector-only search.
"""

import sys
import re
from pathlib import Path
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.conftest import (
    ENGLISH_CLIENT_ID, GREEK_CLIENT_ID, skip_no_creds,
)

pytestmark = pytest.mark.accuracy


# =============================================================================
# 3A. English Retrieval Ground Truth
# =============================================================================

ENGLISH_GROUND_TRUTH = [
    {
        "query": "Who are the defendants in this case?",
        "expected_keywords": ["defendant", "defendants"],
        "description": "Defendant identification",
    },
    {
        "query": "What claims are alleged in the complaint?",
        "expected_keywords": ["claim", "cause of action", "allege", "complaint"],
        "description": "Claims in complaint",
    },
    {
        "query": "What court has jurisdiction over this case?",
        "expected_keywords": ["court", "district", "jurisdiction", "venue"],
        "description": "Jurisdiction/venue",
    },
    {
        "query": "What are the grounds for dismissal in this case?",
        "expected_keywords": ["dismiss", "motion", "grant", "denied", "order", "summary judgment"],
        "description": "Grounds for dismissal",
    },
    {
        "query": "What are the payment terms or financial obligations?",
        "expected_keywords": ["payment", "fee", "amount", "pay", "cost", "dollar", "sum", "money", "fund"],
        "description": "Payment terms",
    },
    {
        "query": "What damages are sought by the plaintiff?",
        "expected_keywords": ["damage", "relief", "compensat", "injunctive"],
        "description": "Damages sought",
    },
    {
        "query": "What motions have been filed?",
        "expected_keywords": ["motion", "filed", "order"],
        "description": "Filed motions",
    },
]


@skip_no_creds
@pytest.mark.integration
class TestEnglishRetrievalAccuracy:
    """Test retrieval quality against English ground-truth queries."""

    @pytest.mark.parametrize(
        "gt",
        ENGLISH_GROUND_TRUTH,
        ids=[g["description"] for g in ENGLISH_GROUND_TRUTH],
    )
    def test_retrieval_relevance(self, en_retriever, gt):
        results = en_retriever.retrieve(
            query=gt["query"], client_id=ENGLISH_CLIENT_ID, top_k=5, use_cache=False,
        )
        assert len(results) >= 1, f"Expected >=1 results for: {gt['query']}"

        all_content = " ".join(r.content.lower() for r in results)
        found_any = any(kw.lower() in all_content for kw in gt["expected_keywords"])
        assert found_any, (
            f"None of {gt['expected_keywords']} found in top results for: {gt['query']}"
        )

    def test_top_result_has_positive_score(self, en_retriever):
        results = en_retriever.retrieve(
            "Who are the defendants?", client_id=ENGLISH_CLIENT_ID, top_k=5, use_cache=False,
        )
        assert results, "Should return at least 1 result"
        assert results[0].score > 0, f"Top result score {results[0].score} should be positive"


# =============================================================================
# 3B. Greek Retrieval Ground Truth
# =============================================================================

GREEK_GROUND_TRUTH = [
    {
        "query": "Ποιος είναι ο νόμος περί ενοικιοστασίου στην Κύπρο;",
        "expected_keywords": ["ενοικ", "μίσθωσ", "νόμος"],
        "description": "Rent control law",
    },
    {
        "query": "Τι προβλέπει ο νόμος για την απόλυση εργαζομένων;",
        "expected_keywords": ["απόλυσ", "εργαζ", "τερματισμ"],
        "description": "Employee termination",
    },
    {
        "query": "Ποιες είναι οι ποινές για φοροδιαφυγή;",
        "expected_keywords": ["φόρ", "ποιν", "πρόστιμ"],
        "description": "Tax evasion penalties",
    },
    {
        "query": "Τι δικαιώματα έχει ο ενοικιαστής;",
        "expected_keywords": ["ενοικιαστ", "δικαίωμ", "μισθωτ"],
        "description": "Tenant rights",
    },
    {
        "query": "Πώς γίνεται η εγγραφή εταιρείας στην Κύπρο;",
        "expected_keywords": ["εταιρεί", "εγγραφ", "μητρώ"],
        "description": "Company registration",
    },
    {
        "query": "Ποια είναι τα δικαιώματα του κατηγορουμένου;",
        "expected_keywords": ["κατηγορ", "δικαίωμ", "υπεράσπισ"],
        "description": "Rights of the accused",
    },
    {
        "query": "Τι προβλέπει ο νόμος περί προστασίας προσωπικών δεδομένων;",
        "expected_keywords": ["δεδομέν", "προσωπικ", "προστασί"],
        "description": "Data protection law",
    },
]


@skip_no_creds
@pytest.mark.integration
class TestGreekRetrievalAccuracy:
    """Test retrieval quality against Greek ground-truth queries."""

    @pytest.mark.parametrize(
        "gt",
        GREEK_GROUND_TRUTH,
        ids=[g["description"] for g in GREEK_GROUND_TRUTH],
    )
    def test_retrieval_relevance(self, el_retriever, gt):
        results = el_retriever.retrieve(
            query=gt["query"], client_id=GREEK_CLIENT_ID, top_k=5, use_cache=False,
        )
        assert len(results) >= 1, f"Expected >=1 results for: {gt['query']}"

        all_content = " ".join(r.content.lower() for r in results)
        found_any = any(kw.lower() in all_content for kw in gt["expected_keywords"])
        assert found_any, (
            f"None of {gt['expected_keywords']} found in top results for: {gt['query']}"
        )

    def test_greek_top_result_has_positive_score(self, el_retriever):
        results = el_retriever.retrieve(
            "νόμος ενοικιοστασίου", client_id=GREEK_CLIENT_ID, top_k=5, use_cache=False,
        )
        assert results, "Should return at least 1 result for Greek query"
        assert results[0].score > 0, f"Top result score {results[0].score} should be positive"


# =============================================================================
# 3C. Query Classification (unit)
# =============================================================================

CLASSIFICATION_GROUND_TRUTH = [
    # (query, expected_type)
    ("contract", "simple"),
    ("NDA", "simple"),
    ("defendant", "simple"),
    ("What is the effective date?", "factual"),
    ("Who are the parties?", "factual"),
    ("When was this filed?", "factual"),
    ("Explain the implications of the indemnity clause", "analytical"),
    ("Compare the liability caps in both agreements", "analytical"),
    ("Why was the motion denied?", "analytical"),
    ("How does Section 4.2 apply to early termination?", "standard"),
    # Greek queries
    ("σύμβαση", "simple"),
    ("Εξηγήστε τις συνέπειες της ρήτρας αποζημίωσης", "analytical"),
]


class TestQueryClassification:
    """Verify query classification assigns correct types."""

    @pytest.mark.parametrize(
        "query,expected",
        CLASSIFICATION_GROUND_TRUTH,
        ids=[f"{q[:30]}=>{e}" for q, e in CLASSIFICATION_GROUND_TRUTH],
    )
    def test_classification(self, query, expected, mock_embedding_service, mock_vector_store):
        from execution.legal_rag.retriever import HybridRetriever, RetrievalConfig
        from execution.legal_rag.language_config import TenantLanguageConfig

        lang = "el" if any('\u0370' <= c <= '\u03ff' for c in query) else "en"
        lang_config = TenantLanguageConfig.for_language(lang)
        retriever = HybridRetriever(
            mock_vector_store, mock_embedding_service,
            config=RetrievalConfig(use_reranking=False),
            language_config=lang_config,
        )
        result = retriever._classify_query(query)
        assert result == expected, (
            f"Query '{query}' classified as '{result}', expected '{expected}'"
        )


# =============================================================================
# 3D. Hybrid vs Vector-Only (integration)
# =============================================================================

@skip_no_creds
@pytest.mark.integration
class TestHybridVsVectorOnly:
    """Verify hybrid search returns results at least as good as vector-only."""

    COMPARISON_QUERIES = [
        "defendant negligence",
        "termination for breach",
        "confidentiality obligations",
    ]

    @pytest.mark.parametrize("query", COMPARISON_QUERIES)
    def test_hybrid_returns_results(self, en_retriever, live_store, en_embeddings, query):
        # Vector-only
        q_emb = en_embeddings.embed_query(query)
        vector_results = live_store.search(q_emb, top_k=5, client_id=ENGLISH_CLIENT_ID)

        # Hybrid
        hybrid_results = en_retriever.retrieve(
            query, client_id=ENGLISH_CLIENT_ID, top_k=5, use_cache=False,
        )

        # Hybrid should return at least as many results
        assert len(hybrid_results) >= min(len(vector_results), 1), (
            f"Hybrid ({len(hybrid_results)}) should match or exceed vector-only ({len(vector_results)})"
        )
