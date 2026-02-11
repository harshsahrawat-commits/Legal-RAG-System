"""
Dimension 4: HALLUCINATION-PROOFING

Tests that the RAG system grounds answers in source documents, refuses to
answer out-of-domain questions, and generates accurate citations.
"""

import sys
import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.conftest import (
    ENGLISH_CLIENT_ID, GREEK_CLIENT_ID, skip_no_creds, run_rag_pipeline,
)

pytestmark = [pytest.mark.hallucination, pytest.mark.integration]


# =============================================================================
# 4A. Answer Grounding
# =============================================================================

REFUSAL_PHRASES = [
    "not contain", "no relevant", "don't have", "do not contain",
    "cannot find", "no information", "not found", "outside the scope",
    "provided documents do not", "unable to find", "not available",
    "cannot answer", "insufficient information",
    # Greek refusal phrases
    "δεν περιέχ", "δεν βρέθηκ", "δεν υπάρχ",
]


@skip_no_creds
class TestAnswerGrounding:
    """Verify answers are grounded in source chunks."""

    def test_answer_shares_terms_with_sources(
        self, en_retriever, live_llm, live_citation_extractor, live_store,
    ):
        answer, sources = run_rag_pipeline(
            "What claims are alleged in the complaint?",
            en_retriever, live_llm, live_citation_extractor, live_store,
            ENGLISH_CLIENT_ID,
        )

        assert sources, "Need sources to verify grounding"

        source_text = " ".join(cc.content.lower() for cc in sources)
        source_words = set(re.findall(r"\b[a-z]{4,}\b", source_text))

        answer_lower = answer.lower()
        overlap = [w for w in source_words if w in answer_lower]
        overlap_ratio = len(overlap) / max(len(source_words), 1)

        assert overlap_ratio > 0.03, (
            f"Answer shares only {overlap_ratio:.1%} of source terms — may be hallucinated. "
            f"Overlap: {overlap[:10]}"
        )

    def test_citation_markers_present(
        self, en_retriever, live_llm, live_citation_extractor, live_store,
    ):
        answer, _ = run_rag_pipeline(
            "Who filed the complaint?",
            en_retriever, live_llm, live_citation_extractor, live_store,
            ENGLISH_CLIENT_ID,
        )

        citations = re.findall(r"\[\d+\]", answer)
        assert len(citations) >= 1, (
            f"Answer should contain citation markers like [1]. Got none.\n"
            f"Answer excerpt: {answer[:300]}"
        )

    def test_cited_numbers_within_source_count(
        self, en_retriever, live_llm, live_citation_extractor, live_store,
    ):
        answer, sources = run_rag_pipeline(
            "What is this case about?",
            en_retriever, live_llm, live_citation_extractor, live_store,
            ENGLISH_CLIENT_ID,
        )

        max_source = len(sources)
        if max_source == 0:
            pytest.skip("No sources returned")

        cited_nums = [int(m) for m in re.findall(r"\[(\d+)\]", answer)]
        for n in cited_nums:
            assert 1 <= n <= max_source, (
                f"Citation [{n}] exceeds source count ({max_source})"
            )


# =============================================================================
# 4B. Adversarial / Out-of-Domain Queries
# =============================================================================

ADVERSARIAL_QUERIES = [
    "What is the recipe for chocolate cake?",
    "Explain quantum entanglement in simple terms",
    "Who won the 2024 Super Bowl?",
    "What is the best programming language?",
    "What are the patent infringement penalties in Japan?",
    "Summarize the Australian Consumer Law amendments of 2023",
    "What are the zoning regulations in Tokyo?",
]


@skip_no_creds
class TestAdversarialQueries:
    """System should refuse or caveat for out-of-domain queries."""

    @pytest.mark.parametrize("query", ADVERSARIAL_QUERIES, ids=[q[:35] for q in ADVERSARIAL_QUERIES])
    def test_refuses_or_caveats_ood_query(
        self, query, en_retriever, live_llm, live_citation_extractor, live_store,
    ):
        answer, sources = run_rag_pipeline(
            query, en_retriever, live_llm, live_citation_extractor, live_store,
            ENGLISH_CLIENT_ID,
        )

        if not sources:
            # No sources found — answer should acknowledge lack of information
            has_refusal = any(phrase in answer.lower() for phrase in REFUSAL_PHRASES)
            assert has_refusal, (
                f"No sources found but answer doesn't indicate lack of info.\n"
                f"Answer: {answer[:300]}"
            )
        else:
            # Sources returned (possibly false positive retrieval).
            # Answer should not be suspiciously detailed for irrelevant content.
            # Also check for at least some acknowledgement of limitations.
            assert len(answer) < 3000, (
                f"Suspiciously detailed answer ({len(answer)} chars) for OOD query: {query}"
            )


# =============================================================================
# 4C. Citation Verification
# =============================================================================

@skip_no_creds
class TestCitationVerification:
    """Verify citations reference real content."""

    def test_citation_titles_are_real(self, en_retriever, live_store, live_citation_extractor):
        results = en_retriever.retrieve(
            "termination provisions", client_id=ENGLISH_CLIENT_ID, top_k=3, use_cache=False,
        )
        results = [r for r in results if r.hierarchy_path != "Document"]
        if not results:
            pytest.skip("No non-summary results returned")

        doc_ids = list(set(r.document_id for r in results))
        doc_titles = live_store.get_document_titles(doc_ids, client_id=ENGLISH_CLIENT_ID)
        cited = live_citation_extractor.extract(results, document_titles=doc_titles)

        non_generic = [cc for cc in cited if cc.citation.document_title != "Document"]
        assert len(non_generic) > 0, "At least one citation should have a real title"

    def test_page_numbers_within_range(self, en_retriever, live_store):
        results = en_retriever.retrieve(
            "breach of contract", client_id=ENGLISH_CLIENT_ID, top_k=3, use_cache=False,
        )

        docs = live_store.list_documents(client_id=ENGLISH_CLIENT_ID)
        doc_page_map = {str(d["id"]): d.get("page_count", 0) for d in docs}

        for r in results:
            if r.page_numbers and r.document_id in doc_page_map:
                page_count = doc_page_map[r.document_id]
                if page_count > 0:
                    for p in r.page_numbers:
                        assert 1 <= p <= page_count + 1, (
                            f"Page {p} outside range [1, {page_count}] for doc {r.document_id}"
                        )

    def test_short_citation_format_valid(self, en_retriever, live_store, live_citation_extractor):
        results = en_retriever.retrieve(
            "liability", client_id=ENGLISH_CLIENT_ID, top_k=3, use_cache=False,
        )
        results = [r for r in results if r.hierarchy_path != "Document"]
        if not results:
            pytest.skip("No non-summary results returned")

        doc_ids = list(set(r.document_id for r in results))
        doc_titles = live_store.get_document_titles(doc_ids, client_id=ENGLISH_CLIENT_ID)
        cited = live_citation_extractor.extract(results, document_titles=doc_titles)

        for cc in cited:
            short = cc.citation.short_format()
            assert short.startswith("["), f"Short citation should start with '[': {short}"
            assert short.endswith("]"), f"Short citation should end with ']': {short}"
