"""
Live integration tests for RAG accuracy.

These tests hit the real Neon database, Voyage AI embeddings, and NVIDIA NIM LLM
to verify end-to-end accuracy across ingestion, retrieval, and answer generation.

Requires valid credentials in .env:
    POSTGRES_URL, VOYAGE_API_KEY, NVIDIA_API_KEY

Run with:
    pytest tests/test_integration.py -v -m integration
"""

import os
import re
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

# Skip the entire module if credentials are missing
pytestmark = pytest.mark.integration

DEMO_CLIENT_ID = "00000000-0000-0000-0000-000000000001"
TEST_FILES_DIR = Path.home() / "test_files"
# The 23-page initial complaint — the richest document in the test set
TEST_PDF = TEST_FILES_DIR / "gov.uscourts.arwd.28923.1.0.pdf"


def _credentials_available() -> bool:
    """Check if all required credentials are present."""
    return all(os.getenv(k) for k in ("POSTGRES_URL", "VOYAGE_API_KEY", "NVIDIA_API_KEY"))


skip_no_creds = pytest.mark.skipif(
    not _credentials_available(),
    reason="Missing POSTGRES_URL, VOYAGE_API_KEY, or NVIDIA_API_KEY in .env",
)

skip_no_test_files = pytest.mark.skipif(
    not TEST_PDF.exists(),
    reason=f"Test PDF not found: {TEST_PDF}",
)


# =============================================================================
# Fixtures — module-scoped to share expensive connections
# =============================================================================

@pytest.fixture(scope="module")
def live_store():
    """Real VectorStore connected to Neon database."""
    from execution.legal_rag.vector_store import VectorStore
    store = VectorStore()
    store.connect()
    store.set_tenant_context(DEMO_CLIENT_ID)
    yield store
    store.close()


@pytest.fixture(scope="module")
def live_embeddings():
    """Real Voyage AI embedding service."""
    from execution.legal_rag.embeddings import get_embedding_service
    from execution.legal_rag.language_config import TenantLanguageConfig
    lang_config = TenantLanguageConfig.for_language("en")
    return get_embedding_service(provider="voyage", language_config=lang_config)


@pytest.fixture(scope="module")
def live_retriever(live_store, live_embeddings):
    """Real HybridRetriever with live DB and embeddings."""
    from execution.legal_rag.retriever import HybridRetriever
    from execution.legal_rag.language_config import TenantLanguageConfig
    lang_config = TenantLanguageConfig.for_language("en")
    return HybridRetriever(live_store, live_embeddings, language_config=lang_config)


@pytest.fixture(scope="module")
def live_llm():
    """Real OpenAI client pointed at NVIDIA NIM."""
    from openai import OpenAI
    return OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.getenv("NVIDIA_API_KEY"),
        timeout=30.0,
    )


@pytest.fixture(scope="module")
def live_citation_extractor():
    """Real CitationExtractor."""
    from execution.legal_rag.citation import CitationExtractor
    return CitationExtractor()


@pytest.fixture(scope="module")
def parsed_document():
    """Parse the test PDF once for all ingestion tests."""
    from execution.legal_rag.document_parser import LegalDocumentParser
    parser = LegalDocumentParser(use_docling=False)
    return parser.parse(str(TEST_PDF), client_id=DEMO_CLIENT_ID)


@pytest.fixture(scope="module")
def chunked_document(parsed_document):
    """Chunk the parsed document once for all ingestion tests."""
    from execution.legal_rag.chunker import LegalChunker
    chunker = LegalChunker()
    return chunker.chunk(parsed_document)


# =============================================================================
# Group 1: Ingestion Accuracy
# =============================================================================

@skip_no_creds
@skip_no_test_files
class TestIngestionAccuracy:
    """Verify PDF parsing and chunking produce correct output."""

    def test_parse_real_pdf(self, parsed_document):
        """Parse a real court filing and verify metadata."""
        doc = parsed_document

        # Must have extracted text
        assert doc.raw_text, "raw_text should be non-empty"
        assert len(doc.raw_text) > 100, "Expected substantial text from 23-page PDF"

        # Metadata checks
        assert doc.metadata.page_count > 0, "page_count should be positive"
        assert doc.metadata.title, "title should be non-empty"
        assert doc.metadata.document_id, "document_id should be set"

        # Should have sections
        assert len(doc.sections) >= 1, "Expected at least 1 section"

    def test_chunk_real_document(self, parsed_document, chunked_document):
        """Chunk the parsed document and verify chunk quality."""
        chunks = chunked_document

        assert len(chunks) > 0, "Should produce at least 1 chunk"

        for i, chunk in enumerate(chunks):
            assert chunk.content, f"Chunk {i} has empty content"
            assert len(chunk.content) > 50, f"Chunk {i} content too short ({len(chunk.content)} chars)"
            assert chunk.hierarchy_path, f"Chunk {i} missing hierarchy_path"
            assert chunk.chunk_id, f"Chunk {i} missing chunk_id"
            assert chunk.document_id == parsed_document.metadata.document_id, (
                f"Chunk {i} document_id mismatch"
            )

            # Page numbers should be within document range if present
            for p in chunk.page_numbers:
                assert 1 <= p <= parsed_document.metadata.page_count + 1, (
                    f"Chunk {i} page {p} outside range [1, {parsed_document.metadata.page_count}]"
                )

    @skip_no_creds
    def test_embed_real_chunks(self, chunked_document, live_embeddings):
        """Embed chunk texts with live Voyage AI and verify dimensions."""
        # Use a subset to control cost
        sample_texts = [c.content for c in chunked_document[:5]]
        embeddings = live_embeddings.embed_documents(sample_texts)

        assert len(embeddings) == len(sample_texts), "Embedding count should match input count"

        for i, emb in enumerate(embeddings):
            assert len(emb) == 1024, f"Embedding {i} should be 1024-dim, got {len(emb)}"
            assert all(isinstance(v, float) for v in emb), f"Embedding {i} values should be floats"
            assert all(-2.0 <= v <= 2.0 for v in emb), f"Embedding {i} has out-of-range values"


# =============================================================================
# Group 2: Retrieval Accuracy
# =============================================================================

@skip_no_creds
class TestRetrievalAccuracy:
    """Verify search returns relevant results from the live database."""

    def test_vector_search_returns_results(self, live_store, live_embeddings):
        """Vector search should return scored results."""
        query_embedding = live_embeddings.embed_query("Who are the defendants?")
        results = live_store.search(
            query_embedding=query_embedding,
            top_k=5,
            client_id=DEMO_CLIENT_ID,
        )

        assert len(results) > 0, "Vector search should return at least 1 result"

        for r in results:
            assert 0.0 <= r.score <= 1.0, f"Score {r.score} out of [0,1] range"
            assert r.content, "Result content should be non-empty"
            assert r.document_id, "Result should have document_id"

    def test_keyword_search_returns_results(self, live_store):
        """Keyword search should find documents containing the query term."""
        results = live_store.keyword_search(
            query="defendant",
            top_k=5,
            client_id=DEMO_CLIENT_ID,
        )

        assert len(results) > 0, "Keyword search for 'defendant' should return results"

        # At least one result should mention defendant-related terms
        found_relevant = any(
            "defend" in r.content.lower() for r in results
        )
        assert found_relevant, "At least one result should contain 'defend*'"

    def test_hybrid_retrieval_relevance(self, live_retriever):
        """Hybrid retrieval should return relevant, scored results."""
        results = live_retriever.retrieve(
            query="Who are the defendants in this case?",
            client_id=DEMO_CLIENT_ID,
            top_k=5,
        )

        assert len(results) >= 1, "Should retrieve at least 1 result"

        # Results should have content
        for r in results:
            assert r.content, "Each result should have content"
            assert r.score > 0, "Each result should have a positive score"

    def test_retrieval_tenant_isolation(self, live_retriever):
        """Retrieval with a non-existent tenant should return 0 results."""
        fake_client_id = "ffffffff-ffff-ffff-ffff-ffffffffffff"
        results = live_retriever.retrieve(
            query="defendant",
            client_id=fake_client_id,
            top_k=5,
        )

        assert len(results) == 0, (
            f"Fake tenant should get 0 results, got {len(results)}"
        )


# =============================================================================
# Group 3: Answer Generation Accuracy
# =============================================================================

@skip_no_creds
class TestAnswerGeneration:
    """Verify the full RAG pipeline produces grounded answers."""

    def _run_rag_pipeline(self, query, live_retriever, live_llm, live_citation_extractor, live_store):
        """Helper: run the full RAG pipeline and return (answer, sources)."""
        from execution.legal_rag.language_patterns import LLM_PROMPTS

        results = live_retriever.retrieve(
            query=query,
            client_id=DEMO_CLIENT_ID,
            top_k=5,
        )
        results = [r for r in results if r.hierarchy_path != "Document"]

        if not results:
            return "", []

        # Look up real titles
        doc_ids = list(set(r.document_id for r in results))
        doc_titles = live_store.get_document_titles(doc_ids, client_id=DEMO_CLIENT_ID)

        cited_contents = live_citation_extractor.extract(
            results, document_titles=doc_titles
        )

        # Build context
        context = "\n\n---\n\n".join([
            f"**[{i+1}]** {cc.citation.short_format()}:\n{cc.content}"
            for i, cc in enumerate(cited_contents)
        ])

        system_prompt = LLM_PROMPTS["en"]["rag_system"]
        response = live_llm.chat.completions.create(
            model="qwen/qwen3-235b-a22b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Based on the following sources, answer this question: {query}\n\nSOURCES:\n{context}\n\nProvide a clear, well-cited answer."},
            ],
            max_tokens=1500,
            temperature=0.2,
        )
        answer = response.choices[0].message.content

        return answer, cited_contents

    def test_full_rag_pipeline_returns_answer(
        self, live_retriever, live_llm, live_citation_extractor, live_store
    ):
        """Full pipeline should return a substantive answer with sources."""
        answer, sources = self._run_rag_pipeline(
            "What is this case about?",
            live_retriever, live_llm, live_citation_extractor, live_store,
        )

        assert answer, "Answer should be non-empty"
        assert len(answer) > 50, f"Answer too short ({len(answer)} chars)"
        assert len(sources) > 0, "Should have at least 1 source"

        # Each source should have required fields
        for cc in sources:
            assert cc.citation.document_title, "Source should have document_title"
            assert cc.content, "Source should have content"
            assert cc.citation.relevance_score is not None, "Source should have relevance_score"

    def test_answer_contains_citations(
        self, live_retriever, live_llm, live_citation_extractor, live_store
    ):
        """Answer should contain citation markers like [1], [2]."""
        answer, _ = self._run_rag_pipeline(
            "Who filed the complaint?",
            live_retriever, live_llm, live_citation_extractor, live_store,
        )

        # Look for citation markers [1], [2], etc.
        citation_pattern = re.compile(r"\[\d+\]")
        citations_found = citation_pattern.findall(answer)
        assert len(citations_found) > 0, (
            f"Answer should contain citation markers like [1], got none.\n"
            f"Answer excerpt: {answer[:300]}"
        )

    def test_answer_grounded_in_sources(
        self, live_retriever, live_llm, live_citation_extractor, live_store
    ):
        """Answer should contain terms from the source content (not hallucinated)."""
        answer, sources = self._run_rag_pipeline(
            "What claims are alleged in the complaint?",
            live_retriever, live_llm, live_citation_extractor, live_store,
        )

        assert sources, "Need sources to check grounding"

        # Extract significant words (4+ chars) from top source content
        top_source_text = sources[0].content.lower()
        words = set(w for w in re.findall(r"\b[a-z]{4,}\b", top_source_text))

        # At least some source terms should appear in the answer
        answer_lower = answer.lower()
        overlap = [w for w in words if w in answer_lower]

        assert len(overlap) >= 3, (
            f"Answer should share at least 3 terms with top source. "
            f"Found {len(overlap)} overlap: {overlap[:10]}"
        )


# =============================================================================
# Group 4: Citation Accuracy
# =============================================================================

@skip_no_creds
class TestCitationAccuracy:
    """Verify citation extraction produces real document titles and valid formats."""

    def test_citation_extractor_real_titles(
        self, live_retriever, live_store, live_citation_extractor
    ):
        """Citation extractor should use real document titles, not 'Document'."""
        results = live_retriever.retrieve(
            query="What motions have been filed?",
            client_id=DEMO_CLIENT_ID,
            top_k=5,
        )
        results = [r for r in results if r.hierarchy_path != "Document"]
        assert results, "Need results to test citations"

        doc_ids = list(set(r.document_id for r in results))
        doc_titles = live_store.get_document_titles(doc_ids, client_id=DEMO_CLIENT_ID)

        cited_contents = live_citation_extractor.extract(
            results, document_titles=doc_titles
        )

        # At least one citation should have a real title (not the generic fallback)
        titles = [cc.citation.document_title for cc in cited_contents]
        non_generic = [t for t in titles if t != "Document"]
        assert len(non_generic) > 0, (
            f"At least one citation should have a real document title, "
            f"got all 'Document': {titles}"
        )

    def test_citation_format_valid(
        self, live_retriever, live_store, live_citation_extractor
    ):
        """Short and long citations should be non-empty with expected structure."""
        results = live_retriever.retrieve(
            query="termination provisions",
            client_id=DEMO_CLIENT_ID,
            top_k=3,
        )
        results = [r for r in results if r.hierarchy_path != "Document"]
        assert results, "Need results to test citation format"

        doc_ids = list(set(r.document_id for r in results))
        doc_titles = live_store.get_document_titles(doc_ids, client_id=DEMO_CLIENT_ID)

        cited_contents = live_citation_extractor.extract(
            results, document_titles=doc_titles
        )

        for cc in cited_contents:
            short = cc.citation.short_format()
            long = cc.citation.long_format()

            assert short, "short_citation should be non-empty"
            assert long, "long_citation should be non-empty"

            # Short citation should be bracketed
            assert short.startswith("["), f"Short citation should start with '[': {short}"
            assert short.endswith("]"), f"Short citation should end with ']': {short}"

            # Long citation should contain the document title
            assert cc.citation.document_title in long, (
                f"Long citation should contain document title.\n"
                f"Title: {cc.citation.document_title}\nLong: {long}"
            )
