"""
Shared fixtures and test utilities for Legal RAG System tests.

Provides mock services, sample data, and reusable fixtures so that all tests
can run without API keys, databases, or external network access.

Also registers the evaluation report plugin for the 5-dimension eval suite.
"""

import os
import sys
import uuid
import hashlib
from pathlib import Path
from unittest.mock import MagicMock
from datetime import datetime

import pytest
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Path setup - ensure the execution package is importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Sample legal document text (from existing test_pipeline.py)
# ---------------------------------------------------------------------------
SAMPLE_DOCUMENT = """
# SOFTWARE LICENSE AGREEMENT

This Software License Agreement ("Agreement") is entered into as of January 1, 2024
("Effective Date") by and between:

**LICENSOR:** TechCorp Inc., a Delaware corporation ("Licensor")

**LICENSEE:** ClientCo LLC, a California limited liability company ("Licensee")

## ARTICLE I - DEFINITIONS

Section 1.1 "Software" means the proprietary software application known as "LegalAI Pro"
including all updates, modifications, and enhancements.

Section 1.2 "Documentation" means user manuals, technical specifications, and other
materials describing the Software's functionality.

Section 1.3 "Licensed Users" means employees of Licensee authorized to use the Software.

## ARTICLE II - LICENSE GRANT

Section 2.1 Grant of License. Subject to the terms of this Agreement, Licensor hereby
grants to Licensee a non-exclusive, non-transferable license to use the Software.

Section 2.2 Restrictions. Licensee shall not:
(a) Copy, modify, or distribute the Software;
(b) Reverse engineer or decompile the Software;
(c) Sublicense or transfer the Software to third parties;
(d) Use the Software for any unlawful purpose.

## ARTICLE III - FEES AND PAYMENT

Section 3.1 License Fees. Licensee shall pay Licensor an annual license fee of
$50,000 USD, payable in advance.

Section 3.2 Payment Terms. All payments are due within thirty (30) days of invoice date.

Section 3.3 Late Payments. Overdue amounts shall accrue interest at 1.5% per month.

## ARTICLE IV - TERM AND TERMINATION

Section 4.1 Term. This Agreement shall commence on the Effective Date and continue
for a period of one (1) year, unless earlier terminated.

Section 4.2 Termination for Convenience. Either party may terminate this Agreement
upon sixty (60) days written notice.

Section 4.3 Termination for Breach. Either party may terminate immediately if the
other party materially breaches this Agreement and fails to cure within thirty (30) days.

Section 4.4 Effect of Termination. Upon termination:
(a) Licensee shall cease using the Software;
(b) Licensee shall return or destroy all copies of the Software;
(c) Sections 5, 6, and 7 shall survive termination.

## ARTICLE V - CONFIDENTIALITY

Section 5.1 Confidential Information. Each party agrees to maintain the confidentiality
of the other party's proprietary information.

Section 5.2 Permitted Disclosures. Confidential Information may be disclosed if required
by law or court order.

## ARTICLE VI - WARRANTIES

Section 6.1 Performance Warranty. Licensor warrants that the Software will perform
substantially in accordance with the Documentation for a period of ninety (90) days.

Section 6.2 Disclaimer. EXCEPT AS EXPRESSLY SET FORTH HEREIN, THE SOFTWARE IS PROVIDED
"AS IS" WITHOUT WARRANTY OF ANY KIND.

## ARTICLE VII - LIMITATION OF LIABILITY

Section 7.1 Cap on Damages. IN NO EVENT SHALL EITHER PARTY'S LIABILITY EXCEED THE
AMOUNTS PAID BY LICENSEE IN THE TWELVE (12) MONTHS PRECEDING THE CLAIM.

Section 7.2 Exclusion of Damages. NEITHER PARTY SHALL BE LIABLE FOR INDIRECT,
INCIDENTAL, OR CONSEQUENTIAL DAMAGES.

## ARTICLE VIII - GENERAL PROVISIONS

Section 8.1 Governing Law. This Agreement shall be governed by the laws of the
State of Delaware.

Section 8.2 Entire Agreement. This Agreement constitutes the entire agreement between
the parties regarding the subject matter hereof.

IN WITNESS WHEREOF, the parties have executed this Agreement as of the Effective Date.
"""


# ---------------------------------------------------------------------------
# Fixtures: data classes from document_parser
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_document_text():
    """Return the sample legal document markdown text."""
    return SAMPLE_DOCUMENT


@pytest.fixture
def sample_metadata():
    """Return a LegalMetadata instance for the sample document."""
    from execution.legal_rag.document_parser import LegalMetadata
    return LegalMetadata(
        document_id=str(uuid.uuid4()),
        title="Software License Agreement",
        document_type="contract",
        jurisdiction="Delaware",
        page_count=10,
        file_path="/tmp/test_document.pdf",
    )


@pytest.fixture
def sample_sections():
    """Return a list of DocumentSection instances parsed from sample text."""
    from execution.legal_rag.document_parser import DocumentSection
    sections = []
    current_section = None
    for line in SAMPLE_DOCUMENT.split("\n"):
        if line.startswith("## "):
            if current_section:
                sections.append(current_section)
            title = line[3:].strip()
            current_section = DocumentSection(
                section_id=str(uuid.uuid4()),
                title=title,
                content="",
                level=2,
                hierarchy_path=f"Document/{title.replace(' ', '_')}",
            )
        elif current_section:
            current_section.content += line + "\n"
    if current_section:
        sections.append(current_section)
    return sections


@pytest.fixture
def sample_parsed_document(sample_metadata, sample_sections):
    """Return a fully formed ParsedDocument for the sample contract."""
    from execution.legal_rag.document_parser import ParsedDocument
    return ParsedDocument(
        metadata=sample_metadata,
        sections=sample_sections,
        raw_text=SAMPLE_DOCUMENT,
        raw_markdown=SAMPLE_DOCUMENT,
    )


@pytest.fixture
def chunker():
    """Return a default LegalChunker instance."""
    from execution.legal_rag.chunker import LegalChunker
    return LegalChunker()


@pytest.fixture
def sample_chunks(chunker, sample_parsed_document):
    """Return chunks produced from the sample document."""
    return chunker.chunk(sample_parsed_document)


# ---------------------------------------------------------------------------
# Mock embedding service
# ---------------------------------------------------------------------------

class MockEmbeddingService:
    """Deterministic mock embedding service -- never calls external APIs."""

    def __init__(self, dimensions=1024):
        self._dimensions = dimensions
        self._call_count = 0

    def embed_documents(self, texts):
        return [self._deterministic_embedding(t) for t in texts]

    def embed_query(self, query):
        self._call_count += 1
        return self._deterministic_embedding(query)

    def _deterministic_embedding(self, text):
        h = hashlib.sha256(text.encode()).hexdigest()
        seed = int(h[:8], 16)
        return [((seed + i) % 1000) / 1000.0 for i in range(self._dimensions)]

    @property
    def dimensions(self):
        return self._dimensions


@pytest.fixture
def mock_embedding_service():
    return MockEmbeddingService(dimensions=1024)


# ---------------------------------------------------------------------------
# Mock vector store (no database needed)
# ---------------------------------------------------------------------------

class MockVectorStore:
    """In-memory mock of VectorStore for testing without PostgreSQL."""

    def __init__(self):
        self._documents = {}
        self._chunks = {}
        self._conn = MagicMock()
        self._pool = None
        self._current_tenant = None

    def connect(self):
        pass

    def initialize_schema(self):
        pass

    def insert_document(self, document_id, title, document_type, **kwargs):
        self._documents[document_id] = {
            "id": document_id, "title": title,
            "document_type": document_type, **kwargs,
        }

    def insert_chunks(self, chunks, embeddings, client_id=None):
        for chunk, emb in zip(chunks, embeddings):
            self._chunks[chunk["chunk_id"]] = {
                **chunk, "embedding": emb, "client_id": client_id,
            }

    def search(self, query_embedding, top_k=10, client_id=None,
               document_id=None, min_score=0.0):
        from execution.legal_rag.vector_store import SearchResult
        results = []
        for cid, chunk in list(self._chunks.items())[:top_k]:
            results.append(SearchResult(
                chunk_id=cid,
                document_id=chunk.get("document_id", ""),
                content=chunk.get("content", ""),
                section_title=chunk.get("section_title", ""),
                hierarchy_path=chunk.get("hierarchy_path", ""),
                page_numbers=chunk.get("page_numbers", []),
                score=0.85,
                metadata={"level": chunk.get("level", 0)},
                paragraph_start=chunk.get("paragraph_start"),
                paragraph_end=chunk.get("paragraph_end"),
                original_paragraph_numbers=chunk.get(
                    "original_paragraph_numbers", []),
            ))
        return results

    def keyword_search(self, query, top_k=10, client_id=None,
                       document_id=None, fts_language="english"):
        from execution.legal_rag.vector_store import SearchResult
        results = []
        for cid, chunk in self._chunks.items():
            content = chunk.get("content", "")
            if any(w in content.lower() for w in query.lower().split()):
                results.append(SearchResult(
                    chunk_id=cid,
                    document_id=chunk.get("document_id", ""),
                    content=content,
                    section_title=chunk.get("section_title", ""),
                    hierarchy_path=chunk.get("hierarchy_path", ""),
                    page_numbers=chunk.get("page_numbers", []),
                    score=0.5,
                    metadata={"level": chunk.get("level", 0)},
                ))
                if len(results) >= top_k:
                    break
        return results

    def list_documents(self, client_id=None):
        return list(self._documents.values())

    def delete_document(self, document_id, client_id=None):
        deleted = document_id in self._documents
        self._documents.pop(document_id, None)
        return deleted
        to_remove = [
            k for k, v in self._chunks.items()
            if v.get("document_id") == document_id
        ]
        for k in to_remove:
            del self._chunks[k]

    def set_tenant_context(self, client_id):
        self._current_tenant = client_id

    def clear_tenant_context(self):
        self._current_tenant = None

    def close(self):
        pass


@pytest.fixture
def mock_vector_store():
    return MockVectorStore()


# ---------------------------------------------------------------------------
# Search results for citation / retriever tests
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_search_results():
    """Return a list of SearchResult objects for testing."""
    from execution.legal_rag.vector_store import SearchResult
    return [
        SearchResult(
            chunk_id="chunk-001", document_id="doc-001",
            content="Section 4.2 Termination for Convenience. Either party may terminate this Agreement upon sixty (60) days written notice.",
            section_title="ARTICLE IV - TERM AND TERMINATION",
            hierarchy_path="Document/ARTICLE_IV_-_TERM_AND_TERMINATION/part_0",
            page_numbers=[4, 5], score=0.92,
            metadata={"level": 2, "legal_references": ["Section 4.2"],
                       "context_before": "The term shall be one year.",
                       "context_after": "Upon termination, licensee shall cease use."},
            paragraph_start=15, paragraph_end=17,
            original_paragraph_numbers=[15, 16, 17],
        ),
        SearchResult(
            chunk_id="chunk-002", document_id="doc-001",
            content="Section 4.3 Termination for Breach. Either party may terminate immediately if the other party materially breaches.",
            section_title="ARTICLE IV - TERM AND TERMINATION",
            hierarchy_path="Document/ARTICLE_IV_-_TERM_AND_TERMINATION/part_1",
            page_numbers=[5], score=0.87,
            metadata={"level": 2, "legal_references": ["Section 4.3"],
                       "context_before": "", "context_after": ""},
            paragraph_start=18, paragraph_end=18,
            original_paragraph_numbers=[18],
        ),
        SearchResult(
            chunk_id="chunk-003", document_id="doc-001",
            content="Section 7.1 Cap on Damages. IN NO EVENT SHALL EITHER PARTY'S LIABILITY EXCEED THE AMOUNTS PAID.",
            section_title="ARTICLE VII - LIMITATION OF LIABILITY",
            hierarchy_path="Document/ARTICLE_VII_-_LIMITATION_OF_LIABILITY",
            page_numbers=[8], score=0.71,
            metadata={"level": 2, "legal_references": ["Section 7.1"],
                       "context_before": "", "context_after": ""},
        ),
    ]


# ---------------------------------------------------------------------------
# Singleton resets between tests
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Greek document sample and fixtures
# ---------------------------------------------------------------------------

SAMPLE_GREEK_DOCUMENT = """
# ΣΥΜΒΑΣΗ ΑΔΕΙΑΣ ΛΟΓΙΣΜΙΚΟΥ

Η παρούσα Σύμβαση Άδειας Λογισμικού («Σύμβαση») συνάπτεται στις 1 Ιανουαρίου 2024
(«Ημερομηνία Έναρξης») μεταξύ:

**ΑΔΕΙΟΔΟΤΗΣ:** TechCorp ΑΕ, εταιρεία με έδρα την Αθήνα («Αδειοδότης»)

**ΑΔΕΙΟΥΧΟΣ:** ClientCo ΕΠΕ, εταιρεία περιορισμένης ευθύνης («Αδειούχος»)

## ΑΡΘΡΟ I - ΟΡΙΣΜΟΙ

Τμήμα 1.1 «Λογισμικό» σημαίνει την ιδιόκτητη εφαρμογή λογισμικού γνωστή ως "LegalAI Pro"
συμπεριλαμβανομένων όλων των ενημερώσεων, τροποποιήσεων και βελτιώσεων.

Τμήμα 1.2 «Τεκμηρίωση» σημαίνει εγχειρίδια χρήστη, τεχνικές προδιαγραφές
και άλλα υλικά που περιγράφουν τη λειτουργικότητα του Λογισμικού.

## ΑΡΘΡΟ II - ΠΑΡΑΧΩΡΗΣΗ ΑΔΕΙΑΣ

Τμήμα 2.1 Παραχώρηση Άδειας. Σύμφωνα με τους όρους της παρούσας Σύμβασης, ο Αδειοδότης
παραχωρεί στον Αδειούχο μη αποκλειστική, μη μεταβιβάσιμη άδεια χρήσης του Λογισμικού.

## ΑΡΘΡΟ III - ΤΕΡΜΑΤΙΣΜΟΣ

Τμήμα 3.1 Διάρκεια. Η παρούσα Σύμβαση τίθεται σε ισχύ κατά την Ημερομηνία Έναρξης
και συνεχίζεται για χρονικό διάστημα ενός (1) έτους.

Τμήμα 3.2 Καταγγελία. Κάθε μέρος μπορεί να καταγγείλει την παρούσα Σύμβαση
με εξήντα (60) ημερών γραπτή ειδοποίηση.

## ΑΡΘΡΟ IV - ΕΦΑΡΜΟΣΤΕΟ ΔΙΚΑΙΟ

Τμήμα 4.1 Η παρούσα Σύμβαση διέπεται από το δίκαιο της Ελληνικής Δημοκρατίας.
"""


@pytest.fixture
def sample_greek_document_text():
    """Return the sample Greek legal document markdown text."""
    return SAMPLE_GREEK_DOCUMENT


@pytest.fixture
def greek_language_config():
    """Return a TenantLanguageConfig for Greek."""
    from execution.legal_rag.language_config import TenantLanguageConfig
    return TenantLanguageConfig.for_language("el")


@pytest.fixture
def english_language_config():
    """Return a TenantLanguageConfig for English."""
    from execution.legal_rag.language_config import TenantLanguageConfig
    return TenantLanguageConfig.for_language("en")


# ---------------------------------------------------------------------------
# Singleton resets between tests
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_metrics_singleton():
    """Reset the MetricsCollector singleton between tests."""
    from execution.legal_rag.metrics import MetricsCollector
    MetricsCollector._instance = None
    yield
    MetricsCollector._instance = None


@pytest.fixture(autouse=True)
def reset_quota_manager_singleton():
    """Reset the global QuotaManager between tests."""
    import execution.legal_rag.quotas as quotas_mod
    quotas_mod._manager = None
    yield
    quotas_mod._manager = None


# ---------------------------------------------------------------------------
# Evaluation suite: constants, credential checks, live fixtures, report
# ---------------------------------------------------------------------------

ENGLISH_CLIENT_ID = "00000000-0000-0000-0000-000000000001"
GREEK_CLIENT_ID = "00000000-0000-0000-0000-000000000002"
GREEK_API_KEY = "lrag_3AnZ81BxLt_ro4pVtJGWNoasfB7I77BJRVBdI5D3qR4"
FAKE_CLIENT_ID = "ffffffff-ffff-ffff-ffff-ffffffffffff"


def _credentials_available() -> bool:
    return all(os.getenv(k) for k in ("POSTGRES_URL", "VOYAGE_API_KEY", "NVIDIA_API_KEY"))


skip_no_creds = pytest.mark.skipif(
    not _credentials_available(),
    reason="Missing POSTGRES_URL, VOYAGE_API_KEY, or NVIDIA_API_KEY in .env",
)


# --- Session-scoped live fixtures for integration tests ---

@pytest.fixture(scope="session")
def live_store():
    """Real VectorStore connected to the database."""
    from execution.legal_rag.vector_store import VectorStore
    store = VectorStore()
    store.connect()
    yield store
    store.close()


@pytest.fixture(scope="session")
def en_embeddings():
    """Real Voyage AI embedding service for English (voyage-law-2)."""
    from execution.legal_rag.embeddings import get_embedding_service
    from execution.legal_rag.language_config import TenantLanguageConfig
    return get_embedding_service(
        provider="voyage",
        language_config=TenantLanguageConfig.for_language("en"),
    )


@pytest.fixture(scope="session")
def el_embeddings():
    """Real Voyage AI embedding service for Greek (voyage-multilingual-2)."""
    from execution.legal_rag.embeddings import get_embedding_service
    from execution.legal_rag.language_config import TenantLanguageConfig
    return get_embedding_service(
        provider="voyage",
        language_config=TenantLanguageConfig.for_language("el"),
    )


@pytest.fixture(scope="session")
def en_retriever(live_store, en_embeddings):
    """Real HybridRetriever for English tenant."""
    from execution.legal_rag.retriever import HybridRetriever
    from execution.legal_rag.language_config import TenantLanguageConfig
    return HybridRetriever(
        live_store, en_embeddings,
        language_config=TenantLanguageConfig.for_language("en"),
    )


@pytest.fixture(scope="session")
def el_retriever(live_store, el_embeddings):
    """Real HybridRetriever for Greek tenant."""
    from execution.legal_rag.retriever import HybridRetriever
    from execution.legal_rag.language_config import TenantLanguageConfig
    return HybridRetriever(
        live_store, el_embeddings,
        language_config=TenantLanguageConfig.for_language("el"),
    )


@pytest.fixture(scope="session")
def live_llm():
    """Real OpenAI client pointed at NVIDIA NIM (Qwen 3 235B)."""
    from openai import OpenAI
    return OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.getenv("NVIDIA_API_KEY"),
        timeout=30.0,
    )


@pytest.fixture(scope="session")
def live_citation_extractor():
    """Real CitationExtractor."""
    from execution.legal_rag.citation import CitationExtractor
    return CitationExtractor()


@pytest.fixture(scope="session")
def live_api_client():
    """FastAPI TestClient backed by real services."""
    from fastapi.testclient import TestClient
    from execution.legal_rag.api import app
    return TestClient(app)


# --- Evaluation Report Plugin ---

from collections import defaultdict as _defaultdict

_DIMENSION_MAP = {
    "test_functionality_eval": "Functionality",
    "test_efficiency_eval": "Efficiency",
    "test_accuracy_eval": "Accuracy",
    "test_hallucination_eval": "Hallucination-Proofing",
    "test_speed_eval": "Speed",
}
_WEIGHTED_DIMENSIONS = {"Accuracy", "Hallucination-Proofing"}
_eval_results = _defaultdict(lambda: {"passed": 0, "failed": 0, "skipped": 0})


def _get_dimension(item) -> str:
    module_name = item.module.__name__ if hasattr(item, "module") and item.module else ""
    for prefix, dimension in _DIMENSION_MAP.items():
        if prefix in module_name:
            return dimension
    return ""


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.when == "call":
        dim = _get_dimension(item)
        if dim:
            _eval_results[dim]["passed" if report.passed else "failed"] += 1
    elif report.when == "setup" and report.skipped:
        dim = _get_dimension(item)
        if dim:
            _eval_results[dim]["skipped"] += 1


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if not any(_eval_results.values()):
        return
    lines = [
        "",
        "=" * 60,
        "     LEGAL RAG EVALUATION REPORT",
        "=" * 60,
    ]
    total_weight = 0.0
    total_earned = 0.0
    for dim in ["Functionality", "Efficiency", "Accuracy", "Hallucination-Proofing", "Speed"]:
        r = _eval_results.get(dim)
        if not r:
            lines.append(f"  {dim:<28s}  -- not run --")
            continue
        total = r["passed"] + r["failed"]
        score = round(r["passed"] / total * 100, 1) if total else 0.0
        w = 2.0 if dim in _WEIGHTED_DIMENSIONS else 1.0
        if total:
            total_weight += w
            total_earned += w * (r["passed"] / total)
        sk = f"  ({r['skipped']} skipped)" if r["skipped"] else ""
        lines.append(f"  {dim:<28s} {score:5.1f}%  ({r['passed']}/{total} passed){sk}")
    overall = round(total_earned / total_weight * 100, 1) if total_weight else 0.0
    lines.append("-" * 60)
    lines.append(f"  {'OVERALL SCORE':<28s} {overall:5.1f}%")
    lines.append("=" * 60)
    terminalreporter.write("\n".join(lines) + "\n")


# --- Helper: run full RAG pipeline ---

def run_rag_pipeline(query, retriever, llm, citation_extractor, store, client_id):
    """Run the full RAG pipeline and return (answer, cited_contents)."""
    import re as _re
    from execution.legal_rag.language_patterns import LLM_PROMPTS

    results = retriever.retrieve(query=query, client_id=client_id, top_k=5)
    results = [r for r in results if r.hierarchy_path != "Document"]

    if not results:
        return "No relevant information found in the provided documents.", []

    doc_ids = list(set(r.document_id for r in results))
    doc_titles = store.get_document_titles(doc_ids, client_id=client_id)

    cited_contents = citation_extractor.extract(results, document_titles=doc_titles)

    context = "\n\n---\n\n".join([
        f"**[{i+1}]** {cc.citation.short_format()}:\n{cc.content}"
        for i, cc in enumerate(cited_contents)
    ])

    lang = "el" if any('\u0370' <= c <= '\u03ff' for c in query) else "en"
    system_prompt = LLM_PROMPTS[lang]["rag_system"]

    response = llm.chat.completions.create(
        model="qwen/qwen3-235b-a22b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": (
                f"Based on the following sources, answer this question: {query}\n\n"
                f"SOURCES:\n{context}\n\nProvide a clear, well-cited answer."
            )},
        ],
        max_tokens=1500,
        temperature=0.2,
    )
    answer = response.choices[0].message.content
    return answer, cited_contents
