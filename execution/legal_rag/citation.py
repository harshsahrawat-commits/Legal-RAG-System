"""
Citation Extraction and Formatting for Legal Documents

Professional legal citation formatting based on source type:
- Case law (cylaw):  [Title (Court, Case No., Year) [outcome]]
- HUDOC (ECHR):      [Title, App. No. XXXXX/XX, ECHR Year [outcome]]
- Legislation:       [Law Title, Section/Article]
- Generic fallback:  [Title, Section, Page N]

Supports bilingual (EN/EL) labels, score normalization, paragraph references,
and cross-lingual court name normalization (Greek → English display).
"""

import re
import logging
from typing import Optional
from dataclasses import dataclass

from .vector_store import SearchResult
from .language_config import TenantLanguageConfig
from .language_patterns import CITATION_SECTION_PATTERNS, LABELS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Greek-to-English court name mapping for normalized citation display.
# Keys are Greek court names (as they appear in metadata from cylaw/HUDOC
# documents).  Values are the English equivalents used in citations.
# ---------------------------------------------------------------------------
GREEK_COURT_NAME_MAP: dict[str, str] = {
    # Supreme / Constitutional
    "Ανώτατο Δικαστήριο": "Supreme Court",
    "Ανώτατο Συνταγματικό Δικαστήριο": "Supreme Constitutional Court",
    "ανώτατο δικαστήριο": "Supreme Court",
    "ανώτατο συνταγματικό δικαστήριο": "Supreme Constitutional Court",
    # Constitutional
    "Συνταγματικό Δικαστήριο": "Constitutional Court",
    "συνταγματικό δικαστήριο": "Constitutional Court",
    # Appeal
    "Εφετείο": "Court of Appeal",
    "εφετείο": "Court of Appeal",
    "Δευτεροβάθμιο Δικαστήριο": "Court of Appeal",
    "δευτεροβάθμιο δικαστήριο": "Court of Appeal",
    "Διοικητικό Εφετείο": "Administrative Court of Appeal",
    "διοικητικό εφετείο": "Administrative Court of Appeal",
    # Administrative
    "Διοικητικό Δικαστήριο": "Administrative Court",
    "διοικητικό δικαστήριο": "Administrative Court",
    # First Instance
    "Επαρχιακό Δικαστήριο": "District Court",
    "επαρχιακό δικαστήριο": "District Court",
    "Πρωτοδικείο": "Court of First Instance",
    "πρωτοδικείο": "Court of First Instance",
    "Πρωτόδικο Δικαστήριο": "Court of First Instance",
    "πρωτόδικο δικαστήριο": "Court of First Instance",
    "Δικαστήριο Εργατικών Διαφορών": "Labour Disputes Court",
    "δικαστήριο εργατικών διαφορών": "Labour Disputes Court",
    "Οικογενειακό Δικαστήριο": "Family Court",
    "οικογενειακό δικαστήριο": "Family Court",
    # ECHR / CJEU (occasionally stored in Greek)
    "ΕΔΔΑ": "ECHR",
    "Ευρωπαϊκό Δικαστήριο Ανθρωπίνων Δικαιωμάτων": "European Court of Human Rights",
    "ΔΕΕ": "CJEU",
    "Δικαστήριο της Ευρωπαϊκής Ένωσης": "Court of Justice of the European Union",
}


@dataclass
class Citation:
    """A formatted legal citation."""
    document_title: str
    section: str
    page_numbers: list[int]
    hierarchy_path: str
    chunk_id: str
    document_id: str
    relevance_score: float
    paragraph_numbers: list[int] = None  # Paragraph numbers for precise citation
    language: str = "en"  # Language for formatting labels
    source_metadata: dict = None  # Document-level metadata (source_origin, court_level, etc.)

    def __post_init__(self):
        if self.source_metadata is None:
            self.source_metadata = {}

    def _normalize_court_name(self, court: str) -> str:
        """Normalize a court name to English for consistent citation display.

        Handles Greek court names from cylaw metadata by mapping them to
        their English equivalents.  Returns the original string unchanged
        if no mapping is found (i.e. it is already in English or unknown).
        """
        if not court:
            return court
        # Exact match first (covers capitalised and lowercase variants)
        if court in GREEK_COURT_NAME_MAP:
            return GREEK_COURT_NAME_MAP[court]
        # Case-insensitive fallback: build a lowered lookup
        court_lower = court.lower().strip()
        for greek, english in GREEK_COURT_NAME_MAP.items():
            if greek.lower() == court_lower:
                return english
        return court

    def short_format(self) -> str:
        """Short inline citation in professional legal format.

        Format varies by source type:
        - Case law: Title (Court, Case No., Year) [outcome]
        - Legislation: Law Title, Section/Article
        - HUDOC: Title, App. No. XXXXX/XX, ECHR Year
        - Fallback: Title, Section, Page
        """
        origin = self.source_metadata.get("source_origin", "")
        doc_type = self.source_metadata.get("document_type", "")

        # Case law from cylaw
        if origin == "cylaw" and doc_type == "case_law":
            return self._format_case_law_citation()

        # HUDOC (always case law from ECHR)
        if origin == "hudoc":
            return self._format_hudoc_citation()

        # Legislation: cylaw statutes or eurlex
        if doc_type == "statute" or origin == "eurlex":
            return self._format_legislation_citation()

        # Fallback: generic format
        return self._format_generic_citation()

    def long_format(self) -> str:
        """Detailed citation with all available metadata."""
        origin = self.source_metadata.get("source_origin", "")
        page_str = self._format_pages()
        para_str = self._format_paragraphs()

        # Start with the professional short citation (without brackets)
        base = self.short_format().strip("[]")
        parts = [base]

        # Add section detail if not already included in the short citation
        if self.section and self.section not in base:
            parts.append(f"Section: {self.section}")

        if para_str and para_str not in base:
            parts.append(para_str)

        # Only add page numbers if not already present in the short citation
        if self.page_numbers and page_str not in base:
            parts.append(page_str)

        if self.hierarchy_path and origin not in ("hudoc", "eurlex"):
            parts.append(f"Path: {self.hierarchy_path}")

        return " | ".join(parts)

    # --- Professional legal citation formatters ---

    def _format_case_law_citation(self) -> str:
        """Format: Title (Court, Case No., Year) [outcome]

        Greek court names are normalized to English for consistent display.
        """
        meta = self.source_metadata
        parts_inner = []

        court = meta.get("court_level")
        if court:
            parts_inner.append(self._normalize_court_name(court))

        case_no = meta.get("case_number")
        if case_no:
            # Avoid duplicating "Case No." if already present in the value
            if re.match(r'(?i)case\s+no', str(case_no)):
                parts_inner.append(str(case_no))
            else:
                parts_inner.append(f"Case No. {case_no}")

        year = meta.get("year")
        if year:
            parts_inner.append(str(year))

        outcome = self._format_outcome_tag()

        if parts_inner:
            citation = f"{self.document_title} ({', '.join(parts_inner)})"
        else:
            citation = self.document_title

        if outcome:
            citation = f"{citation} [{outcome}]"

        return f"[{citation}]"

    def _format_hudoc_citation(self) -> str:
        """Format: Title, App. No. XXXXX/XX, ECHR Year"""
        meta = self.source_metadata
        parts = [self.document_title]

        app_no = meta.get("application_number") or meta.get("case_number")
        if app_no:
            # Normalize: if it already contains "App. No." don't duplicate
            if "app" not in str(app_no).lower():
                parts.append(f"App. No. {app_no}")
            else:
                parts.append(str(app_no))

        year = meta.get("year")
        if year:
            parts.append(f"ECHR {year}")
        else:
            parts.append("ECHR")

        outcome = self._format_outcome_tag()
        citation = ", ".join(parts)

        if outcome:
            citation = f"{citation} [{outcome}]"

        return f"[{citation}]"

    def _format_legislation_citation(self) -> str:
        """Format: Law Title, Section/Article

        Handles both English and Greek article/section references
        extracted from hierarchy paths.
        """
        parts = [self.document_title]

        if self.section:
            parts.append(self.section)
        elif self.hierarchy_path:
            # Try to extract article/section from hierarchy path
            extracted = self._extract_article_from_path()
            if extracted:
                parts.append(extracted)

        # Add paragraph references for precise legislation citations
        para_str = self._format_paragraphs()
        if para_str:
            parts.append(para_str)

        return f"[{', '.join(parts)}]"

    def _format_generic_citation(self) -> str:
        """Fallback format: Title, Section, Page"""
        page_str = self._format_pages()
        para_str = self._format_paragraphs()

        parts = [self.document_title]
        if self.section:
            parts.append(self.section)
        if para_str:
            parts.append(para_str)
        parts.append(page_str)

        return f"[{', '.join(parts)}]"

    def _format_outcome_tag(self) -> str:
        """Build a concise outcome tag from metadata.

        Checks all outcome fields in priority order and returns a short
        English label suitable for display in square brackets, e.g.
        'violation found', 'no violation', 'appeal dismissed',
        'annulment granted', 'unfair dismissal', 'abuse of discretion'.

        Returns empty string when no outcome metadata is available.
        """
        meta = self.source_metadata

        # --- violation_found can be True *or* explicitly False ---
        vf = meta.get("violation_found")
        if vf is True:
            return "violation found"
        if vf is False:
            return "no violation"

        # --- annulment ---
        if meta.get("annulment_granted"):
            return "annulment granted"

        # --- appeal outcome (stored as string: "dismissed", "allowed", etc.) ---
        appeal = meta.get("appeal_outcome")
        if appeal:
            appeal_str = str(appeal).strip().lower()
            if appeal_str in ("dismissed", "rejected", "denied", "inadmissible"):
                return "appeal dismissed"
            if appeal_str in ("allowed", "granted", "upheld"):
                return "appeal allowed"
            return f"appeal {appeal_str}"

        # --- appeal_dismissed (boolean flag from Phase 4 extraction) ---
        if meta.get("appeal_dismissed"):
            return "appeal dismissed"

        # --- extended outcome types ---
        if meta.get("abuse_of_discretion"):
            return "abuse of discretion"

        if meta.get("unfair_dismissal"):
            return "unfair dismissal"

        return ""

    def _extract_article_from_path(self) -> str:
        """Try to pull an article/section reference from the hierarchy path.

        Supports both English (Article/Art./Section) and Greek
        (Άρθρο/Τμήμα/Μέρος) patterns.
        """
        if not self.hierarchy_path:
            return ""
        # English patterns
        m = re.search(
            r'(?:Art(?:icle)?\.?\s*(\d+[\w.]*)|Section\s*(\d+[\w.]*))',
            self.hierarchy_path, re.IGNORECASE,
        )
        if m:
            num = m.group(1) or m.group(2)
            return f"Art. {num}"
        # Greek patterns: Άρθρο (Article), Τμήμα (Section), Μέρος (Part)
        m = re.search(
            r'(?:Άρθρο[_\s]+(\d+[\w.]*)|Τμήμα[_\s]+(\d+[\w.]*)|Μέρος[_\s]+([IVXLCDM]+|\d+))',
            self.hierarchy_path,
        )
        if m:
            num = m.group(1) or m.group(2) or m.group(3)
            return f"Art. {num}"
        return ""

    def _format_paragraphs(self) -> str:
        """Format paragraph numbers for citation."""
        if not self.paragraph_numbers:
            return ""
        if len(self.paragraph_numbers) == 1:
            return f"¶{self.paragraph_numbers[0]}"
        # Check if consecutive
        if self._is_consecutive(self.paragraph_numbers):
            return f"¶¶{self.paragraph_numbers[0]}-{self.paragraph_numbers[-1]}"
        return f"¶¶{', '.join(map(str, self.paragraph_numbers))}"

    def _format_pages(self) -> str:
        """Format page numbers using language-appropriate labels."""
        labels = LABELS.get(self.language, LABELS["en"])

        if not self.page_numbers:
            return labels["page_na"]

        if len(self.page_numbers) == 1:
            return f"{labels['page_single']} {self.page_numbers[0]}"

        # Check if consecutive
        if self._is_consecutive(self.page_numbers):
            return f"{labels['page_range']} {self.page_numbers[0]}-{self.page_numbers[-1]}"

        return f"{labels['page_range']} {', '.join(map(str, self.page_numbers))}"

    def _is_consecutive(self, nums: list[int]) -> bool:
        """Check if numbers are consecutive."""
        return len(nums) > 1 and nums == list(range(nums[0], nums[-1] + 1))

    def to_dict(self) -> dict:
        return {
            "document_title": self.document_title,
            "section": self.section,
            "page_numbers": self.page_numbers,
            "paragraph_numbers": self.paragraph_numbers,
            "hierarchy_path": self.hierarchy_path,
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "relevance_score": self.relevance_score,
            "short_citation": self.short_format(),
            "long_citation": self.long_format(),
        }


@dataclass
class CitedContent:
    """Content with its citation."""
    content: str
    citation: Citation
    context_before: str
    context_after: str

    def format_with_citation(self) -> str:
        """Format content with inline citation."""
        return f"{self.content}\n{self.citation.short_format()}"

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "citation": self.citation.to_dict(),
            "context_before": self.context_before,
            "context_after": self.context_after,
        }


class CitationExtractor:
    """
    Extracts and formats citations from search results.

    Provides consistent citation formatting for legal documents,
    supporting various output formats and languages.
    """

    def __init__(
        self,
        document_titles: Optional[dict] = None,
        language_config: Optional[TenantLanguageConfig] = None,
    ):
        """
        Initialize citation extractor.

        Args:
            document_titles: Optional mapping of document_id to title
            language_config: Per-tenant language configuration. Defaults to English.
        """
        self._document_titles = document_titles or {}
        self._language_config = language_config or TenantLanguageConfig.for_language("en")
        self._lang = self._language_config.language

    def extract(
        self,
        results: list[SearchResult],
        document_title: Optional[str] = None,
        document_titles: Optional[dict] = None,
        doc_metadata_map: Optional[dict] = None,
    ) -> list[CitedContent]:
        """
        Extract citations from search results.

        Args:
            results: List of search results
            document_title: Optional document title override (single title for all)
            document_titles: Optional mapping of document_id to title (per-document override)
            doc_metadata_map: Optional mapping of document_id to full document metadata
                dict from get_document_source_meta(). Each value has keys:
                "title", "file_path", "metadata" (the JSONB with source_origin,
                court_level, case_number, violation_found, etc.)

        Returns:
            List of CitedContent objects with formatted citations
        """
        titles = document_titles or self._document_titles
        meta_map = doc_metadata_map or {}
        cited_contents = []

        for result in results:
            # Get document title
            title = (
                document_title or
                titles.get(result.document_id) or
                self._extract_title_from_path(result.hierarchy_path) or
                "Document"
            )

            # Extract section from hierarchy path
            section = self._extract_section(result.hierarchy_path)

            # Get paragraph numbers from SearchResult field
            paragraph_numbers = getattr(result, "original_paragraph_numbers", None) or []

            # Build source_metadata for professional legal citation formatting.
            # Merge top-level doc fields (document_type) with the JSONB metadata
            # (source_origin, court_level, case_number, outcome fields, etc.)
            doc_entry = meta_map.get(result.document_id, {})
            jsonb_meta = doc_entry.get("metadata") or {}
            source_metadata = dict(jsonb_meta)  # copy so we don't mutate
            # Promote document_type from the top-level row if not already in JSONB
            if "document_type" not in source_metadata and doc_entry.get("document_type"):
                source_metadata["document_type"] = doc_entry["document_type"]

            # Use display score: prefer cosine similarity > rerank > ts_rank > raw
            # original_score is cosine similarity (0-1) — intuitive as percentage
            # rerank_score is Cohere's proprietary score (clusters 0.05-0.40, not a %)
            # ts_rank (FTS) can exceed 1.0, so cap it
            original = result.metadata.get("original_score")
            rerank = result.metadata.get("rerank_score")
            if original is not None and original <= 1.0:
                display_score = original
            elif rerank is not None:
                display_score = rerank
            elif original is not None:
                display_score = min(original, 0.99)
            else:
                display_score = result.score

            citation = Citation(
                document_title=title,
                section=section,
                page_numbers=result.page_numbers,
                hierarchy_path=result.hierarchy_path,
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                relevance_score=display_score,
                paragraph_numbers=paragraph_numbers,
                language=self._lang,
                source_metadata=source_metadata,
            )

            cited_content = CitedContent(
                content=result.content,
                citation=citation,
                context_before=result.metadata.get("context_before", ""),
                context_after=result.metadata.get("context_after", ""),
            )

            cited_contents.append(cited_content)

        # Normalize display scores to an intuitive range (65%-95%)
        # Raw cosine similarities for cross-lingual legal queries cluster 0.4-0.75,
        # which confuses users ("55% = barely relevant?"). Normalizing within each
        # batch maps the best result to ~95% and worst to ~65%, making scores
        # meaningful as relative confidence indicators.
        if len(cited_contents) > 1:
            scores = [cc.citation.relevance_score for cc in cited_contents]
            min_s, max_s = min(scores), max(scores)
            spread = max_s - min_s
            for cc in cited_contents:
                raw = cc.citation.relevance_score
                if spread > 0.01:
                    cc.citation.relevance_score = round(
                        0.65 + 0.30 * (raw - min_s) / spread, 3
                    )
                else:
                    cc.citation.relevance_score = 0.85
        elif len(cited_contents) == 1:
            # Single result: show high confidence
            cc = cited_contents[0]
            cc.citation.relevance_score = max(0.80, min(0.95, cc.citation.relevance_score))

        return cited_contents

    def _extract_title_from_path(self, path: str) -> Optional[str]:
        """Extract document title from hierarchy path."""
        if not path:
            return None

        parts = path.split("/")
        if parts:
            return parts[0].replace("_", " ")
        return None

    def _extract_section(self, hierarchy_path: str) -> str:
        """Extract section reference from hierarchy path (language-aware)."""
        if not hierarchy_path:
            return ""

        # Use language-specific patterns
        patterns = CITATION_SECTION_PATTERNS.get(self._lang, CITATION_SECTION_PATTERNS["en"])

        for pattern, template in patterns:
            match = re.search(pattern, hierarchy_path, re.IGNORECASE)
            if match:
                return template.format(match.group(1))

        # Return last part of path
        parts = hierarchy_path.split("/")
        if len(parts) > 1:
            return parts[-1].replace("_", " ")

        return ""



# CLI for testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # --- Case law example (cylaw) ---
    case_result = SearchResult(
        chunk_id="case-001",
        document_id="doc-case",
        content="The court held that the termination was unlawful...",
        section_title="Judgment",
        hierarchy_path="Yiallourou_v_Republic/Judgment/Para_42",
        page_numbers=[8],
        score=0.85,
        metadata={"context_before": "", "context_after": ""},
    )

    # --- HUDOC example ---
    hudoc_result = SearchResult(
        chunk_id="hudoc-001",
        document_id="doc-hudoc",
        content="The applicant alleged a violation of Article 6...",
        section_title="Merits",
        hierarchy_path="Yiallourou_v_Turkey/Merits",
        page_numbers=[],
        score=0.82,
        metadata={"context_before": "", "context_after": ""},
    )

    # --- Legislation example ---
    statute_result = SearchResult(
        chunk_id="stat-001",
        document_id="doc-statute",
        content="Every employee shall be entitled to...",
        section_title="Entitlements",
        hierarchy_path="Termination_of_Employment_Law/Part_II/Article_5",
        page_numbers=[3],
        score=0.78,
        metadata={"context_before": "", "context_after": ""},
    )

    # --- Generic (user-uploaded contract) ---
    generic_result = SearchResult(
        chunk_id="gen-001",
        document_id="doc-generic",
        content="The party may terminate this agreement upon 30 days written notice...",
        section_title="Termination",
        hierarchy_path="Contract/Part_II/Section_8/Article_3",
        page_numbers=[15, 16],
        score=0.89,
        metadata={"context_before": "", "context_after": ""},
    )

    # --- Greek court name case law (cylaw, court stored in Greek) ---
    greek_case_result = SearchResult(
        chunk_id="case-002",
        document_id="doc-greek-case",
        content="Το δικαστήριο αποφάσισε ότι η απόλυση ήταν παράνομη...",
        section_title="Απόφαση",
        hierarchy_path="Παπαδόπουλος_κατά_Δημοκρατίας/Απόφαση",
        page_numbers=[5],
        score=0.80,
        metadata={"context_before": "", "context_after": ""},
    )

    # --- Case with no violation (HUDOC, explicit False) ---
    hudoc_no_viol = SearchResult(
        chunk_id="hudoc-002",
        document_id="doc-hudoc-nv",
        content="The Court finds no violation of Article 8...",
        section_title="Judgment",
        hierarchy_path="Smith_v_Cyprus/Judgment",
        page_numbers=[],
        score=0.75,
        metadata={"context_before": "", "context_after": ""},
    )

    doc_meta = {
        "doc-case": {
            "title": "Yiallourou v. Republic",
            "file_path": None,
            "metadata": {
                "source_origin": "cylaw",
                "court_level": "Supreme Court",
                "case_number": "1234/2018",
                "year": 2019,
                "violation_found": True,
            },
            "document_type": "case_law",
        },
        "doc-hudoc": {
            "title": "Yiallourou v. Turkey",
            "file_path": None,
            "metadata": {
                "source_origin": "hudoc",
                "application_number": "69781/01",
                "year": 2019,
                "violation_found": True,
            },
            "document_type": "case_law",
        },
        "doc-statute": {
            "title": "Termination of Employment Law 24/1967",
            "file_path": None,
            "metadata": {"source_origin": "cylaw"},
            "document_type": "statute",
        },
        "doc-generic": {
            "title": "Software License Agreement",
            "file_path": None,
            "metadata": {"source_origin": "user"},
            "document_type": "contract",
        },
        "doc-greek-case": {
            "title": "Παπαδόπουλος κατά Δημοκρατίας",
            "file_path": None,
            "metadata": {
                "source_origin": "cylaw",
                "court_level": "Ανώτατο Δικαστήριο",
                "case_number": "567/2020",
                "year": 2021,
                "unfair_dismissal": True,
            },
            "document_type": "case_law",
        },
        "doc-hudoc-nv": {
            "title": "Smith v. Cyprus",
            "file_path": None,
            "metadata": {
                "source_origin": "hudoc",
                "application_number": "12345/22",
                "year": 2023,
                "violation_found": False,
            },
            "document_type": "case_law",
        },
    }

    doc_titles = {did: m["title"] for did, m in doc_meta.items()}

    extractor = CitationExtractor(document_titles=doc_titles)
    all_results = [
        case_result, hudoc_result, statute_result, generic_result,
        greek_case_result, hudoc_no_viol,
    ]

    cited_contents = extractor.extract(
        all_results, document_titles=doc_titles, doc_metadata_map=doc_meta,
    )

    for cc in cited_contents:
        print(f"\nContent: {cc.content[:80]}...")
        print(f"  Short: {cc.citation.short_format()}")
        print(f"  Long:  {cc.citation.long_format()}")
