"""
Citation Extraction and Formatting for Legal Documents

Formats search results with proper legal citations:
[Document Title, Section X.Y, Page N]

Supports various citation styles and formats.
"""

import re
import logging
from typing import Optional
from dataclasses import dataclass

from .vector_store import SearchResult
from .language_config import TenantLanguageConfig
from .language_patterns import CITATION_SECTION_PATTERNS, LABELS

logger = logging.getLogger(__name__)


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

    def short_format(self) -> str:
        """Short inline citation format."""
        page_str = self._format_pages()
        para_str = self._format_paragraphs()

        parts = [self.document_title]
        if self.section:
            parts.append(self.section)
        if para_str:
            parts.append(para_str)
        parts.append(page_str)

        return f"[{', '.join(parts)}]"

    def long_format(self) -> str:
        """Detailed citation format."""
        page_str = self._format_pages()
        para_str = self._format_paragraphs()
        parts = [self.document_title]

        if self.hierarchy_path:
            parts.append(f"Path: {self.hierarchy_path}")

        if self.section:
            parts.append(f"Section: {self.section}")

        if para_str:
            parts.append(para_str)

        parts.append(page_str)
        return " | ".join(parts)

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
    ) -> list[CitedContent]:
        """
        Extract citations from search results.

        Args:
            results: List of search results
            document_title: Optional document title override (single title for all)
            document_titles: Optional mapping of document_id to title (per-document override)

        Returns:
            List of CitedContent objects with formatted citations
        """
        titles = document_titles or self._document_titles
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
            )

            cited_content = CitedContent(
                content=result.content,
                citation=citation,
                context_before=result.metadata.get("context_before", ""),
                context_after=result.metadata.get("context_after", ""),
            )

            cited_contents.append(cited_content)

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

    # Create sample search result
    result = SearchResult(
        chunk_id="test-123",
        document_id="doc-456",
        content="The party may terminate this agreement upon 30 days written notice...",
        section_title="Termination",
        hierarchy_path="Contract/Part_II/Section_8/Article_3",
        page_numbers=[15, 16],
        score=0.89,
        metadata={
            "context_before": "Prior to termination, all obligations must be fulfilled.",
            "context_after": "Upon termination, the following provisions shall survive.",
        },
    )

    # Extract citation
    extractor = CitationExtractor(
        document_titles={"doc-456": "Software License Agreement"}
    )

    cited_contents = extractor.extract([result])

    for cc in cited_contents:
        print(f"\nContent: {cc.content[:100]}...")
        print(f"Short citation: {cc.citation.short_format()}")
        print(f"Long citation: {cc.citation.long_format()}")
