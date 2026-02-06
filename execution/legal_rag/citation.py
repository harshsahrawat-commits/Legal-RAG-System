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
        """Format page numbers."""
        if not self.page_numbers:
            return "Page N/A"

        if len(self.page_numbers) == 1:
            return f"p. {self.page_numbers[0]}"

        # Check if consecutive
        if self._is_consecutive(self.page_numbers):
            return f"pp. {self.page_numbers[0]}-{self.page_numbers[-1]}"

        return f"pp. {', '.join(map(str, self.page_numbers))}"

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
    supporting various output formats.
    """

    def __init__(self, document_titles: Optional[dict] = None):
        """
        Initialize citation extractor.

        Args:
            document_titles: Optional mapping of document_id to title
        """
        self._document_titles = document_titles or {}

    def extract(
        self,
        results: list[SearchResult],
        document_title: Optional[str] = None,
    ) -> list[CitedContent]:
        """
        Extract citations from search results.

        Args:
            results: List of search results
            document_title: Optional document title override

        Returns:
            List of CitedContent objects with formatted citations
        """
        cited_contents = []

        for result in results:
            # Get document title
            title = (
                document_title or
                self._document_titles.get(result.document_id) or
                self._extract_title_from_path(result.hierarchy_path) or
                "Document"
            )

            # Extract section from hierarchy path
            section = self._extract_section(result.hierarchy_path)

            # Get paragraph numbers from SearchResult field
            paragraph_numbers = getattr(result, "original_paragraph_numbers", None) or []

            citation = Citation(
                document_title=title,
                section=section,
                page_numbers=result.page_numbers,
                hierarchy_path=result.hierarchy_path,
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                relevance_score=result.score,
                paragraph_numbers=paragraph_numbers,
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
        """Extract section reference from hierarchy path."""
        if not hierarchy_path:
            return ""

        # Look for common patterns
        patterns = [
            (r"Article[_\s]+([IVXLCDM]+|\d+)", "Article {}"),
            (r"Section[_\s]+(\d+(?:\.\d+)*)", "Section {}"),
            (r"Clause[_\s]+(\d+(?:\.\d+)*)", "Clause {}"),
            (r"Part[_\s]+([IVXLCDM]+|\d+)", "Part {}"),
            (r"Chapter[_\s]+(\d+)", "Chapter {}"),
        ]

        for pattern, template in patterns:
            match = re.search(pattern, hierarchy_path, re.IGNORECASE)
            if match:
                return template.format(match.group(1))

        # Return last part of path
        parts = hierarchy_path.split("/")
        if len(parts) > 1:
            return parts[-1].replace("_", " ")

        return ""

    def format_response_with_citations(
        self,
        response_text: str,
        cited_contents: list[CitedContent],
    ) -> str:
        """
        Format a response with citations appended.

        Args:
            response_text: The AI-generated response
            cited_contents: List of cited contents used in response

        Returns:
            Response text with sources section
        """
        if not cited_contents:
            return response_text

        # Build sources section
        sources = "\n\n---\n**Sources:**\n"
        seen_citations = set()

        for i, cc in enumerate(cited_contents, 1):
            citation_key = (cc.citation.document_title, cc.citation.section)
            if citation_key not in seen_citations:
                sources += f"{i}. {cc.citation.long_format()}\n"
                seen_citations.add(citation_key)

        return response_text + sources

    def create_citation_index(
        self,
        cited_contents: list[CitedContent],
    ) -> dict:
        """
        Create an indexed map of citations for reference.

        Returns:
            Dictionary mapping citation numbers to CitedContent
        """
        return {
            i: cc.to_dict()
            for i, cc in enumerate(cited_contents, 1)
        }


class LegalCitationFormatter:
    """
    Formats citations in standard legal citation styles.

    Supports:
    - Bluebook format (US)
    - OSCOLA format (UK)
    - Simple inline format
    """

    @staticmethod
    def bluebook(citation: Citation) -> str:
        """Format citation in Bluebook style."""
        # Simplified Bluebook format
        parts = []

        if citation.document_title:
            parts.append(citation.document_title)

        if citation.section:
            parts.append(f"§ {citation.section.replace('Section ', '')}")

        if citation.page_numbers:
            if len(citation.page_numbers) == 1:
                parts.append(f"at {citation.page_numbers[0]}")
            else:
                parts.append(
                    f"at {citation.page_numbers[0]}-{citation.page_numbers[-1]}"
                )

        return ", ".join(parts) + "."

    @staticmethod
    def oscola(citation: Citation) -> str:
        """Format citation in OSCOLA style (UK)."""
        parts = [citation.document_title]

        if citation.section:
            parts.append(f"s {citation.section.replace('Section ', '')}")

        return ", ".join(parts)

    @staticmethod
    def inline(citation: Citation) -> str:
        """Format as inline citation."""
        return citation.short_format()


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
        print(f"Bluebook: {LegalCitationFormatter.bluebook(cc.citation)}")
