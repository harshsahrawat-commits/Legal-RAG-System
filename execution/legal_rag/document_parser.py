"""
Legal Document Parser - Extracts structured content from legal PDFs

Uses Docling for high-accuracy PDF extraction (97.9% on complex tables).
Preserves legal document structure: articles, clauses, sections, appendices.
"""

import os
import re
import uuid
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
import html

logger = logging.getLogger(__name__)


@dataclass
class LegalMetadata:
    """Metadata extracted from legal documents."""
    document_id: str
    title: str
    document_type: str  # contract, statute, case_law, regulation, brief, memo
    jurisdiction: Optional[str] = None
    effective_date: Optional[datetime] = None
    parties: list[str] = field(default_factory=list)
    page_count: int = 0
    file_path: str = ""
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "document_id": self.document_id,
            "title": self.title,
            "document_type": self.document_type,
            "jurisdiction": self.jurisdiction,
            "effective_date": self.effective_date.isoformat() if self.effective_date else None,
            "parties": self.parties,
            "page_count": self.page_count,
            "file_path": self.file_path,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class DocumentSection:
    """A section of a legal document with hierarchy information."""
    section_id: str
    title: str
    content: str
    level: int  # 0=document, 1=part/chapter, 2=section, 3=article, 4=clause
    parent_id: Optional[str] = None
    page_numbers: list[int] = field(default_factory=list)
    hierarchy_path: str = ""  # e.g., "Part_II/Section_4/Article_3"

    def to_dict(self) -> dict:
        return {
            "section_id": self.section_id,
            "title": self.title,
            "content": self.content,
            "level": self.level,
            "parent_id": self.parent_id,
            "page_numbers": self.page_numbers,
            "hierarchy_path": self.hierarchy_path,
        }


@dataclass
class ParsedDocument:
    """Complete parsed legal document with structure."""
    metadata: LegalMetadata
    sections: list[DocumentSection]
    raw_text: str
    raw_markdown: str

    def to_dict(self) -> dict:
        return {
            "metadata": self.metadata.to_dict(),
            "sections": [s.to_dict() for s in self.sections],
            "raw_text": self.raw_text,
            "raw_markdown": self.raw_markdown,
        }


class LegalDocumentParser:
    """
    Parses legal documents with structure preservation.

    Supports PDF extraction via Docling with fallback to PyMuPDF.
    Automatically detects document type and extracts legal structure.
    """

    # Patterns for detecting legal document structure
    SECTION_PATTERNS = {
        "part": re.compile(r"^(?:PART|Part)\s+([IVXLCDM]+|\d+)[.:\s]", re.MULTILINE),
        "chapter": re.compile(r"^(?:CHAPTER|Chapter)\s+(\d+|[IVXLCDM]+)[.:\s]", re.MULTILINE),
        "section": re.compile(r"^(?:SECTION|Section|§)\s*(\d+(?:\.\d+)*)[.:\s]", re.MULTILINE),
        "article": re.compile(r"^(?:ARTICLE|Article)\s+(\d+|[IVXLCDM]+)[.:\s]", re.MULTILINE),
        "clause": re.compile(r"^(?:Clause|\d+\.)\s*(\d+(?:\.\d+)*)[.:\s]", re.MULTILINE),
    }

    # Patterns for document type detection
    DOCTYPE_PATTERNS = {
        "contract": [
            r"(?i)agreement|contract|terms and conditions|parties agree",
            r"(?i)witnesseth|whereas|now therefore",
            r"(?i)executed as of|effective date",
        ],
        "statute": [
            r"(?i)enacted by|legislature|public law",
            r"(?i)be it enacted|section \d+\.",
            r"(?i)code of|statutes at large",
        ],
        "case_law": [
            r"(?i)plaintiff|defendant|court|judge",
            r"(?i)opinion|held|judgment|ruling",
            r"(?i)appeal|appellant|appellee",
        ],
        "regulation": [
            r"(?i)regulation|rule|agency|federal register",
            r"(?i)promulgated|pursuant to",
            r"(?i)cfr|code of federal regulations",
        ],
    }

    def __init__(self, use_docling: bool = True):
        """
        Initialize the parser.

        Args:
            use_docling: Whether to use Docling for PDF extraction.
                        Falls back to PyMuPDF if Docling unavailable.
        """
        self.use_docling = use_docling
        self._docling_available = False
        self._pymupdf_available = False

        # Try to import extraction libraries
        try:
            from docling.document_converter import DocumentConverter
            self._docling_converter = DocumentConverter()
            self._docling_available = True
            logger.info("Docling initialized successfully")
        except ImportError:
            logger.warning("Docling not available, will try PyMuPDF")

        try:
            import pymupdf4llm
            self._pymupdf_available = True
            logger.info("PyMuPDF4LLM available as fallback")
        except ImportError:
            logger.warning("PyMuPDF4LLM not available")

        if not self._docling_available and not self._pymupdf_available:
            raise ImportError(
                "Neither Docling nor PyMuPDF available. "
                "Install with: pip install docling pymupdf4llm"
            )

    def parse(self, file_path: str, client_id: Optional[str] = None) -> ParsedDocument:
        """
        Parse a legal document from file.

        Args:
            file_path: Path to the PDF or document file
            client_id: Optional client ID for multi-tenant tracking

        Returns:
            ParsedDocument with extracted structure and content
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

        logger.info(f"Parsing document: {path.name}")

        # Extract raw content with page tracking
        if self._docling_available and self.use_docling:
            raw_markdown, page_count, page_ranges = self._extract_with_docling(file_path)
        else:
            raw_markdown, page_count, page_ranges = self._extract_with_pymupdf(file_path)

        # Clean up the text
        raw_text = self._markdown_to_text(raw_markdown)

        # Detect document type
        doc_type = self._detect_document_type(raw_text)

        # Extract title (first heading or first line)
        title = self._extract_title(raw_markdown, path.stem)

        # Create metadata
        metadata = LegalMetadata(
            document_id=str(uuid.uuid4()),
            title=title,
            document_type=doc_type,
            page_count=page_count,
            file_path=str(path.absolute()),
        )

        # Extract document structure with page tracking
        sections = self._extract_sections(raw_markdown, metadata.document_id, page_ranges)

        # Try to extract additional metadata
        metadata.jurisdiction = self._extract_jurisdiction(raw_text)
        metadata.parties = self._extract_parties(raw_text, doc_type)
        metadata.effective_date = self._extract_date(raw_text)

        return ParsedDocument(
            metadata=metadata,
            sections=sections,
            raw_text=raw_text,
            raw_markdown=raw_markdown,
        )

    def _extract_with_docling(self, file_path: str) -> tuple[str, int, list[tuple[int, int, int]]]:
        """
        Extract content using Docling.

        Returns:
            tuple: (markdown, page_count, page_ranges)
                page_ranges is a list of (page_num, start_char, end_char) tuples
        """
        from docling.document_converter import DocumentConverter

        result = self._docling_converter.convert(file_path)
        markdown = result.document.export_to_markdown()

        # Get page count from result
        page_count = len(result.document.pages) if hasattr(result.document, 'pages') else 0

        # Docling doesn't provide easy character-to-page mapping, so estimate evenly
        # This is a rough approximation - for better accuracy, use PyMuPDF
        page_ranges = []
        if page_count > 0:
            chars_per_page = len(markdown) // page_count
            for page_num in range(page_count):
                start = page_num * chars_per_page
                end = (page_num + 1) * chars_per_page if page_num < page_count - 1 else len(markdown)
                page_ranges.append((page_num + 1, start, end))

        return markdown, page_count, page_ranges

    def _extract_with_pymupdf(self, file_path: str) -> tuple[str, int, list[tuple[int, int, int]]]:
        """
        Extract content using PyMuPDF4LLM with page tracking.

        Returns:
            tuple: (markdown, page_count, page_ranges)
                page_ranges is a list of (page_num, start_char, end_char) tuples
        """
        import pymupdf4llm
        import fitz  # PyMuPDF

        # Get markdown content
        markdown = pymupdf4llm.to_markdown(file_path)

        # Get page count and build page-to-text mapping
        page_ranges = []
        with fitz.open(file_path) as doc:
            page_count = len(doc)

            # Extract text per page to estimate character ranges
            cumulative_char = 0
            for page_num in range(page_count):
                page = doc[page_num]
                page_text = page.get_text()
                text_len = len(page_text)

                # Store (1-indexed page, start_char, end_char)
                page_ranges.append((page_num + 1, cumulative_char, cumulative_char + text_len))
                cumulative_char += text_len

        return markdown, page_count, page_ranges

    def _markdown_to_text(self, markdown: str) -> str:
        """Convert markdown to plain text."""
        # Remove markdown formatting
        text = re.sub(r'#+\s*', '', markdown)  # Headers
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*([^*]+)\*', r'\1', text)  # Italic
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)  # Links
        text = re.sub(r'`([^`]+)`', r'\1', text)  # Code
        text = re.sub(r'!\[([^\]]*)\]\([^)]+\)', '', text)  # Images
        
        # Clean OCR artifacts
        text = re.sub(r'GLYPH&lt;\d+&gt;', '', text)  # HTML encoded GLYPH
        text = re.sub(r'GLYPH<\d+>', '', text)  # Raw GLYPH
        text = re.sub(r'<!--.*?-->', '', text)  # HTML comments/placeholders
        text = re.sub(r'\[image\]', '', text, flags=re.IGNORECASE)
        
        # Unescape HTML entities (e.g. &amp; -> &)
        text = html.unescape(text)
        
        return text.strip()

    def _detect_document_type(self, text: str) -> str:
        """Detect the type of legal document."""
        scores = {}

        for doc_type, patterns in self.DOCTYPE_PATTERNS.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, text[:5000]))  # Check first 5000 chars
                score += matches
            scores[doc_type] = score

        if max(scores.values()) > 0:
            return max(scores, key=scores.get)
        return "unknown"

    def _extract_title(self, markdown: str, fallback: str) -> str:
        """Extract document title from content."""
        # Look for first heading
        match = re.search(r'^#+\s*(.+)$', markdown, re.MULTILINE)
        if match:
            return match.group(1).strip()

        # Look for title-like first line (all caps or mixed case, no period)
        lines = markdown.strip().split('\n')
        for line in lines[:8]:  # Check first 8 lines
            line = line.strip()
            # Remove markdown chars for validation
            clean_line = re.sub(r'[#*`]', '', line).strip()
            
            if not clean_line or len(clean_line) < 3 or len(clean_line) > 200:
                continue
                
            # Heuristics for valid legal title:
            # 1. Mostly letters (avoid "123.456")
            # 2. Contains spaces (avoid "wesruefr,rodffif")
            # 3. No excessive punctuation
            
            letter_count = len(re.findall(r'[a-zA-Z]', clean_line))
            if letter_count / len(clean_line) < 0.4:
                continue
                
            if ' ' not in clean_line and len(clean_line) > 20: # Long string with no spaces is suspicious
                continue

            if clean_line.endswith('.'):
                continue
                
            return clean_line

        return fallback

    def _extract_sections(
        self,
        markdown: str,
        document_id: str,
        page_ranges: list[tuple[int, int, int]] = None
    ) -> list[DocumentSection]:
        """
        Extract hierarchical sections from document.

        Args:
            markdown: Document content in markdown format
            document_id: ID of the parent document
            page_ranges: List of (page_num, start_char, end_char) for page tracking
        """
        sections = []
        current_hierarchy = {}
        page_ranges = page_ranges or []

        # Split by markdown headers
        header_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)

        # Find all headers with their positions
        headers = []
        for match in header_pattern.finditer(markdown):
            level = len(match.group(1))
            title = match.group(2).strip()
            start = match.end()
            headers.append((level, title, start, match.start()))

        # Extract content between headers
        for i, (level, title, start, header_start) in enumerate(headers):
            # Find end of this section
            end = headers[i + 1][3] if i + 1 < len(headers) else len(markdown)
            content = markdown[start:end].strip()

            # Build hierarchy path
            current_hierarchy[level] = title
            # Clear deeper levels
            for l in list(current_hierarchy.keys()):
                if l > level:
                    del current_hierarchy[l]

            hierarchy_path = "/".join(
                current_hierarchy.get(l, "")
                for l in sorted(current_hierarchy.keys())
                if current_hierarchy.get(l)
            )

            # Find parent
            parent_id = None
            if level > 1:
                for s in reversed(sections):
                    if s.level < level:
                        parent_id = s.section_id
                        break

            # Calculate page numbers based on character positions
            page_numbers = self._calculate_page_numbers(header_start, end, page_ranges)

            section = DocumentSection(
                section_id=str(uuid.uuid4()),
                title=title,
                content=content,
                level=level,
                parent_id=parent_id,
                hierarchy_path=hierarchy_path,
                page_numbers=page_numbers,
            )
            sections.append(section)

        # If no headers found, create a single section with all content
        if not sections:
            # All pages for single section
            all_pages = sorted(set(pr[0] for pr in page_ranges)) if page_ranges else []
            sections.append(DocumentSection(
                section_id=str(uuid.uuid4()),
                title="Document Content",
                content=markdown,
                level=0,
                hierarchy_path="Document",
                page_numbers=all_pages if all_pages else [],
            ))

        return sections

    def _calculate_page_numbers(
        self,
        start_char: int,
        end_char: int,
        page_ranges: list[tuple[int, int, int]]
    ) -> list[int]:
        """
        Calculate which pages a section spans based on character positions.

        Args:
            start_char: Start character position in markdown
            end_char: End character position in markdown
            page_ranges: List of (page_num, start_char, end_char) tuples

        Returns:
            List of page numbers (1-indexed) that the section spans
        """
        if not page_ranges:
            return []

        pages = set()
        total_doc_chars = page_ranges[-1][2] if page_ranges else 0

        for page_num, page_start, page_end in page_ranges:
            # Check if section overlaps with this page's character range
            # Use proportional mapping since markdown != raw text exactly
            if total_doc_chars > 0:
                # Normalize positions to account for markdown/text differences
                section_overlaps = (
                    start_char < page_end and end_char > page_start
                )
                if section_overlaps:
                    pages.add(page_num)

        return sorted(pages) if pages else [1]  # Default to page 1 if can't determine

    def _extract_jurisdiction(self, text: str) -> Optional[str]:
        """Extract jurisdiction from document text."""
        # Common jurisdiction patterns
        patterns = [
            r"(?i)state of ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
            r"(?i)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:court|district|circuit)",
            r"(?i)laws of (?:the state of )?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
            r"(?i)governed by (?:the laws of )?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text[:3000])
            if match:
                return match.group(1).strip()

        return None

    def _extract_parties(self, text: str, doc_type: str) -> list[str]:
        """Extract party names from contracts."""
        if doc_type != "contract":
            return []

        parties = []

        # Look for common party patterns
        patterns = [
            r'(?i)(?:between|by and between)\s+([A-Z][A-Za-z\s,\.]+?)(?:\s+\("|\s+and\s+)',
            r'(?i)"([A-Z][A-Za-z]+)"\s*(?:,\s*)?(?:a|an|the)',
            r'(?i)(?:hereinafter|referred to as)\s+"([A-Z][A-Za-z]+)"',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text[:2000])
            parties.extend(m.strip() for m in matches if len(m.strip()) < 100)

        return list(set(parties))[:5]  # Dedupe and limit

    def _extract_date(self, text: str) -> Optional[datetime]:
        """Extract effective date from document."""
        patterns = [
            r'(?i)effective\s+(?:as\s+of\s+)?(\w+\s+\d{1,2},?\s+\d{4})',
            r'(?i)dated\s+(?:as\s+of\s+)?(\w+\s+\d{1,2},?\s+\d{4})',
            r'(?i)this\s+(\d{1,2}(?:st|nd|rd|th)?\s+day\s+of\s+\w+,?\s+\d{4})',
        ]

        for pattern in patterns:
            match = re.search(pattern, text[:3000])
            if match:
                date_str = match.group(1)
                # Try parsing common formats
                formats = [
                    "%B %d, %Y",
                    "%B %d %Y",
                    "%d day of %B, %Y",
                    "%d day of %B %Y",
                ]
                for fmt in formats:
                    try:
                        # Clean up ordinals
                        clean_date = re.sub(r'(\d+)(?:st|nd|rd|th)', r'\1', date_str)
                        return datetime.strptime(clean_date, fmt)
                    except ValueError:
                        continue

        return None


# CLI for testing
if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python document_parser.py <pdf_path>")
        sys.exit(1)

    logging.basicConfig(level=logging.INFO)

    parser = LegalDocumentParser()
    result = parser.parse(sys.argv[1])

    print(json.dumps(result.to_dict(), indent=2, default=str))
