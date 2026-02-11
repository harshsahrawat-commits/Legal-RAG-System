"""
Legal Document Parser - Extracts structured content from legal PDFs

Uses PyMuPDF4LLM for PDF extraction with legal structure preservation.
Automatically detects document type and extracts articles, clauses, sections, appendices.
"""

import os
# Set cache paths to a writable directory for cloud deployment
os.environ["HF_HOME"] = "/tmp/huggingface"
os.environ["XDG_CACHE_HOME"] = "/tmp/cache"
import re
import uuid
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
import html

from .language_config import TenantLanguageConfig
from .language_patterns import (
    DOCTYPE_PATTERNS,
    SECTION_PATTERNS,
    DATE_PATTERNS,
    PARTY_PATTERNS,
    JURISDICTION_PATTERNS,
    TITLE_LETTER_REGEX,
)

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

    Uses PyMuPDF4LLM for PDF extraction (with optional Docling fallback).
    Automatically detects document type and extracts legal structure.
    Supports multilingual documents via language_config.
    """

    def __init__(
        self,
        language_config: Optional[TenantLanguageConfig] = None,
        use_docling: Optional[bool] = None,
        enable_ocr: bool = True,
    ):
        """
        Initialize the parser.

        Args:
            language_config: Per-tenant language configuration. Defaults to English.
            use_docling: Force Docling on/off. None = auto-detect by RAM.
            enable_ocr: Enable Surya OCR fallback for scanned PDFs. Set False to skip.
        """
        self._enable_ocr = enable_ocr
        self._language_config = language_config or TenantLanguageConfig.for_language("en")
        self._lang = self._language_config.language

        # Load language-specific patterns from centralized module
        self._doctype_patterns = DOCTYPE_PATTERNS.get(self._lang, DOCTYPE_PATTERNS["en"])
        self._section_patterns = SECTION_PATTERNS.get(self._lang, SECTION_PATTERNS["en"])

        self._pymupdf_available = False
        self._docling_available = False
        self._surya_available = False
        self._use_docling = use_docling

        # Try to load Docling if requested or auto-detecting
        if use_docling is not False:
            try:
                import docling  # noqa: F401
                self._docling_available = True
                logger.info("Docling is available")
            except ImportError:
                logger.debug("Docling not installed")

        # Auto-detect: use Docling only if enough RAM (>2GB)
        if use_docling is None and self._docling_available:
            try:
                import psutil
                available_ram_gb = psutil.virtual_memory().available / (1024 ** 3)
                self._use_docling = available_ram_gb > 2.0
                if self._use_docling:
                    logger.info(f"Auto-enabled Docling ({available_ram_gb:.1f}GB RAM available)")
                else:
                    logger.info(f"Auto-disabled Docling ({available_ram_gb:.1f}GB RAM, need >2GB)")
            except ImportError:
                self._use_docling = False
                logger.debug("psutil not available, defaulting to PyMuPDF4LLM")

        try:
            import pymupdf4llm  # noqa: F401
            self._pymupdf_available = True
            logger.info("PyMuPDF4LLM initialized successfully")
        except ImportError:
            logger.warning("PyMuPDF4LLM not available")

        # Check for Surya OCR (fallback for scanned PDFs)
        if self._enable_ocr:
            try:
                from surya.recognition import RecognitionPredictor  # noqa: F401
                self._surya_available = True
                logger.info("Surya OCR available (fallback for scanned PDFs)")
            except ImportError:
                logger.debug("Surya OCR not installed — scanned PDFs will be skipped")
        else:
            logger.info("OCR disabled — scanned PDFs will be skipped")

        if not self._pymupdf_available and not self._docling_available:
            raise ImportError(
                "At least one PDF engine is required. "
                "Run: pip install pymupdf4llm  OR  pip install docling"
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

        # Route to appropriate extraction engine
        if self._use_docling and self._docling_available:
            try:
                raw_markdown, page_count, page_ranges = self._extract_with_docling(file_path)
            except Exception as e:
                logger.warning(f"Docling extraction failed, falling back to PyMuPDF: {e}")
                raw_markdown, page_count, page_ranges = self._extract_with_pymupdf(file_path)
        else:
            raw_markdown, page_count, page_ranges = self._extract_with_pymupdf(file_path)

        # Clean up the text
        raw_text = self._markdown_to_text(raw_markdown)

        # Fallback to Surya OCR if text extraction yielded too little content
        # (indicates a scanned/image-based PDF)
        if len(raw_text.strip()) < 50 and self._surya_available:
            logger.info(f"  PyMuPDF extracted too little text ({len(raw_text.strip())} chars), trying Surya OCR...")
            try:
                raw_markdown, page_count, page_ranges = self._extract_with_surya(file_path)
                raw_text = self._markdown_to_text(raw_markdown)
                logger.info(f"  Surya OCR extracted {len(raw_text)} chars from {page_count} pages")
            except Exception as e:
                logger.warning(f"  Surya OCR failed: {e}")

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

    def _extract_with_docling(self, file_path: str) -> tuple[str, int, list[tuple[int, int, int]]]:
        """
        Extract content using Docling for high-quality OCR and table extraction.

        Docling provides superior handling of complex PDFs (scanned docs, tables,
        multi-column layouts) but requires >2GB RAM.

        Returns:
            tuple: (markdown, page_count, page_ranges)
        """
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(file_path)

        # Export to markdown
        markdown = result.document.export_to_markdown()

        # Get page count from Docling result
        page_count = getattr(result.document, 'num_pages', 0)
        if page_count == 0:
            # Fallback: count from PyMuPDF if Docling doesn't report it
            try:
                import fitz
                with fitz.open(file_path) as doc:
                    page_count = len(doc)
            except Exception:
                page_count = 1

        # Build approximate page ranges
        page_ranges = []
        if page_count > 0:
            chars_per_page = max(1, len(markdown) // page_count)
            for i in range(page_count):
                start = i * chars_per_page
                end = (i + 1) * chars_per_page
                page_ranges.append((i + 1, start, end))

        logger.info(f"Docling extracted {page_count} pages, {len(markdown)} chars")
        return markdown, page_count, page_ranges

    def _extract_with_surya(self, file_path: str) -> tuple[str, int, list[tuple[int, int, int]]]:
        """
        Extract text from scanned PDFs using Surya OCR.

        Surya provides high-accuracy (97.7%) OCR with layout analysis,
        supporting 90+ languages including Greek. Used as a fallback when
        PyMuPDF extracts too little text (indicating image-based PDFs).

        Returns:
            tuple: (markdown, page_count, page_ranges)
        """
        from surya.foundation import FoundationPredictor
        from surya.recognition import RecognitionPredictor
        from surya.detection import DetectionPredictor
        from surya.common.surya.schema import TaskNames
        from surya.input.load import load_pdf

        # Lazy-init predictors (cached on first call for reuse across documents)
        if not hasattr(self, '_surya_rec_predictor'):
            logger.info("  Loading Surya OCR models (first time, may take a moment)...")
            self._surya_foundation = FoundationPredictor()
            self._surya_det_predictor = DetectionPredictor()
            self._surya_rec_predictor = RecognitionPredictor(self._surya_foundation)

        # Load PDF pages as images
        images, names = load_pdf(file_path)
        page_count = len(images)

        if page_count == 0:
            return "", 0, []

        # Run OCR
        task_names = [TaskNames.ocr_with_boxes] * len(images)
        predictions = self._surya_rec_predictor(
            images=images,
            task_names=task_names,
            det_predictor=self._surya_det_predictor,
        )

        # Build markdown from OCR results with page tracking
        page_ranges = []
        markdown_parts = []
        cumulative_char = 0

        for page_idx, prediction in enumerate(predictions):
            page_start = cumulative_char

            # Add page header
            page_header = f"\n## Page {page_idx + 1}\n\n"
            markdown_parts.append(page_header)
            cumulative_char += len(page_header)

            # Add each text line
            for line in prediction.text_lines:
                line_text = line.text.strip()
                if line_text:
                    markdown_parts.append(line_text + "\n")
                    cumulative_char += len(line_text) + 1

            page_end = cumulative_char
            page_ranges.append((page_idx + 1, page_start, page_end))

        markdown = "".join(markdown_parts)
        return markdown, page_count, page_ranges

    def _markdown_to_text(self, markdown: str) -> str:
        """Convert markdown to plain text."""
        # Remove markdown formatting (images before links to avoid partial match)
        text = re.sub(r'#+\s*', '', markdown)  # Headers
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*([^*]+)\*', r'\1', text)  # Italic
        text = re.sub(r'!\[([^\]]*)\]\([^)]+\)', '', text)  # Images
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)  # Links
        text = re.sub(r'`([^`]+)`', r'\1', text)  # Code
        
        # Clean OCR artifacts
        text = re.sub(r'GLYPH&lt;\d+&gt;', '', text)  # HTML encoded GLYPH
        text = re.sub(r'GLYPH<\d+>', '', text)  # Raw GLYPH
        text = re.sub(r'<!--.*?-->', '', text)  # HTML comments/placeholders
        text = re.sub(r'\[image\]', '', text, flags=re.IGNORECASE)
        
        # Unescape HTML entities (e.g. &amp; -> &)
        text = html.unescape(text)
        
        return text.strip()

    def _detect_document_type(self, text: str) -> str:
        """Detect the type of legal document using language-aware patterns."""
        scores = {}

        for doc_type, patterns in self._doctype_patterns.items():
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
            
            letter_regex = TITLE_LETTER_REGEX.get(self._lang, TITLE_LETTER_REGEX["en"])
            letter_count = len(re.findall(letter_regex, clean_line))
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
        """Extract jurisdiction from document text using language-aware patterns."""
        patterns = JURISDICTION_PATTERNS.get(self._lang, JURISDICTION_PATTERNS["en"])

        for pattern in patterns:
            match = re.search(pattern, text[:3000])
            if match:
                return match.group(1).strip()

        return None

    def _extract_parties(self, text: str, doc_type: str) -> list[str]:
        """Extract party names from contracts using language-aware patterns."""
        if doc_type != "contract":
            return []

        parties = []
        patterns = PARTY_PATTERNS.get(self._lang, PARTY_PATTERNS["en"])

        for pattern in patterns:
            matches = re.findall(pattern, text[:2000])
            parties.extend(m.strip() for m in matches if len(m.strip()) < 100)

        return list(set(parties))[:5]  # Dedupe and limit

    def _extract_date(self, text: str) -> Optional[datetime]:
        """Extract effective date from document using language-aware patterns."""
        lang_date = DATE_PATTERNS.get(self._lang, DATE_PATTERNS["en"])

        if self._lang == "el":
            # Greek date parsing: "15 Ιανουαρίου 2024"
            for pattern in lang_date["patterns"]:
                match = re.search(pattern, text[:3000])
                if match:
                    date_str = match.group(1)
                    month_map = lang_date["month_map"]
                    for greek_month, month_num in month_map.items():
                        if greek_month in date_str.lower():
                            # Replace Greek month with numeric
                            parts = date_str.lower().replace(greek_month, "").split()
                            try:
                                day = int(parts[0])
                                year = int(parts[-1])
                                return datetime(year, int(month_num), day)
                            except (ValueError, IndexError):
                                continue
            return None

        # English date parsing
        for pattern in lang_date["patterns"]:
            match = re.search(pattern, text[:3000])
            if match:
                date_str = match.group(1)
                for fmt in lang_date["formats"]:
                    try:
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
