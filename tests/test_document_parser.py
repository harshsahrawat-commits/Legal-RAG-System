"""
Tests for execution/legal_rag/document_parser.py

Covers: LegalMetadata, DocumentSection, ParsedDocument data classes,
        LegalDocumentParser methods for text processing, type detection,
        title extraction, section extraction, jurisdiction/parties/date extraction,
        and page-number calculation.

All tests mock the PDF library imports so no real PDF files are needed.
"""

import re
import uuid
from datetime import datetime
from unittest.mock import patch, MagicMock

import pytest

from execution.legal_rag.language_config import TenantLanguageConfig
from execution.legal_rag.language_patterns import (
    DOCTYPE_PATTERNS, JURISDICTION_PATTERNS, PARTY_PATTERNS,
    DATE_PATTERNS, TITLE_LETTER_REGEX,
)


def _set_parser_lang_defaults(parser):
    """Set English language defaults on a parser created via object.__new__."""
    from execution.legal_rag.language_patterns import SECTION_PATTERNS
    parser._language_config = TenantLanguageConfig.for_language("en")
    parser._lang = "en"
    parser._doctype_patterns = DOCTYPE_PATTERNS["en"]
    parser._section_patterns = SECTION_PATTERNS["en"]


# ---------------------------------------------------------------------------
# Data-class unit tests
# ---------------------------------------------------------------------------

class TestLegalMetadata:
    """Tests for the LegalMetadata dataclass."""

    def test_to_dict_with_all_fields(self):
        """Verify to_dict returns every field, including formatted dates."""
        from execution.legal_rag.document_parser import LegalMetadata

        now = datetime(2024, 6, 15, 10, 30, 0)
        meta = LegalMetadata(
            document_id="abc-123",
            title="Test Contract",
            document_type="contract",
            jurisdiction="California",
            effective_date=datetime(2024, 1, 1),
            parties=["Alpha Corp", "Beta LLC"],
            page_count=42,
            file_path="/docs/test.pdf",
            created_at=now,
        )
        d = meta.to_dict()

        assert d["document_id"] == "abc-123"
        assert d["title"] == "Test Contract"
        assert d["document_type"] == "contract"
        assert d["jurisdiction"] == "California"
        assert d["effective_date"] == "2024-01-01T00:00:00"
        assert d["parties"] == ["Alpha Corp", "Beta LLC"]
        assert d["page_count"] == 42
        assert d["created_at"] == "2024-06-15T10:30:00"

    def test_to_dict_with_none_effective_date(self):
        """Verify effective_date serialises to None when not set."""
        from execution.legal_rag.document_parser import LegalMetadata

        meta = LegalMetadata(
            document_id="id-1", title="T", document_type="unknown",
        )
        assert meta.to_dict()["effective_date"] is None

    def test_default_parties_is_empty_list(self):
        """Verify parties defaults to an empty list, not a shared mutable."""
        from execution.legal_rag.document_parser import LegalMetadata

        a = LegalMetadata(document_id="1", title="A", document_type="x")
        b = LegalMetadata(document_id="2", title="B", document_type="y")
        a.parties.append("Acme")
        assert b.parties == []


class TestDocumentSection:
    """Tests for the DocumentSection dataclass."""

    def test_to_dict_includes_all_keys(self):
        from execution.legal_rag.document_parser import DocumentSection

        sec = DocumentSection(
            section_id="s1", title="Definitions",
            content="Some text", level=2,
            parent_id="p1", page_numbers=[1, 2],
            hierarchy_path="Doc/Definitions",
        )
        d = sec.to_dict()
        expected_keys = {
            "section_id", "title", "content", "level",
            "parent_id", "page_numbers", "hierarchy_path",
        }
        assert set(d.keys()) == expected_keys
        assert d["page_numbers"] == [1, 2]


class TestParsedDocument:
    """Tests for ParsedDocument.to_dict()."""

    def test_to_dict_structure(self, sample_parsed_document):
        d = sample_parsed_document.to_dict()
        assert "metadata" in d
        assert "sections" in d
        assert "raw_text" in d
        assert "raw_markdown" in d
        assert isinstance(d["sections"], list)


# ---------------------------------------------------------------------------
# LegalDocumentParser -- pure-function unit tests (no PDF I/O)
# ---------------------------------------------------------------------------

class TestMarkdownToText:
    """Tests for _markdown_to_text."""

    @pytest.fixture
    def parser(self):
        with patch("execution.legal_rag.document_parser.LegalDocumentParser.__init__",
                    lambda self, **kw: None):
            p = object.__new__(
                __import__("execution.legal_rag.document_parser",
                           fromlist=["LegalDocumentParser"]).LegalDocumentParser
            )
            return p

    def test_strips_headers(self, parser):
        assert "TITLE" in parser._markdown_to_text("## TITLE")
        assert "#" not in parser._markdown_to_text("## TITLE")

    def test_strips_bold(self, parser):
        assert parser._markdown_to_text("**bold**") == "bold"

    def test_strips_italic(self, parser):
        assert parser._markdown_to_text("*italic*") == "italic"

    def test_strips_links(self, parser):
        result = parser._markdown_to_text("[click](http://example.com)")
        assert result == "click"

    def test_strips_code(self, parser):
        assert parser._markdown_to_text("`code`") == "code"

    def test_strips_images(self, parser):
        result = parser._markdown_to_text("![alt](image.png)")
        assert result == ""

    def test_strips_glyph_artifacts(self, parser):
        assert parser._markdown_to_text("GLYPH<123>") == ""
        assert parser._markdown_to_text("GLYPH&lt;456&gt;") == ""

    def test_strips_html_comments(self, parser):
        assert parser._markdown_to_text("<!-- comment -->") == ""

    def test_unescapes_html_entities(self, parser):
        assert "&" in parser._markdown_to_text("Smith &amp; Jones")

    def test_strips_image_placeholder(self, parser):
        result = parser._markdown_to_text("[image]")
        assert result.strip() == ""


class TestDetectDocumentType:
    """Tests for _detect_document_type."""

    @pytest.fixture
    def parser(self):
        from execution.legal_rag.document_parser import LegalDocumentParser
        with patch.object(LegalDocumentParser, "__init__", lambda s, **kw: None):
            p = object.__new__(LegalDocumentParser)
            _set_parser_lang_defaults(p)
            return p

    @pytest.mark.parametrize("text,expected", [
        ("This Agreement is entered into between parties agree", "contract"),
        ("WITNESSETH whereas the parties", "contract"),
        ("Be it enacted by the legislature section 1.", "statute"),
        ("The plaintiff filed against the defendant in court", "case_law"),
        ("Pursuant to the Code of Federal Regulations", "regulation"),
    ])
    def test_detection(self, parser, text, expected):
        assert parser._detect_document_type(text) == expected

    def test_unknown_when_no_patterns_match(self, parser):
        assert parser._detect_document_type("Lorem ipsum dolor sit amet") == "unknown"


class TestExtractTitle:
    """Tests for _extract_title."""

    @pytest.fixture
    def parser(self):
        from execution.legal_rag.document_parser import LegalDocumentParser
        with patch.object(LegalDocumentParser, "__init__", lambda s, **kw: None):
            p = object.__new__(LegalDocumentParser)
            _set_parser_lang_defaults(p)
            return p

    def test_extracts_from_markdown_heading(self, parser):
        md = "# My Contract Title\n\nSome content."
        assert parser._extract_title(md, "fallback") == "My Contract Title"

    def test_uses_first_suitable_line_when_no_heading(self, parser):
        md = "SOFTWARE LICENSE AGREEMENT\n\nSome content below."
        result = parser._extract_title(md, "fallback")
        assert result == "SOFTWARE LICENSE AGREEMENT"

    def test_skips_short_lines(self, parser):
        md = "ab\nSoftware License Agreement\nContent."
        assert "Software License Agreement" in parser._extract_title(md, "fb")

    def test_falls_back_when_nothing_matches(self, parser):
        md = "\n\n\n"
        assert parser._extract_title(md, "my_fallback") == "my_fallback"

    def test_skips_lines_ending_with_period(self, parser):
        md = "This is a sentence.\nSoftware License Agreement\n"
        result = parser._extract_title(md, "fb")
        assert result == "Software License Agreement"


class TestExtractJurisdiction:
    """Tests for _extract_jurisdiction."""

    @pytest.fixture
    def parser(self):
        from execution.legal_rag.document_parser import LegalDocumentParser
        with patch.object(LegalDocumentParser, "__init__", lambda s, **kw: None):
            p = object.__new__(LegalDocumentParser)
            _set_parser_lang_defaults(p)
            return p

    def test_state_of_pattern(self, parser):
        text = "governed by the laws of the State of Delaware"
        # The regex requires capital letter start
        # Try the 'laws of' pattern
        assert parser._extract_jurisdiction(text) is not None

    def test_no_match_returns_none(self, parser):
        assert parser._extract_jurisdiction("No jurisdiction info here") is None


class TestExtractParties:
    """Tests for _extract_parties."""

    @pytest.fixture
    def parser(self):
        from execution.legal_rag.document_parser import LegalDocumentParser
        with patch.object(LegalDocumentParser, "__init__", lambda s, **kw: None):
            p = object.__new__(LegalDocumentParser)
            _set_parser_lang_defaults(p)
            return p

    def test_returns_empty_for_non_contract(self, parser):
        assert parser._extract_parties("Some text", "statute") == []

    def test_extracts_hereinafter_pattern(self, parser):
        text = 'hereinafter "Licensor" and hereinafter "Licensee"'
        parties = parser._extract_parties(text, "contract")
        names = [p.lower() for p in parties]
        # Should find at least one party name
        assert len(parties) >= 1


class TestExtractDate:
    """Tests for _extract_date."""

    @pytest.fixture
    def parser(self):
        from execution.legal_rag.document_parser import LegalDocumentParser
        with patch.object(LegalDocumentParser, "__init__", lambda s, **kw: None):
            p = object.__new__(LegalDocumentParser)
            _set_parser_lang_defaults(p)
            return p

    def test_effective_date_parsing(self, parser):
        text = "effective as of January 1, 2024"
        result = parser._extract_date(text)
        assert result is not None
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 1

    def test_dated_as_of_pattern(self, parser):
        text = "dated as of March 15, 2023"
        result = parser._extract_date(text)
        assert result is not None
        assert result.month == 3

    def test_returns_none_when_no_date(self, parser):
        assert parser._extract_date("No date here.") is None


class TestCalculatePageNumbers:
    """Tests for _calculate_page_numbers."""

    @pytest.fixture
    def parser(self):
        from execution.legal_rag.document_parser import LegalDocumentParser
        with patch.object(LegalDocumentParser, "__init__", lambda s, **kw: None):
            return object.__new__(LegalDocumentParser)

    def test_empty_page_ranges_returns_empty(self, parser):
        assert parser._calculate_page_numbers(0, 100, []) == []

    def test_single_page(self, parser):
        ranges = [(1, 0, 500)]
        result = parser._calculate_page_numbers(0, 100, ranges)
        assert result == [1]

    def test_multi_page_overlap(self, parser):
        ranges = [(1, 0, 100), (2, 100, 200), (3, 200, 300)]
        result = parser._calculate_page_numbers(50, 250, ranges)
        assert 1 in result
        assert 2 in result
        assert 3 in result

    def test_no_overlap_returns_default_page_1(self, parser):
        ranges = [(2, 1000, 2000)]
        result = parser._calculate_page_numbers(0, 10, ranges)
        # No overlap, returns [1] as default
        assert result == [1]


class TestExtractSections:
    """Tests for _extract_sections."""

    @pytest.fixture
    def parser(self):
        from execution.legal_rag.document_parser import LegalDocumentParser
        with patch.object(LegalDocumentParser, "__init__", lambda s, **kw: None):
            return object.__new__(LegalDocumentParser)

    def test_extracts_from_markdown_headers(self, parser):
        md = "# Title\n\nIntro.\n\n## Section A\n\nContent A.\n\n## Section B\n\nContent B."
        sections = parser._extract_sections(md, "doc-1")
        titles = [s.title for s in sections]
        assert "Title" in titles
        assert "Section A" in titles
        assert "Section B" in titles

    def test_single_section_when_no_headers(self, parser):
        md = "Plain text with no markdown headers at all."
        sections = parser._extract_sections(md, "doc-1")
        assert len(sections) == 1
        assert sections[0].title == "Document Content"
        assert sections[0].level == 0

    def test_hierarchy_path_built(self, parser):
        md = "# Root\n\nText\n\n## Child\n\nMore"
        sections = parser._extract_sections(md, "doc-1")
        child = [s for s in sections if s.title == "Child"][0]
        assert "Root" in child.hierarchy_path
        assert "Child" in child.hierarchy_path

    def test_parent_id_assigned(self, parser):
        md = "# Root\n\nText\n\n## Child\n\nMore"
        sections = parser._extract_sections(md, "doc-1")
        root = [s for s in sections if s.title == "Root"][0]
        child = [s for s in sections if s.title == "Child"][0]
        assert child.parent_id == root.section_id


class TestLegalDocumentParserInit:
    """Test the __init__ of LegalDocumentParser with mocked PDF libraries."""

    def test_raises_when_no_pdf_library(self):
        """Parser should raise ImportError when no PDF library is available."""
        from execution.legal_rag.document_parser import LegalDocumentParser

        # Mock pymupdf4llm import to fail inside __init__
        original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

        def mock_import(name, *args, **kwargs):
            if name == "pymupdf4llm":
                raise ImportError("no module")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(ImportError, match="At least one PDF engine"):
                LegalDocumentParser(use_docling=False)

    def test_parse_raises_on_missing_file(self):
        """parse() should raise FileNotFoundError for non-existent paths."""
        from execution.legal_rag.document_parser import LegalDocumentParser

        # Mock __init__ to skip actual library imports
        with patch.object(LegalDocumentParser, "__init__", lambda self, **kw: None):
            parser = object.__new__(LegalDocumentParser)
            parser._pymupdf_available = True
            parser._docling_available = False
            parser._use_docling = False
            parser.is_cloud = False
            _set_parser_lang_defaults(parser)

            with pytest.raises(FileNotFoundError):
                parser.parse("/nonexistent/file.pdf")
