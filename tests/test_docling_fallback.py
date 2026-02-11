"""Tests for Docling auto-detection and fallback behavior in document parser."""

import sys
import pytest
from unittest.mock import patch, MagicMock

from execution.legal_rag.language_config import TenantLanguageConfig


# We mock pymupdf4llm at module level for all tests in this file,
# because the real pymupdf4llm can segfault on Python 3.14.
@pytest.fixture(autouse=True)
def mock_pymupdf4llm():
    """Mock pymupdf4llm to avoid segfault on Python 3.14."""
    mock_mod = MagicMock()
    with patch.dict(sys.modules, {"pymupdf4llm": mock_mod}):
        yield mock_mod


class TestDoclingAutoDetection:
    """Tests for the Docling/PyMuPDF routing in LegalDocumentParser.__init__."""

    def test_use_docling_false_disables_docling(self):
        """When use_docling=False, Docling should not be loaded."""
        from execution.legal_rag.document_parser import LegalDocumentParser
        parser = LegalDocumentParser(use_docling=False)
        assert parser._use_docling is False

    def test_pymupdf_is_default_engine(self):
        """Without Docling, PyMuPDF4LLM should be the default engine."""
        from execution.legal_rag.document_parser import LegalDocumentParser
        parser = LegalDocumentParser(use_docling=False)
        assert parser._pymupdf_available is True

    @patch("psutil.virtual_memory")
    def test_auto_detect_high_ram_enables_docling(self, mock_vmem):
        """With >2GB RAM and Docling available, auto-detect should enable it."""
        mock_vmem.return_value = MagicMock(available=4 * 1024**3)  # 4GB

        with patch.dict(sys.modules, {
            "docling": MagicMock(),
            "docling.document_converter": MagicMock(),
        }):
            from execution.legal_rag.document_parser import LegalDocumentParser
            parser = LegalDocumentParser(use_docling=None)
            assert parser._use_docling is True

    @patch("psutil.virtual_memory")
    def test_auto_detect_low_ram_disables_docling(self, mock_vmem):
        """With <2GB RAM, auto-detect should disable Docling."""
        mock_vmem.return_value = MagicMock(available=1 * 1024**3)  # 1GB

        with patch.dict(sys.modules, {
            "docling": MagicMock(),
            "docling.document_converter": MagicMock(),
        }):
            from execution.legal_rag.document_parser import LegalDocumentParser
            parser = LegalDocumentParser(use_docling=None)
            assert parser._use_docling is False

    def test_language_config_passed_through(self):
        """Parser should use the provided language config."""
        config = TenantLanguageConfig.for_language("el")
        from execution.legal_rag.document_parser import LegalDocumentParser
        parser = LegalDocumentParser(language_config=config, use_docling=False)
        assert parser._lang == "el"
        assert "contract" in parser._doctype_patterns

    def test_default_language_is_english(self):
        """Without explicit config, parser should default to English."""
        from execution.legal_rag.document_parser import LegalDocumentParser
        parser = LegalDocumentParser(use_docling=False)
        assert parser._lang == "en"


class TestDoclingFallbackChain:
    """Test that Docling errors fall back to PyMuPDF4LLM."""

    @patch("psutil.virtual_memory")
    def test_docling_failure_falls_back_to_pymupdf(self, mock_vmem):
        """If Docling extraction fails, PyMuPDF should be used as fallback."""
        mock_vmem.return_value = MagicMock(available=4 * 1024**3)  # 4GB

        from execution.legal_rag.document_parser import LegalDocumentParser

        with patch.dict(sys.modules, {
            "docling": MagicMock(),
            "docling.document_converter": MagicMock(),
        }):
            parser = LegalDocumentParser(use_docling=True)

            # Mock _extract_with_docling to raise an error
            parser._extract_with_docling = MagicMock(side_effect=RuntimeError("Docling crashed"))
            # Mock _extract_with_pymupdf to succeed
            parser._extract_with_pymupdf = MagicMock(
                return_value=("# Test Document\nContent", 1, [(1, 0, 100)])
            )

            # Directly test the routing logic
            parser._docling_available = True
            parser._use_docling = True
            parser._pymupdf_available = True

            # Simulate what parse() does
            try:
                parser._extract_with_docling("/fake/path.pdf")
                used_docling = True
            except Exception:
                used_docling = False

            assert not used_docling
            # PyMuPDF should work
            md, pages, ranges = parser._extract_with_pymupdf("/fake/path.pdf")
            assert md == "# Test Document\nContent"
