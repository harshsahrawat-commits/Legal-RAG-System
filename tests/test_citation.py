"""
Tests for execution/legal_rag/citation.py

Covers: Citation dataclass (short_format, long_format, Bluebook, OSCOLA),
        CitedContent, CitationExtractor,
        paragraph citations, and page formatting.
"""

import pytest


# ---------------------------------------------------------------------------
# Citation dataclass
# ---------------------------------------------------------------------------

class TestCitation:
    """Tests for Citation formatting methods."""

    @pytest.fixture
    def citation(self):
        from execution.legal_rag.citation import Citation
        return Citation(
            document_title="Software License Agreement",
            section="Section 4.2",
            page_numbers=[4, 5],
            hierarchy_path="Doc/Article_IV/Section_4.2",
            chunk_id="c1",
            document_id="d1",
            relevance_score=0.92,
            paragraph_numbers=[15, 16, 17],
        )

    def test_short_format_includes_title(self, citation):
        result = citation.short_format()
        assert "Software License Agreement" in result
        assert result.startswith("[")
        assert result.endswith("]")

    def test_short_format_includes_section(self, citation):
        assert "Section 4.2" in citation.short_format()

    def test_short_format_includes_pages(self, citation):
        result = citation.short_format()
        assert "pp. 4-5" in result

    def test_short_format_includes_paragraphs(self, citation):
        result = citation.short_format()
        # Consecutive paragraphs should use range format
        assert "15-17" in result

    def test_long_format_includes_path(self, citation):
        result = citation.long_format()
        assert "Doc/Article_IV/Section_4.2" in result

    def test_to_dict_keys(self, citation):
        d = citation.to_dict()
        expected_keys = {
            "document_title", "section", "page_numbers",
            "paragraph_numbers", "hierarchy_path", "chunk_id",
            "document_id", "relevance_score",
            "short_citation", "long_citation",
        }
        assert set(d.keys()) == expected_keys


class TestCitationPageFormatting:
    """Tests for _format_pages edge cases."""

    def test_no_pages(self):
        from execution.legal_rag.citation import Citation
        c = Citation(
            document_title="T", section="S", page_numbers=[],
            hierarchy_path="P", chunk_id="c", document_id="d",
            relevance_score=0.5,
        )
        assert c._format_pages() == "Page N/A"

    def test_single_page(self):
        from execution.legal_rag.citation import Citation
        c = Citation(
            document_title="T", section="S", page_numbers=[7],
            hierarchy_path="P", chunk_id="c", document_id="d",
            relevance_score=0.5,
        )
        assert c._format_pages() == "p. 7"

    def test_consecutive_pages(self):
        from execution.legal_rag.citation import Citation
        c = Citation(
            document_title="T", section="S", page_numbers=[3, 4, 5],
            hierarchy_path="P", chunk_id="c", document_id="d",
            relevance_score=0.5,
        )
        assert c._format_pages() == "pp. 3-5"

    def test_non_consecutive_pages(self):
        from execution.legal_rag.citation import Citation
        c = Citation(
            document_title="T", section="S", page_numbers=[1, 3, 7],
            hierarchy_path="P", chunk_id="c", document_id="d",
            relevance_score=0.5,
        )
        assert c._format_pages() == "pp. 1, 3, 7"


class TestCitationParagraphFormatting:
    """Tests for _format_paragraphs edge cases."""

    def test_no_paragraphs(self):
        from execution.legal_rag.citation import Citation
        c = Citation(
            document_title="T", section="S", page_numbers=[],
            hierarchy_path="P", chunk_id="c", document_id="d",
            relevance_score=0.5, paragraph_numbers=None,
        )
        assert c._format_paragraphs() == ""

    def test_single_paragraph(self):
        from execution.legal_rag.citation import Citation
        c = Citation(
            document_title="T", section="S", page_numbers=[],
            hierarchy_path="P", chunk_id="c", document_id="d",
            relevance_score=0.5, paragraph_numbers=[42],
        )
        # Single paragraph uses single pilcrow
        assert c._format_paragraphs() == "\u00b642"

    def test_consecutive_paragraphs_range(self):
        from execution.legal_rag.citation import Citation
        c = Citation(
            document_title="T", section="S", page_numbers=[],
            hierarchy_path="P", chunk_id="c", document_id="d",
            relevance_score=0.5, paragraph_numbers=[10, 11, 12],
        )
        result = c._format_paragraphs()
        assert "10-12" in result

    def test_non_consecutive_paragraphs(self):
        from execution.legal_rag.citation import Citation
        c = Citation(
            document_title="T", section="S", page_numbers=[],
            hierarchy_path="P", chunk_id="c", document_id="d",
            relevance_score=0.5, paragraph_numbers=[5, 8, 12],
        )
        result = c._format_paragraphs()
        assert "5" in result
        assert "8" in result
        assert "12" in result


class TestIsConsecutive:
    """Tests for _is_consecutive helper."""

    def test_consecutive(self):
        from execution.legal_rag.citation import Citation
        c = Citation(
            document_title="T", section="S", page_numbers=[],
            hierarchy_path="P", chunk_id="c", document_id="d",
            relevance_score=0.5,
        )
        assert c._is_consecutive([1, 2, 3, 4]) is True

    def test_not_consecutive(self):
        from execution.legal_rag.citation import Citation
        c = Citation(
            document_title="T", section="S", page_numbers=[],
            hierarchy_path="P", chunk_id="c", document_id="d",
            relevance_score=0.5,
        )
        assert c._is_consecutive([1, 3, 5]) is False

    def test_single_element(self):
        from execution.legal_rag.citation import Citation
        c = Citation(
            document_title="T", section="S", page_numbers=[],
            hierarchy_path="P", chunk_id="c", document_id="d",
            relevance_score=0.5,
        )
        assert c._is_consecutive([1]) is False

    def test_empty_list(self):
        from execution.legal_rag.citation import Citation
        c = Citation(
            document_title="T", section="S", page_numbers=[],
            hierarchy_path="P", chunk_id="c", document_id="d",
            relevance_score=0.5,
        )
        assert c._is_consecutive([]) is False


# ---------------------------------------------------------------------------
# CitedContent
# ---------------------------------------------------------------------------

class TestCitedContent:
    """Tests for CitedContent."""

    def test_format_with_citation(self):
        from execution.legal_rag.citation import Citation, CitedContent
        cit = Citation(
            document_title="Contract", section="Section 1",
            page_numbers=[1], hierarchy_path="P",
            chunk_id="c", document_id="d", relevance_score=0.8,
        )
        cc = CitedContent(
            content="Some legal text.",
            citation=cit,
            context_before="Before.",
            context_after="After.",
        )
        result = cc.format_with_citation()
        assert "Some legal text." in result
        assert "[Contract" in result

    def test_to_dict(self):
        from execution.legal_rag.citation import Citation, CitedContent
        cit = Citation(
            document_title="C", section="S", page_numbers=[],
            hierarchy_path="P", chunk_id="c", document_id="d",
            relevance_score=0.5,
        )
        cc = CitedContent(
            content="text", citation=cit,
            context_before="b", context_after="a",
        )
        d = cc.to_dict()
        assert "content" in d
        assert "citation" in d
        assert "context_before" in d
        assert "context_after" in d


# ---------------------------------------------------------------------------
# CitationExtractor
# ---------------------------------------------------------------------------

class TestCitationExtractor:
    """Tests for CitationExtractor.extract() and helpers."""

    def test_extract_returns_cited_contents(self, sample_search_results):
        from execution.legal_rag.citation import CitationExtractor
        extractor = CitationExtractor(
            document_titles={"doc-001": "Software License Agreement"}
        )
        cited = extractor.extract(sample_search_results)
        assert len(cited) == len(sample_search_results)
        assert all(hasattr(cc, "citation") for cc in cited)

    def test_document_title_from_map(self, sample_search_results):
        from execution.legal_rag.citation import CitationExtractor
        extractor = CitationExtractor(
            document_titles={"doc-001": "My Contract"}
        )
        cited = extractor.extract(sample_search_results)
        assert cited[0].citation.document_title == "My Contract"

    def test_document_title_override(self, sample_search_results):
        from execution.legal_rag.citation import CitationExtractor
        extractor = CitationExtractor()
        cited = extractor.extract(
            sample_search_results, document_title="Override Title"
        )
        assert all(
            cc.citation.document_title == "Override Title" for cc in cited
        )

    def test_extract_section_from_hierarchy_path(self):
        from execution.legal_rag.citation import CitationExtractor
        extractor = CitationExtractor()
        assert "Article" in extractor._extract_section("Doc/Article_IV")
        assert "Section" in extractor._extract_section("Doc/Section_3.2")
        assert "Clause" in extractor._extract_section("Doc/Clause_5")
        assert "Part" in extractor._extract_section("Doc/Part_II")
        assert "Chapter" in extractor._extract_section("Doc/Chapter_3")

    def test_extract_section_falls_back_to_last_part(self):
        from execution.legal_rag.citation import CitationExtractor
        extractor = CitationExtractor()
        result = extractor._extract_section("Doc/Some_Thing/Other_Part")
        assert "Other Part" in result

    def test_extract_section_empty_path(self):
        from execution.legal_rag.citation import CitationExtractor
        extractor = CitationExtractor()
        assert extractor._extract_section("") == ""

    def test_extract_title_from_path(self):
        from execution.legal_rag.citation import CitationExtractor
        extractor = CitationExtractor()
        assert extractor._extract_title_from_path("My_Doc/Section_1") == "My Doc"

    def test_extract_title_from_empty_path(self):
        from execution.legal_rag.citation import CitationExtractor
        extractor = CitationExtractor()
        assert extractor._extract_title_from_path("") is None

    def test_paragraph_numbers_passed_to_citation(self, sample_search_results):
        from execution.legal_rag.citation import CitationExtractor
        extractor = CitationExtractor()
        cited = extractor.extract(sample_search_results)
        # First result has paragraphs [15,16,17]
        assert cited[0].citation.paragraph_numbers == [15, 16, 17]


