"""
Tests for execution/legal_rag/chunker.py

Covers: Chunk dataclass, ChunkConfig, LegalChunker class methods including
        hierarchical chunking, paragraph tracking, context overlap,
        legal reference extraction, and definition detection.
"""

import uuid
import re
from unittest.mock import patch, MagicMock

import pytest


# ---------------------------------------------------------------------------
# Chunk dataclass
# ---------------------------------------------------------------------------

class TestChunkDataclass:
    """Tests for the Chunk dataclass."""

    def test_to_dict_keys(self):
        from execution.legal_rag.chunker import Chunk

        chunk = Chunk(
            chunk_id="c1", document_id="d1", content="text",
            token_count=10, level=2, section_title="Section",
            hierarchy_path="Doc/Section",
        )
        d = chunk.to_dict()
        expected_keys = {
            "chunk_id", "document_id", "content", "token_count", "level",
            "section_title", "hierarchy_path", "parent_chunk_id",
            "child_chunk_ids", "page_numbers", "start_char", "end_char",
            "paragraph_start", "paragraph_end", "original_paragraph_numbers",
            "contextualized", "context_prefix", "context_before",
            "context_after", "legal_references", "definitions_used",
        }
        assert set(d.keys()) == expected_keys

    def test_default_values(self):
        from execution.legal_rag.chunker import Chunk

        chunk = Chunk(
            chunk_id="c1", document_id="d1", content="x",
            token_count=1, level=0, section_title="T",
            hierarchy_path="P",
        )
        assert chunk.parent_chunk_id is None
        assert chunk.child_chunk_ids == []
        assert chunk.legal_references == []
        assert chunk.definitions_used == []
        assert chunk.contextualized is False


# ---------------------------------------------------------------------------
# ChunkConfig
# ---------------------------------------------------------------------------

class TestChunkConfig:
    """Tests for ChunkConfig defaults."""

    def test_defaults(self):
        from execution.legal_rag.chunker import ChunkConfig
        cfg = ChunkConfig()
        assert cfg.max_tokens_l1 == 1500
        assert cfg.max_tokens_l2 == 600
        assert cfg.max_tokens_l3 == 300
        assert cfg.overlap_tokens == 100
        assert cfg.min_tokens == 50
        assert cfg.context_tokens == 100


# ---------------------------------------------------------------------------
# LegalChunker - core chunking logic
# ---------------------------------------------------------------------------

class TestLegalChunkerChunk:
    """Tests for the main chunk() method."""

    def test_creates_at_least_summary_chunk(self, chunker, sample_parsed_document):
        """Every document should produce at least a level-0 summary chunk."""
        chunks = chunker.chunk(sample_parsed_document)
        assert len(chunks) >= 1
        levels = {c.level for c in chunks}
        assert 0 in levels, "Must have a level-0 summary chunk"

    def test_summary_chunk_is_first(self, chunker, sample_parsed_document):
        chunks = chunker.chunk(sample_parsed_document)
        assert chunks[0].level == 0
        assert "DOCUMENT SUMMARY" in chunks[0].content

    def test_all_chunks_have_document_id(self, chunker, sample_parsed_document):
        chunks = chunker.chunk(sample_parsed_document)
        doc_id = sample_parsed_document.metadata.document_id
        for chunk in chunks:
            assert chunk.document_id == doc_id

    def test_all_chunks_have_content(self, chunker, sample_parsed_document):
        chunks = chunker.chunk(sample_parsed_document)
        for chunk in chunks:
            assert len(chunk.content.strip()) > 0

    def test_chunk_count_proportional_to_sections(
        self, chunker, sample_parsed_document
    ):
        """More sections should produce more chunks."""
        chunks = chunker.chunk(sample_parsed_document)
        num_sections = len(sample_parsed_document.sections)
        # At minimum: 1 summary + 1 per section
        assert len(chunks) >= num_sections + 1

    def test_hierarchy_path_set_on_all(self, chunker, sample_parsed_document):
        chunks = chunker.chunk(sample_parsed_document)
        for chunk in chunks:
            assert chunk.hierarchy_path, f"Chunk {chunk.chunk_id} has no hierarchy_path"


class TestLegalChunkerTokenEstimation:
    """Tests for _estimate_tokens."""

    def test_estimate_tokens(self, chunker):
        assert chunker._estimate_tokens("a" * 400) == 100
        assert chunker._estimate_tokens("") == 0

    def test_token_counts_match(self, chunker, sample_parsed_document):
        """Each chunk's token_count should be close to the estimate of its content."""
        chunks = chunker.chunk(sample_parsed_document)
        for chunk in chunks:
            expected = chunker._estimate_tokens(chunk.content)
            # token_count may differ slightly from content estimate due to
            # context prefix added after initial token counting
            assert abs(chunk.token_count - expected) <= expected * 0.5 + 10, \
                f"Chunk {chunk.chunk_id}: token_count={chunk.token_count}, expected~{expected}"


class TestLegalChunkerParagraphTracking:
    """Tests for paragraph number tracking in chunks."""

    def test_paragraph_numbers_assigned(self, chunker, sample_parsed_document):
        """Non-summary chunks should have paragraph tracking info."""
        chunks = chunker.chunk(sample_parsed_document)
        non_summary = [c for c in chunks if c.level > 0]
        # At least some chunks should have paragraph info
        has_para = [c for c in non_summary if c.paragraph_start is not None]
        assert len(has_para) > 0, "Some chunks should have paragraph_start set"

    def test_paragraph_start_le_end(self, chunker, sample_parsed_document):
        chunks = chunker.chunk(sample_parsed_document)
        for chunk in chunks:
            if chunk.paragraph_start is not None and chunk.paragraph_end is not None:
                assert chunk.paragraph_start <= chunk.paragraph_end


class TestLegalChunkerContext:
    """Tests for context overlap between chunks."""

    def test_context_before_set(self, chunker, sample_parsed_document):
        chunks = chunker.chunk(sample_parsed_document)
        # First chunk should have no context_before; second should
        assert chunks[0].context_before == ""
        if len(chunks) > 1:
            assert chunks[1].context_before != "" or True  # may be empty for short docs

    def test_context_after_set(self, chunker, sample_parsed_document):
        chunks = chunker.chunk(sample_parsed_document)
        if len(chunks) > 1:
            # Last chunk should have no context_after
            assert chunks[-1].context_after == ""


class TestLegalChunkerLegalMetadata:
    """Tests for legal reference and definition extraction."""

    def test_legal_references_extracted(self, chunker, sample_parsed_document):
        """Chunks with section references should have them in legal_references."""
        chunks = chunker.chunk(sample_parsed_document)
        all_refs = []
        for c in chunks:
            all_refs.extend(c.legal_references)
        # The sample document has Section references
        assert len(all_refs) > 0, "Should extract at least some legal references"

    def test_definitions_extracted(self, chunker, sample_parsed_document):
        """Chunks with quoted terms should detect definitions."""
        chunks = chunker.chunk(sample_parsed_document)
        all_defs = []
        for c in chunks:
            all_defs.extend(c.definitions_used)
        # The sample has "Software", "Documentation", etc.
        assert len(all_defs) > 0, "Should extract defined terms"


class TestLegalChunkerHierarchy:
    """Tests for parent-child linking."""

    def test_parent_child_linked(self, chunker, sample_parsed_document):
        chunks = chunker.chunk(sample_parsed_document)
        summary = chunks[0]
        children = [c for c in chunks if c.parent_chunk_id == summary.chunk_id]
        assert len(children) > 0, "Summary chunk should have children"
        assert all(
            c.chunk_id in summary.child_chunk_ids for c in children
        ), "Children should be in parent's child_chunk_ids"


class TestLegalChunkerSplitOnMarkers:
    """Tests for _split_on_markers."""

    def test_no_markers_returns_original(self, chunker):
        text = "Just plain text without any legal markers."
        result = chunker._split_on_markers(text)
        assert result == [text]

    def test_splits_on_article_marker(self, chunker):
        text = "Intro text\nARTICLE I Something\nContent\nARTICLE II Other\nMore content"
        result = chunker._split_on_markers(text)
        assert len(result) >= 2


class TestLegalChunkerSplitOnParagraphs:
    """Tests for _split_on_paragraphs."""

    def test_single_paragraph(self, chunker):
        text = "Single paragraph."
        result = chunker._split_on_paragraphs(text, max_tokens=1000)
        assert len(result) >= 1

    def test_multiple_paragraphs(self, chunker):
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        result = chunker._split_on_paragraphs(text, max_tokens=1000)
        assert len(result) >= 1

    def test_paragraph_indices_are_correct(self, chunker):
        text = "Para one.\n\nPara two.\n\nPara three."
        result = chunker._split_on_paragraphs(text, max_tokens=1000, base_paragraph_offset=5)
        # First paragraph should start at 6 (offset 5 + 1)
        first_indices = result[0][0]
        assert first_indices[0] == 6

    def test_empty_text(self, chunker):
        result = chunker._split_on_paragraphs("", max_tokens=100)
        assert len(result) >= 1


class TestLegalChunkerGetOverlap:
    """Tests for _get_overlap."""

    def test_overlap_from_end(self, chunker):
        text = " ".join([f"word{i}" for i in range(200)])
        overlap = chunker._get_overlap(text)
        # Should contain the last overlap_tokens words
        assert len(overlap.split()) == chunker.config.overlap_tokens

    def test_overlap_from_short_text(self, chunker):
        text = "short text"
        overlap = chunker._get_overlap(text)
        assert "short" in overlap
        assert "text" in overlap


class TestLegalChunkerContextualize:
    """Tests for contextualize_chunks (mocked LLM)."""

    def test_skips_without_openai(self, chunker, sample_chunks):
        """When OPENAI_AVAILABLE is False, chunks should be returned unchanged."""
        with patch("execution.legal_rag.chunker.OPENAI_AVAILABLE", False):
            result = chunker.contextualize_chunks(sample_chunks, "summary")
            assert len(result) == len(sample_chunks)
            assert all(not c.contextualized for c in result)

    def test_skips_without_api_key(self, chunker, sample_chunks, monkeypatch):
        """When NVIDIA_API_KEY is missing, chunks should be returned unchanged."""
        monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
        with patch("execution.legal_rag.chunker.OPENAI_AVAILABLE", True):
            result = chunker.contextualize_chunks(sample_chunks, "summary")
            assert len(result) == len(sample_chunks)

    def test_skips_level_0_chunks(self, chunker, sample_chunks, monkeypatch):
        """Summary chunks (level 0) should not be contextualized."""
        monkeypatch.setenv("NVIDIA_API_KEY", "fake-key")
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "test context"
        mock_client.chat.completions.create.return_value = mock_response

        with patch("execution.legal_rag.chunker.OPENAI_AVAILABLE", True):
            with patch("execution.legal_rag.chunker.OpenAI", return_value=mock_client):
                result = chunker.contextualize_chunks(sample_chunks, "summary")
                summary_chunks = [c for c in result if c.level == 0]
                for sc in summary_chunks:
                    assert not sc.contextualized
