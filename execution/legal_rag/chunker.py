"""
Legal-Aware Hierarchical Chunker

Splits legal documents into semantically meaningful chunks while
preserving document structure (articles, clauses, sections).

Uses multi-level chunking strategy:
- L0: Document summary (500-1000 tokens)
- L1: Section/Chapter (1000-2000 tokens)
- L2: Article/Clause (300-800 tokens)
- L3: Paragraph (100-300 tokens)
"""

import os
import re
import uuid
import logging
from dataclasses import dataclass, field
from typing import Optional
from .document_parser import ParsedDocument, DocumentSection
from .language_config import TenantLanguageConfig
from .language_patterns import (
    SECTION_MARKERS,
    REFERENCE_PATTERNS,
    DEFINITION_MARKERS,
    LABELS,
    LLM_PROMPTS,
)

# Optional import for contextual chunking
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A chunk of text with metadata for retrieval."""
    chunk_id: str
    document_id: str
    content: str
    token_count: int
    level: int  # Chunk level (0=summary, 1=section, 2=clause, 3=paragraph)

    # Hierarchy and navigation
    section_title: str
    hierarchy_path: str
    parent_chunk_id: Optional[str] = None
    child_chunk_ids: list[str] = field(default_factory=list)

    # Citation info
    page_numbers: list[int] = field(default_factory=list)
    start_char: int = 0
    end_char: int = 0

    # Paragraph tracking (for precise retrieval)
    paragraph_start: Optional[int] = None  # First paragraph number (1-indexed)
    paragraph_end: Optional[int] = None    # Last paragraph number (inclusive)
    original_paragraph_numbers: list[int] = field(default_factory=list)  # All paragraphs in chunk

    # Contextual retrieval (Anthropic method)
    contextualized: bool = False  # Whether context was prepended
    context_prefix: str = ""      # The generated context prefix

    # Context for retrieval
    context_before: str = ""  # Summary of preceding content
    context_after: str = ""   # Summary of following content

    # Legal-specific
    legal_references: list[str] = field(default_factory=list)
    definitions_used: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "content": self.content,
            "token_count": self.token_count,
            "level": self.level,
            "section_title": self.section_title,
            "hierarchy_path": self.hierarchy_path,
            "parent_chunk_id": self.parent_chunk_id,
            "child_chunk_ids": self.child_chunk_ids,
            "page_numbers": self.page_numbers,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "paragraph_start": self.paragraph_start,
            "paragraph_end": self.paragraph_end,
            "original_paragraph_numbers": self.original_paragraph_numbers,
            "contextualized": self.contextualized,
            "context_prefix": self.context_prefix,
            "context_before": self.context_before,
            "context_after": self.context_after,
            "legal_references": self.legal_references,
            "definitions_used": self.definitions_used,
        }


@dataclass
class ChunkConfig:
    """Configuration for chunking parameters."""
    # Token limits per level
    max_tokens_l1: int = 1500  # Section level
    max_tokens_l2: int = 600   # Clause level
    max_tokens_l3: int = 300   # Paragraph level

    # Overlap for context preservation
    overlap_tokens: int = 100

    # Minimum chunk size (don't create tiny chunks)
    min_tokens: int = 50

    # Context window for surrounding content
    context_tokens: int = 100


class LegalChunker:
    """
    Chunks legal documents while preserving structure.

    Creates hierarchical chunks that maintain relationships between
    sections, with overlapping content for context preservation.
    Supports multilingual documents via language_config.
    """

    def __init__(
        self,
        config: Optional[ChunkConfig] = None,
        language_config: Optional[TenantLanguageConfig] = None,
    ):
        """Initialize chunker with optional configuration and language support."""
        self.config = config or ChunkConfig()
        self._language_config = language_config or TenantLanguageConfig.for_language("en")
        self._lang = self._language_config.language

        # Load language-specific patterns from centralized module
        markers = SECTION_MARKERS.get(self._lang, SECTION_MARKERS["en"])
        refs = REFERENCE_PATTERNS.get(self._lang, REFERENCE_PATTERNS["en"])
        self._definition_markers = DEFINITION_MARKERS.get(self._lang, DEFINITION_MARKERS["en"])

        self._section_pattern = re.compile(
            "|".join(markers),
            re.MULTILINE
        )
        self._reference_pattern = re.compile(
            "|".join(refs),
            re.IGNORECASE
        )

    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        """
        Chunk a parsed document into retrieval-ready pieces.

        Args:
            document: ParsedDocument from the parser

        Returns:
            List of Chunk objects with full metadata
        """
        chunks = []
        document_id = document.metadata.document_id

        logger.info(f"Chunking document: {document.metadata.title}")

        # Level 0: Create document summary chunk
        summary_chunk = self._create_summary_chunk(document)
        chunks.append(summary_chunk)

        # Process each section with running paragraph offset
        paragraph_offset = 0
        for section in document.sections:
            section_chunks = self._chunk_section(
                section=section,
                document_id=document_id,
                parent_chunk_id=summary_chunk.chunk_id,
                base_paragraph_offset=paragraph_offset,
            )
            chunks.extend(section_chunks)
            # Update offset based on paragraphs in this section
            section_paragraphs = len([p for p in re.split(r'\n\n+', section.content) if p.strip()])
            paragraph_offset += section_paragraphs

        # Add context between chunks
        self._add_context(chunks)

        # Extract legal references and definitions
        self._extract_legal_metadata(chunks)

        # Link parent-child relationships
        self._link_hierarchy(chunks)

        logger.info(f"Created {len(chunks)} chunks from document")
        return chunks

    def _create_summary_chunk(self, document: ParsedDocument) -> Chunk:
        """Create a document-level summary chunk."""
        # Use first 1000 tokens of content as summary
        summary_text = document.raw_text[:4000]  # ~1000 tokens

        # Truncate at sentence boundary
        last_period = summary_text.rfind('.')
        if last_period > 2000:
            summary_text = summary_text[:last_period + 1]

        summary_label = LABELS.get(self._lang, LABELS["en"])["document_summary"]

        return Chunk(
            chunk_id=str(uuid.uuid4()),
            document_id=document.metadata.document_id,
            content=f"{summary_label} {document.metadata.title}\n\n{summary_text}",
            token_count=self._estimate_tokens(summary_text),
            level=0,
            section_title=document.metadata.title,
            hierarchy_path="Document",
        )

    def _chunk_section(
        self,
        section: DocumentSection,
        document_id: str,
        parent_chunk_id: str,
        base_paragraph_offset: int = 0,
    ) -> list[Chunk]:
        """Chunk a single section, recursively if needed."""
        chunks = []
        content = section.content
        token_count = self._estimate_tokens(content)

        # Extract paragraph numbers for this section
        paragraphs = [p for p in re.split(r'\n\n+', content) if p.strip()]
        para_nums = list(range(base_paragraph_offset + 1, base_paragraph_offset + len(paragraphs) + 1))

        # If section is small enough, create single chunk
        if token_count <= self.config.max_tokens_l2:
            chunk = Chunk(
                chunk_id=str(uuid.uuid4()),
                document_id=document_id,
                content=f"{section.title}\n\n{content}",
                token_count=token_count,
                level=section.level,
                section_title=section.title,
                hierarchy_path=section.hierarchy_path,
                parent_chunk_id=parent_chunk_id,
                page_numbers=section.page_numbers,
                paragraph_start=min(para_nums) if para_nums else None,
                paragraph_end=max(para_nums) if para_nums else None,
                original_paragraph_numbers=para_nums,
            )
            chunks.append(chunk)
            return chunks

        # Section is large - split into smaller chunks
        sub_chunks = self._split_content(
            content=content,
            max_tokens=self.config.max_tokens_l2,
            section_title=section.title,
            hierarchy_path=section.hierarchy_path,
            document_id=document_id,
            parent_chunk_id=parent_chunk_id,
            level=section.level + 1,
            page_numbers=section.page_numbers,
            base_paragraph_offset=base_paragraph_offset,
        )
        chunks.extend(sub_chunks)

        return chunks

    def _split_content(
        self,
        content: str,
        max_tokens: int,
        section_title: str,
        hierarchy_path: str,
        document_id: str,
        parent_chunk_id: str,
        level: int,
        page_numbers: list[int] = None,
        base_paragraph_offset: int = 0,
    ) -> list[Chunk]:
        """Split content into chunks respecting semantic boundaries with paragraph tracking."""
        chunks = []

        # First, try to split on legal section markers
        marker_segments = self._split_on_markers(content)

        if len(marker_segments) == 1:
            # No markers found, split on paragraphs with tracking
            para_segments = self._split_on_paragraphs(content, max_tokens, base_paragraph_offset)
        else:
            # Convert marker segments to paragraph-aware format
            # Each marker segment gets sequential paragraph numbers
            para_segments = []
            current_para = base_paragraph_offset + 1
            for seg in marker_segments:
                # Count paragraphs in this segment
                para_count = len([p for p in re.split(r'\n\n+', seg) if p.strip()])
                para_nums = list(range(current_para, current_para + max(1, para_count)))
                para_segments.append((para_nums, seg))
                current_para += max(1, para_count)

        current_text = ""
        current_tokens = 0
        current_para_nums = []
        chunk_index = 0

        for para_indices, segment in para_segments:
            segment_tokens = self._estimate_tokens(segment)

            # If segment alone exceeds max, split it further
            if segment_tokens > max_tokens:
                # First, save current accumulated text
                if current_text.strip():
                    chunks.append(self._create_chunk(
                        content=current_text,
                        document_id=document_id,
                        parent_chunk_id=parent_chunk_id,
                        level=level,
                        section_title=section_title,
                        hierarchy_path=f"{hierarchy_path}/part_{chunk_index}",
                        page_numbers=page_numbers,
                        paragraph_numbers=current_para_nums,
                    ))
                    chunk_index += 1
                    current_text = ""
                    current_tokens = 0
                    current_para_nums = []

                # Split large segment on sentences (pass paragraph info)
                sub_chunks = self._split_on_sentences(
                    segment, max_tokens, section_title, hierarchy_path,
                    document_id, parent_chunk_id, level, chunk_index, page_numbers,
                    paragraph_numbers=para_indices,
                )
                chunks.extend(sub_chunks)
                chunk_index += len(sub_chunks)
                continue

            # Check if adding segment exceeds limit
            if current_tokens + segment_tokens > max_tokens:
                # Save current chunk
                if current_text.strip():
                    chunks.append(self._create_chunk(
                        content=current_text,
                        document_id=document_id,
                        parent_chunk_id=parent_chunk_id,
                        level=level,
                        section_title=section_title,
                        hierarchy_path=f"{hierarchy_path}/part_{chunk_index}",
                        page_numbers=page_numbers,
                        paragraph_numbers=current_para_nums,
                    ))
                    chunk_index += 1

                # Add overlap from end of previous chunk
                overlap_text = self._get_overlap(current_text)
                current_text = overlap_text + segment
                current_tokens = self._estimate_tokens(current_text)
                current_para_nums = para_indices.copy()
            else:
                current_text += segment
                current_tokens += segment_tokens
                current_para_nums.extend(para_indices)

        # Save remaining text
        if current_text.strip() and self._estimate_tokens(current_text) >= self.config.min_tokens:
            chunks.append(self._create_chunk(
                content=current_text,
                document_id=document_id,
                parent_chunk_id=parent_chunk_id,
                level=level,
                section_title=section_title,
                hierarchy_path=f"{hierarchy_path}/part_{chunk_index}",
                page_numbers=page_numbers,
                paragraph_numbers=current_para_nums,
            ))

        return chunks

    def _split_on_markers(self, content: str) -> list[str]:
        """Split content on legal section markers."""
        splits = self._section_pattern.split(content)

        # Re-attach the markers to the following content
        if len(splits) <= 1:
            return [content]

        result = []
        markers = self._section_pattern.findall(content)

        for i, split in enumerate(splits):
            if i == 0 and split.strip():
                result.append(split)
            elif i > 0 and i - 1 < len(markers):
                result.append(markers[i - 1] + split)
            elif split.strip():
                result.append(split)

        return result if result else [content]

    def _split_on_paragraphs(
        self,
        content: str,
        max_tokens: int,
        base_paragraph_offset: int = 0
    ) -> list[tuple[list[int], str]]:
        """
        Split content on paragraph boundaries with index tracking.

        Args:
            content: Text content to split
            max_tokens: Maximum tokens per segment
            base_paragraph_offset: Starting paragraph number offset (for document-level numbering)

        Returns:
            List of (paragraph_indices, text) tuples where paragraph_indices
            is a list of 1-indexed paragraph numbers included in this segment.
        """
        # Split and preserve paragraph numbers
        paragraphs = re.split(r'\n\n+', content)
        indexed_paragraphs = []

        para_num = base_paragraph_offset + 1
        for p in paragraphs:
            if p.strip():
                indexed_paragraphs.append((para_num, p))
                para_num += 1

        if not indexed_paragraphs:
            return [([base_paragraph_offset + 1], content)]

        # Group into segments under max_tokens
        segments = []
        current_indices = []
        current_text = ""
        current_tokens = 0

        for para_num, para_text in indexed_paragraphs:
            para_tokens = self._estimate_tokens(para_text)

            if current_tokens + para_tokens > max_tokens and current_text:
                # Save current segment
                segments.append((current_indices.copy(), current_text))
                current_indices = [para_num]
                current_text = para_text
                current_tokens = para_tokens
            else:
                current_indices.append(para_num)
                current_text += ("\n\n" if current_text else "") + para_text
                current_tokens += para_tokens

        # Save remaining segment
        if current_text:
            segments.append((current_indices, current_text))

        return segments if segments else [([base_paragraph_offset + 1], content)]

    def _split_on_paragraphs_simple(self, content: str) -> list[str]:
        """Simple paragraph split without index tracking (for backward compatibility)."""
        paragraphs = re.split(r'\n\n+', content)
        return [p for p in paragraphs if p.strip()]

    def _split_on_sentences(
        self,
        content: str,
        max_tokens: int,
        section_title: str,
        hierarchy_path: str,
        document_id: str,
        parent_chunk_id: str,
        level: int,
        start_index: int,
        page_numbers: list[int] = None,
        paragraph_numbers: list[int] = None,
    ) -> list[Chunk]:
        """Split content on sentence boundaries with paragraph tracking."""
        # Simple sentence split (handles common legal abbreviations)
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', content)

        chunks = []
        current_text = ""
        current_tokens = 0
        chunk_index = start_index

        # Distribute paragraph numbers across sentence chunks
        para_nums = paragraph_numbers or []

        for sentence in sentences:
            sentence_tokens = self._estimate_tokens(sentence)

            if current_tokens + sentence_tokens > max_tokens:
                if current_text.strip():
                    chunks.append(self._create_chunk(
                        content=current_text,
                        document_id=document_id,
                        parent_chunk_id=parent_chunk_id,
                        level=level,
                        section_title=section_title,
                        hierarchy_path=f"{hierarchy_path}/part_{chunk_index}",
                        page_numbers=page_numbers,
                        paragraph_numbers=para_nums,  # All sentences share parent's para nums
                    ))
                    chunk_index += 1
                current_text = sentence + " "
                current_tokens = sentence_tokens
            else:
                current_text += sentence + " "
                current_tokens += sentence_tokens

        if current_text.strip() and current_tokens >= self.config.min_tokens:
            chunks.append(self._create_chunk(
                content=current_text,
                document_id=document_id,
                parent_chunk_id=parent_chunk_id,
                level=level,
                section_title=section_title,
                hierarchy_path=f"{hierarchy_path}/part_{chunk_index}",
                page_numbers=page_numbers,
                paragraph_numbers=para_nums,
            ))

        return chunks

    def _create_chunk(
        self,
        content: str,
        document_id: str,
        parent_chunk_id: str,
        level: int,
        section_title: str,
        hierarchy_path: str,
        page_numbers: list[int] = None,
        paragraph_numbers: list[int] = None,
    ) -> Chunk:
        """Create a chunk with all metadata including paragraph tracking."""
        para_nums = paragraph_numbers or []

        return Chunk(
            chunk_id=str(uuid.uuid4()),
            document_id=document_id,
            content=content.strip(),
            token_count=self._estimate_tokens(content),
            level=level,
            section_title=section_title,
            hierarchy_path=hierarchy_path,
            parent_chunk_id=parent_chunk_id,
            page_numbers=page_numbers or [],
            paragraph_start=min(para_nums) if para_nums else None,
            paragraph_end=max(para_nums) if para_nums else None,
            original_paragraph_numbers=para_nums,
        )

    def _get_overlap(self, text: str) -> str:
        """Get overlap text from end of previous chunk."""
        # Get last N tokens worth of text
        words = text.split()
        overlap_words = words[-self.config.overlap_tokens:]
        return " ".join(overlap_words) + " " if overlap_words else ""

    def _add_context(self, chunks: list[Chunk]) -> None:
        """Add context from surrounding chunks."""
        for i, chunk in enumerate(chunks):
            # Add context from previous chunk
            if i > 0:
                prev_content = chunks[i - 1].content
                chunk.context_before = prev_content[-400:]  # ~100 tokens

            # Add context from next chunk
            if i < len(chunks) - 1:
                next_content = chunks[i + 1].content
                chunk.context_after = next_content[:400]

    def _extract_legal_metadata(self, chunks: list[Chunk]) -> None:
        """Extract legal references and definitions from chunks."""
        for chunk in chunks:
            # Find cross-references
            refs = self._reference_pattern.findall(chunk.content)
            chunk.legal_references = list(set(refs))

            # Find defined terms using language-specific markers
            for pattern in self._definition_markers:
                matches = re.findall(pattern, chunk.content)
                chunk.definitions_used.extend(matches)
            chunk.definitions_used = list(set(chunk.definitions_used))

    def _link_hierarchy(self, chunks: list[Chunk]) -> None:
        """Link parent-child relationships in chunks."""
        chunk_map = {c.chunk_id: c for c in chunks}

        for chunk in chunks:
            if chunk.parent_chunk_id and chunk.parent_chunk_id in chunk_map:
                parent = chunk_map[chunk.parent_chunk_id]
                parent.child_chunk_ids.append(chunk.chunk_id)

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count using language-appropriate chars_per_token ratio."""
        return len(text) // self._language_config.chars_per_token

    def contextualize_chunks(
        self,
        chunks: list[Chunk],
        document_summary: str,
        model: str = "meta/llama-3.2-3b-instruct",
    ) -> list[Chunk]:
        """
        Add contextual prefixes to chunks using LLM (Anthropic's Contextual Retrieval method).

        This prepends a short context description to each chunk to improve retrieval
        by providing document-level context. Uses NVIDIA NIM API with Llama 3.2 3B.

        Args:
            chunks: List of chunks to contextualize
            document_summary: Summary of the full document for context
            model: NVIDIA NIM model to use (default: Llama 3.2 3B for speed)

        Returns:
            Same chunks with context_prefix and contextualized fields updated
        """
        if not OPENAI_AVAILABLE:
            logger.warning("OpenAI package not available, skipping contextualization")
            return chunks

        api_key = os.getenv("NVIDIA_API_KEY")
        if not api_key:
            logger.warning("NVIDIA_API_KEY not set, skipping contextualization")
            return chunks

        client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key
        )

        contextualized_chunks = []
        total = len(chunks)

        for i, chunk in enumerate(chunks):
            # Skip summary chunks (level 0)
            if chunk.level == 0:
                contextualized_chunks.append(chunk)
                continue

            try:
                context_prefix = self._generate_context_prefix(
                    client=client,
                    chunk_content=chunk.content,
                    document_summary=document_summary,
                    section_title=chunk.section_title,
                    model=model,
                )

                # Update chunk with context
                chunk.context_prefix = context_prefix
                chunk.contextualized = True
                # Prepend context to content for embedding
                chunk.content = f"{context_prefix}\n\n{chunk.content}"
                chunk.token_count = self._estimate_tokens(chunk.content)

                if (i + 1) % 10 == 0:
                    logger.info(f"Contextualized {i + 1}/{total} chunks")

            except Exception as e:
                logger.warning(f"Failed to contextualize chunk {chunk.chunk_id[:8]}: {e}")
                # Keep original chunk without context

            contextualized_chunks.append(chunk)

        logger.info(f"Contextualized {len([c for c in contextualized_chunks if c.contextualized])}/{total} chunks")
        return contextualized_chunks

    def _generate_context_prefix(
        self,
        client,
        chunk_content: str,
        document_summary: str,
        section_title: str,
        model: str,
    ) -> str:
        """Generate a context prefix for a single chunk using LLM."""
        prompt_template = LLM_PROMPTS.get(self._lang, LLM_PROMPTS["en"])["contextualize_chunk"]
        prompt = prompt_template.format(
            document_summary=document_summary[:1500],
            section_title=section_title,
            chunk_content=chunk_content[:800],
        )

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.0,
        )

        return response.choices[0].message.content.strip()


# CLI for testing
if __name__ == "__main__":
    import sys
    import json
    from .document_parser import LegalDocumentParser

    if len(sys.argv) < 2:
        print("Usage: python chunker.py <pdf_path>")
        sys.exit(1)

    logging.basicConfig(level=logging.INFO)

    parser = LegalDocumentParser()
    document = parser.parse(sys.argv[1])

    chunker = LegalChunker()
    chunks = chunker.chunk(document)

    print(f"\nCreated {len(chunks)} chunks:")
    for chunk in chunks[:5]:
        print(f"\n--- Chunk {chunk.chunk_id[:8]} (Level {chunk.level}) ---")
        print(f"Section: {chunk.section_title}")
        print(f"Path: {chunk.hierarchy_path}")
        print(f"Tokens: {chunk.token_count}")
        print(f"Content preview: {chunk.content[:200]}...")
