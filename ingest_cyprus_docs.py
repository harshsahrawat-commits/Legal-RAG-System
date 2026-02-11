"""
Batch ingestion of Cypriot legal documents into the RAG system.

Processes all PDFs (and .txt files) in a directory using the Greek language pipeline:
- Parser: LegalDocumentParser with Greek config
- Chunker: LegalChunker with Greek section markers
- Embeddings: Voyage AI voyage-multilingual-2
- Storage: PostgreSQL + pgvector with client_id isolation

Usage:
    python ingest_cyprus_docs.py --dir ~/test_files_cy/
    python ingest_cyprus_docs.py --dir ~/test_files_cy/ --client-id 00000000-0000-0000-0000-000000000002
"""

import os
import sys
import uuid
import time
import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

CYPRUS_CLIENT_ID = "00000000-0000-0000-0000-000000000002"


def is_already_ingested(store, filepath: Path, client_id: str) -> bool:
    """Check if a document with this file_path is already in the database."""
    with store.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM legal_documents WHERE file_path = %s AND client_id = %s::uuid LIMIT 1",
                (str(filepath), client_id),
            )
            return cur.fetchone() is not None


def ingest_pdf(
    filepath: Path,
    parser,
    chunker,
    embedding_service,
    store,
    client_id: str,
) -> int:
    """Ingest a single PDF file. Returns number of chunks created."""
    # Parse
    parsed = parser.parse(str(filepath), client_id=client_id)
    if not parsed.raw_text or len(parsed.raw_text.strip()) < 50:
        logger.warning(f"  Skipping {filepath.name}: too little text extracted")
        return 0

    # Chunk
    chunks = chunker.chunk(parsed)
    if not chunks:
        logger.warning(f"  Skipping {filepath.name}: no chunks produced")
        return 0

    # Embed (in batches to respect API limits)
    chunk_texts = [c.content for c in chunks]
    embeddings = embedding_service.embed_documents(chunk_texts)

    # Store document record
    store.insert_document(
        document_id=parsed.metadata.document_id,
        title=parsed.metadata.title,
        document_type=parsed.metadata.document_type,
        client_id=client_id,
        jurisdiction=parsed.metadata.jurisdiction,
        file_path=str(filepath),
        page_count=parsed.metadata.page_count,
    )

    # Store chunks
    chunk_dicts = [c.to_dict() for c in chunks]
    store.insert_chunks(chunk_dicts, embeddings, client_id=client_id)

    return len(chunks)


def ingest_text_file(
    filepath: Path,
    chunker,
    embedding_service,
    store,
    client_id: str,
) -> int:
    """Ingest a plain text file (HTML scrape fallback). Returns chunk count."""
    from execution.legal_rag.document_parser import ParsedDocument, LegalMetadata, DocumentSection

    text = filepath.read_text(encoding="utf-8")
    if len(text.strip()) < 100:
        logger.warning(f"  Skipping {filepath.name}: too short")
        return 0

    # Create a minimal ParsedDocument from text
    doc_id = str(uuid.uuid4())
    title = filepath.stem.replace("law_", "Law ").replace("_", " ")

    # Try to extract a better title from first non-empty line
    for line in text.split("\n"):
        line = line.strip()
        if len(line) > 10:
            title = line[:200]
            break

    metadata = LegalMetadata(
        document_id=doc_id,
        title=title,
        document_type="statute",
        page_count=0,
        file_path=str(filepath),
    )

    # Create a single section from the full text
    section = DocumentSection(
        section_id=str(uuid.uuid4()),
        title=title,
        content=text,
        level=0,
        hierarchy_path="Document",
    )

    parsed = ParsedDocument(
        metadata=metadata,
        sections=[section],
        raw_text=text,
        raw_markdown=text,
    )

    # Chunk
    chunks = chunker.chunk(parsed)
    if not chunks:
        return 0

    # Embed
    chunk_texts = [c.content for c in chunks]
    embeddings = embedding_service.embed_documents(chunk_texts)

    # Store
    store.insert_document(
        document_id=doc_id,
        title=title,
        document_type="statute",
        client_id=client_id,
        file_path=str(filepath),
        page_count=0,
    )
    chunk_dicts = [c.to_dict() for c in chunks]
    store.insert_chunks(chunk_dicts, embeddings, client_id=client_id)

    return len(chunks)


def main():
    arg_parser = argparse.ArgumentParser(description="Ingest Cypriot legal documents")
    arg_parser.add_argument(
        "--dir",
        type=str,
        required=True,
        help="Directory containing PDFs and/or text files",
    )
    arg_parser.add_argument(
        "--client-id",
        type=str,
        default=CYPRUS_CLIENT_ID,
        help=f"Tenant client ID (default: {CYPRUS_CLIENT_ID})",
    )
    arg_parser.add_argument(
        "--skip-ocr",
        action="store_true",
        help="Pass 1: Skip scanned PDFs (only ingest text-extractable ones)",
    )
    arg_parser.add_argument(
        "--ocr-only",
        action="store_true",
        help="Pass 2: Only process PDFs that need OCR (skip text-extractable ones)",
    )
    args = arg_parser.parse_args()

    input_dir = Path(args.dir)
    if not input_dir.exists():
        logger.error(f"Directory not found: {input_dir}")
        sys.exit(1)

    # Collect files
    pdf_files = sorted(input_dir.glob("*.pdf"))
    txt_files = sorted(input_dir.glob("*.txt"))
    total_files = len(pdf_files) + len(txt_files)

    if total_files == 0:
        logger.error(f"No PDF or TXT files found in {input_dir}")
        sys.exit(1)

    logger.info(f"Found {len(pdf_files)} PDFs and {len(txt_files)} text files in {input_dir}")

    # Initialize services with Greek config
    from execution.legal_rag.language_config import TenantLanguageConfig
    from execution.legal_rag.document_parser import LegalDocumentParser
    from execution.legal_rag.chunker import LegalChunker
    from execution.legal_rag.embeddings import get_embedding_service
    from execution.legal_rag.vector_store import VectorStore

    greek_config = TenantLanguageConfig.for_language("el")

    store = VectorStore()
    store.connect()
    store.initialize_schema()
    store.set_tenant_context(args.client_id)

    parser = LegalDocumentParser(
        language_config=greek_config,
        use_docling=False,
        enable_ocr=not args.skip_ocr,
    )
    chunker = LegalChunker(language_config=greek_config)
    embedding_service = get_embedding_service(
        provider=greek_config.embedding_provider,
        language_config=greek_config,
    )

    mode = "ocr-only" if args.ocr_only else ("skip-ocr" if args.skip_ocr else "full")
    logger.info(f"Greek pipeline initialized (mode: {mode}):")
    logger.info(f"  Embedding model: {greek_config.embedding_model}")
    logger.info(f"  FTS language: {greek_config.fts_language}")
    logger.info(f"  Client ID: {args.client_id}")

    # Helper: check if a PDF has extractable text (not scanned)
    def pdf_has_text(filepath: Path) -> bool:
        """Quick check if PyMuPDF can extract text from this PDF."""
        import fitz
        try:
            with fitz.open(str(filepath)) as doc:
                text = "".join(page.get_text() for page in doc[:3])  # Check first 3 pages
                return len(text.strip()) >= 50
        except Exception:
            return False

    # Process files
    start_time = time.time()
    total_chunks = 0
    success_count = 0
    fail_count = 0
    skip_count = 0

    # PDFs
    for i, pdf_path in enumerate(pdf_files):
        # Skip based on mode
        has_text = pdf_has_text(pdf_path)
        if args.skip_ocr and not has_text:
            skip_count += 1
            if (i + 1) % 500 == 0:
                logger.info(f"  [{i+1}/{total_files}] Skipped {skip_count} scanned PDFs so far")
            continue
        if args.ocr_only and has_text:
            skip_count += 1
            continue

        # Skip already-ingested documents
        if is_already_ingested(store, pdf_path, args.client_id):
            skip_count += 1
            if skip_count % 100 == 0:
                logger.info(f"  Skipped {skip_count} already-ingested files so far")
            continue

        logger.info(f"[{i+1}/{total_files}] Processing: {pdf_path.name}")
        try:
            n_chunks = ingest_pdf(
                pdf_path, parser, chunker, embedding_service, store, args.client_id
            )
            total_chunks += n_chunks
            success_count += 1
            logger.info(f"  -> {n_chunks} chunks")
        except Exception as e:
            fail_count += 1
            logger.error(f"  FAILED: {e}")

    # Text files
    for i, txt_path in enumerate(txt_files):
        idx = len(pdf_files) + i + 1
        if txt_path.name == "manifest.json":
            continue
        logger.info(f"[{idx}/{total_files}] Processing: {txt_path.name}")
        try:
            n_chunks = ingest_text_file(
                txt_path, chunker, embedding_service, store, args.client_id
            )
            total_chunks += n_chunks
            success_count += 1
            logger.info(f"  -> {n_chunks} chunks")
        except Exception as e:
            fail_count += 1
            logger.error(f"  FAILED: {e}")

    elapsed = time.time() - start_time
    store.close()

    # Summary
    print("\n" + "=" * 60)
    print("INGESTION COMPLETE")
    print("=" * 60)
    print(f"Files processed: {success_count}/{total_files} ({fail_count} failed, {skip_count} skipped)")
    print(f"Total chunks:    {total_chunks}")
    print(f"Time elapsed:    {elapsed:.1f}s")
    print(f"Client ID:       {args.client_id}")
    print("=" * 60)


if __name__ == "__main__":
    main()
