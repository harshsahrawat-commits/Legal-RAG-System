"""
Pass 2: Ingest scanned Cypriot legal PDFs using Google Cloud Vision OCR.

Processes all scanned PDFs (those without extractable text) using Google Cloud
Vision API's DOCUMENT_TEXT_DETECTION, then chunks, embeds, and stores them.

Uses batch requests (16 pages per API call) to minimize HTTP overhead.
Skips already-ingested documents automatically.

Usage:
    python ingest_cyprus_gcv.py --dir ~/test_files_cy/
    python ingest_cyprus_gcv.py --dir ~/test_files_cy/ --max-pages 50
"""

import os
import sys
import uuid
import time
import base64
import argparse
import logging
import unicodedata
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests as http_requests
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
GCV_API_KEY = os.getenv("GOOGLE_CLOUD_VISION_API_KEY")
GCV_URL = f"https://vision.googleapis.com/v1/images:annotate?key={GCV_API_KEY}"
BATCH_URL = f"https://vision.googleapis.com/v1/images:annotate?key={GCV_API_KEY}"

# Max pages per batch request (Cloud Vision supports up to 16)
GCV_BATCH_SIZE = 16


def pdf_has_text(filepath: Path) -> bool:
    """Quick check if PyMuPDF can extract text from this PDF."""
    import fitz
    try:
        with fitz.open(str(filepath)) as doc:
            text = "".join(page.get_text() for page in doc[:3])
            return len(text.strip()) >= 50
    except Exception:
        return False


def is_already_ingested(store, filepath: Path, client_id: str) -> bool:
    """Check if a document with this file_path is already in the database."""
    with store.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM legal_documents WHERE file_path = %s AND client_id = %s::uuid LIMIT 1",
                (str(filepath), client_id),
            )
            return cur.fetchone() is not None


def normalize_greek_text(text: str) -> str:
    """Normalize Unicode to NFC — critical for Greek diacritics in vector search."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u2126", "\u03A9")  # Ohm → Omega
    text = text.replace("\u00B5", "\u03BC")  # Micro → Mu
    text = text.replace("\u200b", "")  # Zero-width space
    text = text.replace("\ufeff", "")  # BOM
    return text


def ocr_pages_with_gcv(filepath: Path, max_pages: int = 0) -> tuple[str, int, list[tuple[int, int, int]]]:
    """
    OCR all pages of a PDF using Google Cloud Vision API.

    Returns (markdown_text, page_count, page_ranges).
    """
    import fitz

    with fitz.open(str(filepath)) as doc:
        page_count = len(doc)
        if max_pages > 0:
            page_count = min(page_count, max_pages)

        if page_count == 0:
            return "", 0, []

        markdown_parts = []
        page_ranges = []
        cumulative_char = 0

        # Process pages in batches of GCV_BATCH_SIZE
        for batch_start in range(0, page_count, GCV_BATCH_SIZE):
            batch_end = min(batch_start + GCV_BATCH_SIZE, page_count)
            batch_requests = []

            for page_idx in range(batch_start, batch_end):
                page = doc[page_idx]
                pix = page.get_pixmap(dpi=300)
                img_bytes = pix.tobytes("png")
                img_b64 = base64.b64encode(img_bytes).decode()

                batch_requests.append({
                    "image": {"content": img_b64},
                    "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
                })

            # Send batch request
            payload = {"requests": batch_requests}
            resp = http_requests.post(BATCH_URL, json=payload, timeout=120)
            resp.raise_for_status()
            result = resp.json()

            # Process responses
            for i, response in enumerate(result.get("responses", [])):
                page_idx = batch_start + i
                page_start = cumulative_char

                page_header = f"\n## Page {page_idx + 1}\n\n"
                markdown_parts.append(page_header)
                cumulative_char += len(page_header)

                if "fullTextAnnotation" in response:
                    page_text = response["fullTextAnnotation"]["text"]
                    page_text = normalize_greek_text(page_text)
                    markdown_parts.append(page_text + "\n")
                    cumulative_char += len(page_text) + 1

                page_end = cumulative_char
                page_ranges.append((page_idx + 1, page_start, page_end))

    markdown = "".join(markdown_parts)
    return markdown, page_count, page_ranges


def ingest_scanned_pdf(
    filepath: Path,
    chunker,
    embedding_service,
    store,
    client_id: str,
    max_pages: int = 0,
) -> int:
    """Ingest a single scanned PDF via Cloud Vision OCR. Returns chunk count."""
    from execution.legal_rag.document_parser import ParsedDocument, LegalMetadata, DocumentSection

    # OCR with Google Cloud Vision
    raw_markdown, page_count, page_ranges = ocr_pages_with_gcv(filepath, max_pages)
    raw_text = raw_markdown  # Already plain text from GCV

    if len(raw_text.strip()) < 50:
        logger.warning(f"  Skipping {filepath.name}: too little text from OCR ({len(raw_text.strip())} chars)")
        return 0

    # Create ParsedDocument
    doc_id = str(uuid.uuid4())

    # Extract title from first meaningful line
    title = filepath.stem.replace("_", " ")
    for line in raw_text.split("\n"):
        line = line.strip()
        if len(line) > 10 and not line.startswith("##"):
            title = line[:200]
            break

    metadata = LegalMetadata(
        document_id=doc_id,
        title=title,
        document_type="statute",
        page_count=page_count,
        file_path=str(filepath),
    )

    section = DocumentSection(
        section_id=str(uuid.uuid4()),
        title=title,
        content=raw_text,
        level=0,
        hierarchy_path="Document",
    )

    parsed = ParsedDocument(
        metadata=metadata,
        sections=[section],
        raw_text=raw_text,
        raw_markdown=raw_markdown,
    )

    # Chunk
    chunks = chunker.chunk(parsed)
    if not chunks:
        logger.warning(f"  Skipping {filepath.name}: no chunks produced")
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
        page_count=page_count,
    )
    chunk_dicts = [c.to_dict() for c in chunks]
    store.insert_chunks(chunk_dicts, embeddings, client_id=client_id)

    return len(chunks)


def main():
    if not GCV_API_KEY:
        logger.error("GOOGLE_CLOUD_VISION_API_KEY not set in .env")
        sys.exit(1)

    arg_parser = argparse.ArgumentParser(description="Ingest scanned Cypriot PDFs via Google Cloud Vision OCR")
    arg_parser.add_argument("--dir", type=str, required=True, help="Directory containing PDFs")
    arg_parser.add_argument("--client-id", type=str, default=CYPRUS_CLIENT_ID)
    arg_parser.add_argument("--max-pages", type=int, default=0, help="Max pages per PDF (0 = no limit)")
    args = arg_parser.parse_args()

    input_dir = Path(args.dir)
    if not input_dir.exists():
        logger.error(f"Directory not found: {input_dir}")
        sys.exit(1)

    pdf_files = sorted(input_dir.glob("*.pdf"))
    if not pdf_files:
        logger.error(f"No PDFs found in {input_dir}")
        sys.exit(1)

    logger.info(f"Found {len(pdf_files)} PDFs in {input_dir}")

    # Initialize services
    from execution.legal_rag.language_config import TenantLanguageConfig
    from execution.legal_rag.chunker import LegalChunker
    from execution.legal_rag.embeddings import get_embedding_service
    from execution.legal_rag.vector_store import VectorStore

    greek_config = TenantLanguageConfig.for_language("el")

    store = VectorStore()
    store.connect()
    store.initialize_schema()
    store.set_tenant_context(args.client_id)

    chunker = LegalChunker(language_config=greek_config)
    embedding_service = get_embedding_service(
        provider=greek_config.embedding_provider,
        language_config=greek_config,
    )

    logger.info(f"Cloud Vision OCR pipeline initialized:")
    logger.info(f"  Embedding model: {greek_config.embedding_model}")
    logger.info(f"  Client ID: {args.client_id}")
    logger.info(f"  GCV batch size: {GCV_BATCH_SIZE} pages/request")

    # Process files
    start_time = time.time()
    total_chunks = 0
    success_count = 0
    fail_count = 0
    skip_digital = 0
    skip_ingested = 0
    skip_no_text = 0
    total_pages_ocrd = 0

    for i, pdf_path in enumerate(pdf_files):
        # Skip digital PDFs (already ingested in Pass 1)
        if pdf_has_text(pdf_path):
            skip_digital += 1
            if skip_digital % 500 == 0:
                logger.info(f"  Skipped {skip_digital} digital PDFs so far")
            continue

        # Skip already-ingested
        if is_already_ingested(store, pdf_path, args.client_id):
            skip_ingested += 1
            if skip_ingested % 100 == 0:
                logger.info(f"  Skipped {skip_ingested} already-ingested files so far")
            continue

        logger.info(f"[{i+1}/{len(pdf_files)}] OCR: {pdf_path.name}")
        try:
            n_chunks = ingest_scanned_pdf(
                pdf_path, chunker, embedding_service, store, args.client_id, args.max_pages
            )
            if n_chunks > 0:
                total_chunks += n_chunks
                success_count += 1
                logger.info(f"  -> {n_chunks} chunks")
            else:
                skip_no_text += 1
        except Exception as e:
            fail_count += 1
            logger.error(f"  FAILED: {e}")

    elapsed = time.time() - start_time
    store.close()

    # Summary
    print("\n" + "=" * 60)
    print("CLOUD VISION OCR INGESTION COMPLETE")
    print("=" * 60)
    print(f"Successfully ingested: {success_count}")
    print(f"Failed:               {fail_count}")
    print(f"Skipped (digital):    {skip_digital}")
    print(f"Skipped (ingested):   {skip_ingested}")
    print(f"Skipped (no text):    {skip_no_text}")
    print(f"Total chunks:         {total_chunks}")
    print(f"Time elapsed:         {elapsed:.1f}s")
    print(f"Client ID:            {args.client_id}")
    print("=" * 60)


if __name__ == "__main__":
    main()
