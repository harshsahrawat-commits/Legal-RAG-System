#!/usr/bin/env python3
"""
Re-ingest all documents with contextual chunking enabled.
This script processes all PDFs in the test_files folder with:
- Paragraph tracking
- voyage-law-2 embeddings
- Contextual chunking (Anthropic's method)
"""

import os
import sys
import time
import logging
from pathlib import Path
from dotenv import load_dotenv

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# Demo client ID for ingestion
DEMO_CLIENT_ID = "00000000-0000-0000-0000-000000000001"


def reingest_all_documents(
    pdf_folder: str,
    clear_existing: bool = True,
    delay_between_docs: float = 1.0,
):
    """
    Re-ingest all documents with contextual chunking.

    Args:
        pdf_folder: Path to folder containing PDFs
        clear_existing: Whether to clear existing chunks first
        delay_between_docs: Delay between documents (rate limiting)
    """
    from execution.legal_rag.document_parser import LegalDocumentParser
    from execution.legal_rag.chunker import LegalChunker
    from execution.legal_rag.embeddings import get_embedding_service
    from execution.legal_rag.vector_store import VectorStore

    # Initialize services
    logger.info("Initializing services...")
    parser = LegalDocumentParser()
    chunker = LegalChunker()
    embeddings = get_embedding_service(provider="voyage")  # voyage-law-2
    store = VectorStore()
    store.connect()
    store.initialize_schema()

    # Get all PDFs
    pdf_path = Path(pdf_folder)
    if not pdf_path.exists():
        logger.error(f"PDF folder not found: {pdf_folder}")
        return

    pdfs = list(pdf_path.glob("*.pdf"))
    logger.info(f"Found {len(pdfs)} PDF files")

    if not pdfs:
        logger.error("No PDF files found!")
        return

    # Optionally clear existing data
    if clear_existing:
        logger.info("Clearing existing chunks...")
        try:
            with store.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM document_chunks WHERE client_id = %s", (DEMO_CLIENT_ID,))
                    cur.execute("DELETE FROM legal_documents WHERE client_id = %s", (DEMO_CLIENT_ID,))
                    conn.commit()
            logger.info("Existing data cleared")
        except Exception as e:
            logger.warning(f"Could not clear existing data: {e}")

    # Process each document
    successful = 0
    failed = 0
    total_chunks = 0
    contextualized_chunks = 0

    for i, pdf_path in enumerate(pdfs):
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {i+1}/{len(pdfs)}: {pdf_path.name}")
        logger.info("="*60)

        try:
            # Parse document
            logger.info("  Parsing document...")
            parsed = parser.parse(str(pdf_path))

            # Chunk document
            logger.info("  Creating chunks...")
            chunks = chunker.chunk(parsed)
            logger.info(f"  Created {len(chunks)} chunks")

            # Contextualize chunks (Anthropic's method)
            logger.info("  Adding document context...")
            document_summary = parsed.raw_text[:2000] if parsed.raw_text else ""
            if document_summary:
                chunks = chunker.contextualize_chunks(
                    chunks=chunks,
                    document_summary=document_summary,
                )
                ctx_count = sum(1 for c in chunks if c.contextualized)
                logger.info(f"  Contextualized {ctx_count}/{len(chunks)} chunks")
                contextualized_chunks += ctx_count

            # Generate embeddings
            logger.info("  Generating embeddings with voyage-law-2...")
            chunk_texts = [c.content for c in chunks]
            chunk_embeddings = embeddings.embed_documents(chunk_texts)

            # Store in database
            logger.info("  Storing in database...")
            store.insert_document(
                document_id=parsed.metadata.document_id,
                title=parsed.metadata.title,
                document_type=parsed.metadata.document_type,
                client_id=DEMO_CLIENT_ID,
                jurisdiction=parsed.metadata.jurisdiction,
                file_path=str(pdf_path),
                page_count=parsed.metadata.page_count,
            )

            chunk_dicts = [c.to_dict() for c in chunks]
            store.insert_chunks(chunk_dicts, chunk_embeddings, client_id=DEMO_CLIENT_ID)

            successful += 1
            total_chunks += len(chunks)
            logger.info(f"  ✅ Success: {parsed.metadata.title}")

        except Exception as e:
            failed += 1
            logger.error(f"  ❌ Failed: {e}")
            import traceback
            traceback.print_exc()

        # Rate limiting
        if i < len(pdfs) - 1:
            time.sleep(delay_between_docs)

    # Summary
    logger.info("\n" + "="*60)
    logger.info("RE-INGESTION COMPLETE")
    logger.info("="*60)
    logger.info(f"Documents processed: {successful}/{len(pdfs)}")
    logger.info(f"Documents failed: {failed}")
    logger.info(f"Total chunks: {total_chunks}")
    logger.info(f"Contextualized chunks: {contextualized_chunks}")

    # Verify in database
    try:
        with store.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM document_chunks WHERE client_id = %s AND contextualized = true",
                    (DEMO_CLIENT_ID,)
                )
                db_ctx_count = cur.fetchone()[0]
                logger.info(f"Verified in DB: {db_ctx_count} contextualized chunks")
    except Exception as e:
        logger.warning(f"Could not verify: {e}")

    store.close()
    return successful, failed, total_chunks


if __name__ == "__main__":
    import argparse

    arg_parser = argparse.ArgumentParser(description="Re-ingest documents with contextual chunking")
    arg_parser.add_argument(
        "--pdf-folder",
        default=str(Path(__file__).parent.parent.parent / "test_files"),
        help="Path to folder containing PDFs"
    )
    arg_parser.add_argument(
        "--no-clear",
        action="store_true",
        help="Don't clear existing data before re-ingestion"
    )
    arg_parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between documents in seconds"
    )

    args = arg_parser.parse_args()

    reingest_all_documents(
        pdf_folder=args.pdf_folder,
        clear_existing=not args.no_clear,
        delay_between_docs=args.delay,
    )
