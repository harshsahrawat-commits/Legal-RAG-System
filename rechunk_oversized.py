"""
Re-chunk oversized database chunks for the Cyprus tenant.

Queries the DB for all chunks >5,000 chars, splits them using LegalChunker,
re-embeds with Voyage multilingual-2, and replaces old chunks with properly-sized
new ones — all within per-chunk transactions to prevent data loss.

Usage:
    python rechunk_oversized.py --dry-run          # Preview without changes
    python rechunk_oversized.py                     # Execute for real
    python rechunk_oversized.py --threshold 3000    # Custom char threshold
"""

import os
import sys
import uuid
import time
import json
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


def get_oversized_chunks(conn, client_id: str, threshold: int) -> list[dict]:
    """Fetch all chunks exceeding the character threshold."""
    sql = """
        SELECT id, content, document_id, section_title, hierarchy_path,
               level, page_numbers, parent_chunk_id
        FROM document_chunks
        WHERE client_id = %s::uuid AND LENGTH(content) > %s
        ORDER BY LENGTH(content) DESC
    """
    with conn.cursor() as cur:
        cur.execute(sql, (client_id, threshold))
        columns = [desc[0] for desc in cur.description]
        rows = [dict(zip(columns, row)) for row in cur.fetchall()]
    logger.info(f"Found {len(rows)} chunks > {threshold} chars")
    return rows


def force_split_text(text: str, max_chars: int = 1800) -> list[str]:
    """
    Force-split text into pieces of max_chars, preferring sentence/word boundaries.
    Used as a fallback when the chunker can't find semantic boundaries.
    """
    import re as _re

    if len(text) <= max_chars:
        return [text]

    pieces = []
    remaining = text
    while len(remaining) > max_chars:
        # Try to break at a sentence boundary (. ! ? ; ·) near max_chars
        candidate = remaining[:max_chars]
        # Search backward from max_chars for the last sentence-ending punctuation
        break_pos = -1
        for m in _re.finditer(r'[.!?;·]\s', candidate):
            break_pos = m.end()

        if break_pos > max_chars // 3:
            # Found a reasonable break point
            pieces.append(remaining[:break_pos].strip())
            remaining = remaining[break_pos:].strip()
        else:
            # No sentence boundary — break at last space
            space_pos = candidate.rfind(' ')
            if space_pos > max_chars // 3:
                pieces.append(remaining[:space_pos].strip())
                remaining = remaining[space_pos:].strip()
            else:
                # No space either — hard break
                pieces.append(remaining[:max_chars].strip())
                remaining = remaining[max_chars:].strip()

    if remaining.strip():
        pieces.append(remaining.strip())

    return pieces


def split_chunk(chunker, chunk_row: dict, threshold: int = 5000) -> list[dict]:
    """Split an oversized chunk into properly-sized sub-chunks using LegalChunker."""
    content = chunk_row["content"]
    document_id = str(chunk_row["document_id"])
    section_title = chunk_row["section_title"] or ""
    hierarchy_path = chunk_row["hierarchy_path"] or ""
    level = chunk_row["level"] or 2
    page_numbers = chunk_row["page_numbers"] or []
    parent_chunk_id = chunk_row["parent_chunk_id"]

    # Use the chunker's _split_content to get properly-sized Chunk objects
    sub_chunks = chunker._split_content(
        content=content,
        max_tokens=600,  # Target ~1,800 chars for Greek (600 tokens * 3 chars/token)
        section_title=section_title,
        hierarchy_path=hierarchy_path,
        document_id=document_id,
        parent_chunk_id=parent_chunk_id or str(uuid.uuid4()),
        level=level,
        page_numbers=page_numbers,
    )

    result = []
    for chunk in sub_chunks:
        d = chunk.to_dict()
        # If the chunker couldn't split below threshold, force-split
        if len(d["content"]) > threshold:
            pieces = force_split_text(d["content"], max_chars=1800)
            for i, piece in enumerate(pieces):
                sub = dict(d)
                sub["chunk_id"] = str(uuid.uuid4())
                sub["content"] = piece
                sub["token_count"] = max(1, len(piece) // 3)
                sub["hierarchy_path"] = f"{d['hierarchy_path']}/split_{i}"
                result.append(sub)
        else:
            result.append(d)

    return result


def insert_new_chunks(conn, chunks: list[dict], embeddings: list[list[float]], client_id: str):
    """Batch insert new sub-chunks with embeddings."""
    from psycopg2.extras import execute_values

    sql = """
        INSERT INTO document_chunks
            (id, document_id, client_id, content, section_title, hierarchy_path,
             level, page_numbers, paragraph_start, paragraph_end, original_paragraph_numbers,
             contextualized, context_prefix, parent_chunk_id, token_count, embedding,
             legal_references, context_before, context_after, metadata)
        VALUES %s
    """

    values = []
    for chunk, embedding in zip(chunks, embeddings):
        values.append((
            chunk["chunk_id"],
            chunk["document_id"],
            client_id,
            chunk["content"],
            chunk["section_title"],
            chunk["hierarchy_path"],
            chunk["level"],
            chunk.get("page_numbers", []),
            chunk.get("paragraph_start"),
            chunk.get("paragraph_end"),
            chunk.get("original_paragraph_numbers", []),
            chunk.get("contextualized", False),
            chunk.get("context_prefix", ""),
            chunk.get("parent_chunk_id"),
            chunk["token_count"],
            embedding,
            chunk.get("legal_references", []),
            chunk.get("context_before", ""),
            chunk.get("context_after", ""),
            json.dumps(chunk.get("metadata", {})),
        ))

    with conn.cursor() as cur:
        execute_values(
            cur,
            sql,
            values,
            template="(%s::uuid, %s::uuid, %s::uuid, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::uuid, %s, %s::vector, %s, %s, %s, %s)",
            page_size=500,
        )


def delete_chunk(conn, chunk_id: str, client_id: str):
    """Delete a single oversized chunk."""
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM document_chunks WHERE id = %s::uuid AND client_id = %s::uuid",
            (chunk_id, client_id),
        )


def rechunk_batch(
    conn,
    batch: list[dict],
    chunker,
    embedding_service,
    client_id: str,
    dry_run: bool,
) -> tuple[int, int, int]:
    """
    Process a batch of oversized chunks.

    Returns: (chunks_processed, new_chunks_created, errors)
    """
    processed = 0
    created = 0
    errors = 0

    # Phase 1: Split all chunks in this batch
    all_new_chunks = []  # List of (old_chunk_id, [new_chunk_dicts])
    all_texts_to_embed = []

    for chunk_row in batch:
        old_id = str(chunk_row["id"])
        old_len = len(chunk_row["content"])

        try:
            new_chunks = split_chunk(chunker, chunk_row)
        except Exception as e:
            logger.error(f"  Split failed for chunk {old_id[:8]} ({old_len} chars): {e}")
            errors += 1
            continue

        if not new_chunks:
            logger.warning(f"  Chunk {old_id[:8]} produced 0 sub-chunks, skipping")
            errors += 1
            continue

        # Collect texts for batch embedding
        texts = [c["content"] for c in new_chunks]
        all_new_chunks.append((old_id, new_chunks, len(texts)))
        all_texts_to_embed.extend(texts)

        logger.info(
            f"  Chunk {old_id[:8]}: {old_len:,} chars → {len(new_chunks)} sub-chunks"
        )

    if dry_run or not all_texts_to_embed:
        return len(batch), sum(n for _, _, n in all_new_chunks), errors

    # Phase 2: Embed all new chunk texts in one batch call
    logger.info(f"  Embedding {len(all_texts_to_embed)} new chunk texts...")
    try:
        all_embeddings = embedding_service.embed_documents(all_texts_to_embed)
    except Exception as e:
        logger.error(f"  Embedding failed for batch: {e}")
        return 0, 0, len(batch)

    # Phase 3: Delete old + insert new, per chunk, in transactions
    embed_offset = 0
    for old_id, new_chunks, n_texts in all_new_chunks:
        chunk_embeddings = all_embeddings[embed_offset : embed_offset + n_texts]
        embed_offset += n_texts

        try:
            # Single transaction: delete old, insert new
            delete_chunk(conn, old_id, client_id)
            insert_new_chunks(conn, new_chunks, chunk_embeddings, client_id)
            conn.commit()
            processed += 1
            created += len(new_chunks)
        except Exception as e:
            conn.rollback()
            logger.error(f"  Transaction failed for chunk {old_id[:8]}: {e}")
            errors += 1

    return processed, created, errors


def main():
    parser = argparse.ArgumentParser(description="Re-chunk oversized DB chunks")
    parser.add_argument("--db-url", default=os.getenv("POSTGRES_URL"), help="PostgreSQL connection URL")
    parser.add_argument("--threshold", type=int, default=5000, help="Character threshold (default: 5000)")
    parser.add_argument("--batch-size", type=int, default=50, help="Chunks to process per embedding batch (default: 50)")
    parser.add_argument("--client-id", default=CYPRUS_CLIENT_ID, help="Client ID to process")
    parser.add_argument("--dry-run", action="store_true", help="Preview splits without touching DB")
    args = parser.parse_args()

    if not args.db_url:
        logger.error("No database URL. Set POSTGRES_URL env var or pass --db-url")
        sys.exit(1)

    # Initialize components
    from execution.legal_rag.language_config import TenantLanguageConfig
    from execution.legal_rag.chunker import LegalChunker, ChunkConfig
    from execution.legal_rag.embeddings import get_embedding_service

    lang_config = TenantLanguageConfig.for_language("el")
    chunker = LegalChunker(
        config=ChunkConfig(max_tokens_l2=600),
        language_config=lang_config,
    )
    logger.info(f"Initialized Greek chunker (max_tokens_l2=600, ~1800 chars/chunk)")

    if not args.dry_run:
        embedding_service = get_embedding_service(
            provider="voyage",
            language_config=lang_config,
        )
        # Voyage multilingual tokenizer is more aggressive than 1.0 chars/token
        # for Greek. Reduce to 0.75 to stay well under 120K token batch limit.
        embedding_service.config.chars_per_token = 0.75
        embedding_service.config.max_tokens_per_batch = 80000
        logger.info(f"Initialized Voyage embedding service ({lang_config.embedding_model}, "
                     f"cpt=0.75, max_batch_tokens=80K)")
    else:
        embedding_service = None
        logger.info("DRY RUN — no embeddings will be generated, no DB changes")

    # Connect to DB
    import psycopg2
    conn = psycopg2.connect(args.db_url)
    logger.info("Connected to database")

    try:
        # Fetch oversized chunks
        oversized = get_oversized_chunks(conn, args.client_id, args.threshold)

        if not oversized:
            logger.info("No oversized chunks found. Nothing to do.")
            return

        # Show size distribution
        sizes = [len(row["content"]) for row in oversized]
        logger.info(f"Size range: {min(sizes):,} - {max(sizes):,} chars")
        logger.info(f"Unique documents: {len(set(str(row['document_id']) for row in oversized))}")

        # Process in batches
        total_processed = 0
        total_created = 0
        total_errors = 0
        start_time = time.time()

        for batch_start in range(0, len(oversized), args.batch_size):
            batch = oversized[batch_start : batch_start + args.batch_size]
            batch_num = batch_start // args.batch_size + 1
            total_batches = (len(oversized) + args.batch_size - 1) // args.batch_size

            logger.info(f"\n--- Batch {batch_num}/{total_batches} ({len(batch)} chunks) ---")

            processed, created, errors = rechunk_batch(
                conn=conn,
                batch=batch,
                chunker=chunker,
                embedding_service=embedding_service,
                client_id=args.client_id,
                dry_run=args.dry_run,
            )
            total_processed += processed
            total_created += created
            total_errors += errors

            elapsed = time.time() - start_time
            logger.info(
                f"  Batch done: {processed} processed, {created} new chunks, {errors} errors "
                f"({elapsed:.0f}s elapsed)"
            )

        # Summary
        elapsed = time.time() - start_time
        logger.info(f"\n{'='*60}")
        logger.info(f"{'DRY RUN ' if args.dry_run else ''}COMPLETE")
        logger.info(f"  Oversized chunks found:  {len(oversized)}")
        logger.info(f"  Successfully processed:  {total_processed}")
        logger.info(f"  New chunks created:      {total_created}")
        logger.info(f"  Errors:                  {total_errors}")
        logger.info(f"  Time:                    {elapsed:.1f}s")
        logger.info(f"{'='*60}")

        if not args.dry_run and total_errors == 0:
            # Verify
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM document_chunks WHERE client_id = %s::uuid AND LENGTH(content) > %s",
                    (args.client_id, args.threshold),
                )
                remaining = cur.fetchone()[0]
            logger.info(f"  Remaining oversized chunks: {remaining}")

    finally:
        conn.close()
        logger.info("Database connection closed")


if __name__ == "__main__":
    main()
