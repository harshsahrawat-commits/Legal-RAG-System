#!/usr/bin/env python3
"""
Migrate existing document files to the document storage directory.

Matches files in a source folder to database document records by filename stem,
copies them to DOCUMENT_STORAGE_DIR with the correct naming convention
({document_id}.ext), and updates document metadata with the source stem for
CyLaw URL generation.

Usage:
    # Dry run (default) — shows what would happen
    python migrate_document_files.py ./test_files_cy

    # Execute — actually copies files and updates DB
    python migrate_document_files.py ./test_files_cy --execute

    # Custom storage directory
    python migrate_document_files.py ./test_files_cy --execute --storage-dir ./document_files
"""

import os
import sys
import json
import shutil
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Load .env before anything else
from dotenv import load_dotenv
load_dotenv()

import psycopg2
import psycopg2.extras


# =============================================================================
# Logging setup — dual output: console + file
# =============================================================================

def setup_logging() -> logging.Logger:
    log = logging.getLogger("migration")
    log.setLevel(logging.DEBUG)

    # Console handler (INFO+)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(ch)

    # File handler (DEBUG+)
    fh = logging.FileHandler("migration_log.txt", mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s"))
    log.addHandler(fh)

    return log


logger = setup_logging()


# =============================================================================
# File type detection
# =============================================================================

def detect_file_type(file_path: Path) -> str:
    """Detect actual file type by reading magic bytes. Returns 'html' or 'pdf'."""
    try:
        with open(file_path, "rb") as f:
            header = f.read(256)
    except OSError:
        return "html"  # Default

    if header.startswith(b"%PDF-"):
        return "pdf"
    return "html"  # CyLaw docs are predominantly HTML


# =============================================================================
# Database helpers
# =============================================================================

def get_connection():
    """Create a database connection using the DATABASE_URL env var."""
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        logger.error("DATABASE_URL environment variable is not set.")
        sys.exit(1)
    return psycopg2.connect(db_url, cursor_factory=psycopg2.extras.RealDictCursor)


def get_all_documents(conn) -> list[dict]:
    """Fetch all document records from the database."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, title, file_path, metadata FROM legal_documents ORDER BY created_at"
        )
        rows = cur.fetchall()
    return [dict(r) for r in rows]


def update_document_file_path(conn, document_id: str, new_file_path: str, source_stem: str):
    """Update a document's file_path and add source_stem to metadata."""
    with conn.cursor() as cur:
        # Merge source_stem into existing metadata
        cur.execute(
            """
            UPDATE legal_documents
            SET file_path = %s,
                metadata = COALESCE(metadata, '{}'::jsonb) || %s::jsonb
            WHERE id = %s::uuid
            """,
            (new_file_path, json.dumps({"source_stem": source_stem}), document_id),
        )


# =============================================================================
# Source folder scanning
# =============================================================================

def scan_source_folder(folder_path: str) -> dict[str, Path]:
    """Scan source folder for all document files.

    Returns: {base_stem: file_path} dict.
    When both a regular and _EKS version exist for the same law identifier,
    the _EKS version is preferred.
    """
    files: dict[str, Path] = {}
    eks_stems: set[str] = set()  # Track which stems had EKS upgrades

    folder = Path(folder_path)
    if not folder.is_dir():
        logger.error(f"Source folder does not exist: {folder_path}")
        sys.exit(1)

    for f in sorted(folder.rglob("*")):
        if not f.is_file() or f.name.startswith("."):
            continue

        stem = f.stem  # filename without extension
        base_stem = stem.replace("_EKS", "")

        if "_EKS" in stem:
            files[base_stem] = f  # Always prefer EKS version
            eks_stems.add(base_stem)
        elif base_stem not in files:
            files[base_stem] = f  # Only add if no EKS version exists yet

    return files, eks_stems


# =============================================================================
# Matching logic
# =============================================================================

def extract_stem_from_db_path(file_path: str | None) -> str | None:
    """Extract the original filename stem from a database file_path value."""
    if not file_path:
        return None
    p = Path(file_path)
    stem = p.stem
    # If stored as document_files/{uuid}.pdf, this won't be a match stem
    # Check if it looks like a UUID (36 chars with dashes)
    if len(stem) == 36 and stem.count("-") == 4:
        return None
    return stem


def stem_from_title(title: str) -> str | None:
    """Try to derive a filename stem from a document title.

    CyLaw document titles often contain the law identifier like:
    'Ο περί ... Νόμος του 2020 (123(I)/2020)' → might map to 2020_1_123
    This is a best-effort heuristic.
    """
    if not title:
        return None

    import re

    # Pattern: N(I)/YYYY or N(I) of YYYY → YYYY_1_NNN
    m = re.search(r"(\d+)\s*\(I\)\s*/?\s*(\d{4})", title)
    if m:
        number = m.group(1)
        year = m.group(2)
        return f"{year}_1_{number.zfill(3)}"

    return None


def match_documents(db_docs: list[dict], source_files: dict[str, Path]) -> list[tuple]:
    """Match source files to database documents.

    Returns list of (source_path, document_id, document_title, match_method, stem).
    """
    matches = []
    matched_doc_ids = set()
    matched_stems = set()

    # Build reverse index: stem → document
    stem_to_doc: dict[str, dict] = {}

    for doc in db_docs:
        doc_id = str(doc["id"])

        # Method 1: Extract stem from file_path
        db_stem = extract_stem_from_db_path(doc.get("file_path"))
        if db_stem:
            base = db_stem.replace("_EKS", "")
            if base in source_files:
                stem_to_doc[base] = doc
                continue

        # Method 2: Check metadata for source_stem (already migrated)
        meta = doc.get("metadata") or {}
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except (json.JSONDecodeError, TypeError):
                meta = {}
        existing_stem = meta.get("source_stem")
        if existing_stem:
            base = existing_stem.replace("_EKS", "")
            if base in source_files:
                stem_to_doc[base] = doc
                continue

        # Method 3: Derive stem from title
        title_stem = stem_from_title(doc.get("title", ""))
        if title_stem:
            base = title_stem.replace("_EKS", "")
            if base in source_files:
                stem_to_doc[base] = doc
                continue

    # Build final match list
    for stem, doc in stem_to_doc.items():
        source_path = source_files[stem]
        doc_id = str(doc["id"])
        if doc_id not in matched_doc_ids:
            # Determine the actual stem used (could be EKS)
            actual_stem = source_path.stem
            match_method = "file_path" if extract_stem_from_db_path(doc.get("file_path")) else "title"
            matches.append((source_path, doc_id, doc.get("title", ""), match_method, actual_stem))
            matched_doc_ids.add(doc_id)
            matched_stems.add(stem)

    return matches, matched_doc_ids, matched_stems


# =============================================================================
# Migration execution
# =============================================================================

def migrate(
    matches: list[tuple],
    storage_dir: Path,
    conn,
    dry_run: bool = True,
) -> dict:
    """Copy matched files to storage directory with correct naming.

    Returns summary statistics.
    """
    storage_dir.mkdir(parents=True, exist_ok=True)

    stats = {"copied": 0, "skipped_existing": 0, "errors": 0, "db_updated": 0}
    total = len(matches)
    db_updates = []  # Collect all updates, batch at the end

    for i, (source_path, doc_id, title, method, stem) in enumerate(matches, 1):
        # Progress reporting
        if i % 500 == 0 or i == total:
            logger.info(f"  Progress: {i}/{total} ({i*100//total}%)")

        # Detect actual file type
        file_type = detect_file_type(source_path)
        ext = f".{file_type}"
        dest_path = storage_dir / f"{doc_id}{ext}"

        if dry_run:
            eks_note = " [EKS preferred]" if "_EKS" in stem else ""
            logger.info(f"MATCHED: {source_path.name} -> {doc_id[:12]}... ({title[:60]}){eks_note}")
            stats["copied"] += 1
            continue

        # Skip if destination already exists (resumable)
        if dest_path.exists():
            stats["skipped_existing"] += 1
            continue

        # Also check for other extensions (might already be migrated with different ext)
        existing = list(storage_dir.glob(f"{doc_id}.*"))
        if existing:
            stats["skipped_existing"] += 1
            continue

        # Copy file
        try:
            shutil.copy2(str(source_path), str(dest_path))
            stats["copied"] += 1
            db_updates.append((str(dest_path), json.dumps({"source_stem": stem}), doc_id))
        except Exception as e:
            logger.error(f"ERROR copying {source_path.name}: {e}")
            stats["errors"] += 1

    # Batch all DB updates in one round-trip
    if not dry_run and db_updates:
        try:
            logger.info(f"Sending {len(db_updates)} DB updates in batch...")
            with conn.cursor() as cur:
                psycopg2.extras.execute_batch(
                    cur,
                    """
                    UPDATE legal_documents
                    SET file_path = %s,
                        metadata = COALESCE(metadata, '{}'::jsonb) || %s::jsonb
                    WHERE id = %s::uuid
                    """,
                    db_updates,
                    page_size=500,
                )
            conn.commit()
            stats["db_updated"] = len(db_updates)
            logger.info(f"Database committed: {stats['db_updated']} records updated")
        except Exception as e:
            logger.error(f"Database batch update failed: {e}")
            conn.rollback()

    return stats


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Migrate existing document files to document storage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python migrate_document_files.py ./test_files_cy              # Dry run
  python migrate_document_files.py ./test_files_cy --execute    # Actually copy
  python migrate_document_files.py ./test_files_cy --execute --storage-dir ./docs
        """,
    )
    parser.add_argument("source_folder", help="Path to folder containing original document files")
    parser.add_argument("--execute", action="store_true", help="Actually copy files (default is dry-run)")
    parser.add_argument("--storage-dir", default="document_files", help="Target storage directory")
    args = parser.parse_args()

    dry_run = not args.execute
    storage_dir = Path(args.storage_dir)

    logger.info("=" * 60)
    logger.info(f"{'MIGRATION DRY RUN' if dry_run else 'MIGRATION EXECUTE'}")
    logger.info(f"Started: {datetime.now().isoformat()}")
    logger.info("=" * 60)

    # Scan source folder
    logger.info(f"\nScanning source folder: {args.source_folder}")
    source_files, eks_stems = scan_source_folder(args.source_folder)
    logger.info(f"Found {len(source_files)} unique file stems ({len(eks_stems)} EKS upgrades)")

    # Connect to database
    logger.info("\nConnecting to database...")
    conn = get_connection()
    logger.info("Connected.")

    # Fetch all documents
    db_docs = get_all_documents(conn)
    logger.info(f"Database: {len(db_docs)} documents\n")

    # Match
    matches, matched_doc_ids, matched_stems = match_documents(db_docs, source_files)

    # Identify unmatched
    unmatched_files = sorted(set(source_files.keys()) - matched_stems)
    unmatched_docs = [d for d in db_docs if str(d["id"]) not in matched_doc_ids]

    # Report matches
    logger.info(f"\n{'=' * 60}")
    logger.info("MATCH RESULTS")
    logger.info(f"{'=' * 60}")

    # Execute or dry-run
    stats = migrate(matches, storage_dir, conn, dry_run=dry_run)

    # Report unmatched files (first 20)
    if unmatched_files:
        logger.info(f"\nUNMATCHED FILES ({len(unmatched_files)} total):")
        for stem in unmatched_files[:20]:
            logger.info(f"  {stem}")
        if len(unmatched_files) > 20:
            logger.info(f"  ... and {len(unmatched_files) - 20} more")

    # Report unmatched DB records (first 20)
    if unmatched_docs:
        logger.info(f"\nUNMATCHED DB RECORDS ({len(unmatched_docs)} total):")
        for doc in unmatched_docs[:20]:
            logger.info(f"  {str(doc['id'])[:12]}... - {doc.get('title', 'N/A')[:60]}")
        if len(unmatched_docs) > 20:
            logger.info(f"  ... and {len(unmatched_docs) - 20} more")

    # Summary
    logger.info(f"\n{'=' * 60}")
    logger.info("SUMMARY")
    logger.info(f"{'=' * 60}")
    logger.info(f"Matched:           {len(matches)} / {len(db_docs)} DB records "
                f"({len(matches)*100//max(len(db_docs),1)}%)")
    logger.info(f"EKS upgrades:      {len(eks_stems)}")
    logger.info(f"Unmatched files:   {len(unmatched_files)}")
    logger.info(f"Unmatched DB docs: {len(unmatched_docs)}")

    if dry_run:
        logger.info(f"\nFiles to copy:     {stats['copied']}")
        logger.info(f"\nThis was a DRY RUN. Re-run with --execute to copy files.")
    else:
        logger.info(f"\nFiles copied:      {stats['copied']}")
        logger.info(f"Skipped (exist):   {stats['skipped_existing']}")
        logger.info(f"DB updated:        {stats['db_updated']}")
        logger.info(f"Errors:            {stats['errors']}")

    conn.close()
    logger.info(f"\nDone. Log saved to migration_log.txt")


if __name__ == "__main__":
    main()
