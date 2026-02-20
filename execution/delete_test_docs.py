#!/usr/bin/env python3
"""Delete test documents (animal food case) from the production database.

Usage:
    python -m execution.delete_test_docs --dry-run   # Preview what would be deleted
    python -m execution.delete_test_docs --confirm    # Actually delete
"""

import os
import sys
import argparse
from dotenv import load_dotenv

load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Delete test documents from the database")
    parser.add_argument("--dry-run", action="store_true", help="Preview without deleting")
    parser.add_argument("--confirm", action="store_true", help="Actually delete")
    args = parser.parse_args()

    if not args.dry_run and not args.confirm:
        print("Usage: specify --dry-run to preview or --confirm to delete")
        sys.exit(1)

    # Connect to production DB
    db_url = os.getenv("POSTGRES_URL") or os.getenv("DATABASE_URL")
    if not db_url:
        print("ERROR: No POSTGRES_URL or DATABASE_URL found in environment")
        sys.exit(1)

    import psycopg2
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()

    # Check which columns exist
    cur.execute("""
        SELECT column_name FROM information_schema.columns
        WHERE table_name = 'legal_documents'
        ORDER BY ordinal_position
    """)
    columns = [r[0] for r in cur.fetchall()]
    has_source_origin = "source_origin" in columns
    print(f"Table columns: {', '.join(columns)}")
    print(f"Has source_origin: {has_source_origin}\n")

    # Search for documents with "animal" or "food" in the title
    select_cols = "id, title, client_id, created_at"
    if has_source_origin:
        select_cols = "id, title, source_origin, client_id, created_at"

    # Test documents were uploaded with demo client_id (not the HUDOC/EUR-Lex client)
    DEMO_CLIENT = "00000000-0000-0000-0000-000000000001"
    cur.execute(f"""
        SELECT {select_cols}
        FROM legal_documents
        WHERE client_id = %s::uuid
        ORDER BY created_at
    """, (DEMO_CLIENT,))
    rows = cur.fetchall()

    if not rows:
        print("No matching documents found with animal/food keywords.")
        # Show all docs to help identify the test ones
        cur.execute(f"""
            SELECT {select_cols}
            FROM legal_documents
            ORDER BY created_at DESC
            LIMIT 60
        """)
        alt_rows = cur.fetchall()
        if alt_rows:
            print(f"\nShowing last {len(alt_rows)} documents to help identify test docs:")
            for row in alt_rows:
                if has_source_origin:
                    print(f"  {row[0]} | {row[1][:60]} | origin={row[2]} | created={row[4]}")
                else:
                    print(f"  {row[0]} | {row[1][:60]} | client={row[2]} | created={row[3]}")
        cur.close()
        conn.close()
        return

    print(f"Found {len(rows)} matching documents:\n")
    for row in rows:
        if has_source_origin:
            doc_id, title, origin, client_id, created = row
            print(f"  {doc_id} | {title[:60]} | origin={origin} | created={created}")
        else:
            doc_id, title, client_id, created = row
            print(f"  {doc_id} | {title[:60]} | client={client_id} | created={created}")

    if args.dry_run:
        print(f"\n[DRY RUN] Would delete {len(rows)} documents and their chunks.")
        # Count chunks
        doc_ids = [r[0] for r in rows]
        placeholders = ",".join(["%s"] * len(doc_ids))
        cur.execute(f"SELECT COUNT(*) FROM document_chunks WHERE document_id IN ({placeholders})", doc_ids)
        chunk_count = cur.fetchone()[0]
        print(f"[DRY RUN] Would also delete {chunk_count} associated chunks.")
    elif args.confirm:
        doc_ids = [r[0] for r in rows]
        # Delete chunks first (FK), then documents
        placeholders = ",".join(["%s"] * len(doc_ids))
        cur.execute(f"DELETE FROM document_chunks WHERE document_id IN ({placeholders})", doc_ids)
        chunks_deleted = cur.rowcount
        cur.execute(f"DELETE FROM legal_documents WHERE id IN ({placeholders})", doc_ids)
        docs_deleted = cur.rowcount
        conn.commit()
        print(f"\nDeleted {docs_deleted} documents and {chunks_deleted} chunks.")

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
