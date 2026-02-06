
import logging
from execution.legal_rag.vector_store import VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def deduplicate_documents():
    store = VectorStore()
    store.connect()
    
    DEMO_CLIENT_ID = "00000000-0000-0000-0000-000000000001"
    
    # 1. Identify duplicates: groups of docs with same title, keep the NEWEST one
    # We use created_at to determine newest
    
    find_duplicates_query = """
    WITH ordered_docs AS (
        SELECT 
            id,
            title,
            created_at,
            ROW_NUMBER() OVER (PARTITION BY title ORDER BY created_at DESC) as rn
        FROM legal_documents
        WHERE client_id = %s
    )
    SELECT id, title, created_at 
    FROM ordered_docs 
    WHERE rn > 1;
    """
    
    try:
        with store.get_connection() as conn:
            with conn.cursor() as cur:
                # Find duplicates
                cur.execute(find_duplicates_query, (DEMO_CLIENT_ID,))
                duplicates = cur.fetchall()
                
                if not duplicates:
                    logger.info("No duplicates found!")
                    return
                    
                logger.info(f"Found {len(duplicates)} duplicate/old documents to remove.")
                
                # Delete duplicates (cascades to chunks)
                deleted_count = 0
                for doc in duplicates:
                    doc_id = doc[0]
                    title = doc[1]
                    logger.info(f"Deleting old version of: {title} (ID: {doc_id})")
                    
                    cur.execute("DELETE FROM legal_documents WHERE id = %s", (doc_id,))
                    deleted_count += 1
                
                conn.commit()
                logger.info(f"Successfully removed {deleted_count} old documents.")
                
    except Exception as e:
        logger.error(f"Error during deduplication: {e}")
    finally:
        store.close()

if __name__ == "__main__":
    deduplicate_documents()
