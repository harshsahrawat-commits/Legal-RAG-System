import os
import logging
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from execution.legal_rag.vector_store import VectorStore, VectorStoreConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def migrate_to_cloud(connection_string: str):
    """Initialize schema and RLS on the cloud database."""
    logger.info("Initializing cloud database migration...")
    
    # Configure VectorStore with the cloud connection string
    # We disable pooling for the migration script for simplicity
    config = VectorStoreConfig(
        connection_string=connection_string,
        use_pooling=False
    )
    
    store = VectorStore(config=config)
    
    try:
        # 1. Connect and ensure pgvector
        store.connect()
        logger.info("✅ Connected to cloud database")
        
        # 2. Initialize tables
        logger.info("Setting up tables...")
        store.initialize_schema()
        logger.info("✅ Tables initialized successfully")
        
        # 3. Initialize auth tables (API keys)
        logger.info("Setting up auth schema...")
        store.initialize_auth_schema()
        logger.info("✅ Auth schema initialized successfully")
        
        # 4. Enable Row-Level Security
        logger.info("Enabling Row-Level Security...")
        # Note: We don't have a specific 'enable_rls' method in the current VectorStore
        # but initialize_schema and initialize_auth_schema do the heavy lifting.
        # However, let's explicitly run the RLS setup SQL if available.
        # Looking at vector_store.py, it has an enable_rls method.
        store.enable_rls()
        logger.info("✅ RLS policies enabled")
        
        # 5. Create a demo API key
        # This is optional but helpful for the first login
        with store.get_connection() as conn:
            with conn.cursor() as cur:
                # Check if demo API key exists
                cur.execute("SELECT id FROM api_keys WHERE client_id = '00000000-0000-0000-0000-000000000001'")
                if not cur.fetchone():
                    logger.info("Creating demo API key...")
                    import hashlib
                    import secrets
                    demo_key = f"lrag_{secrets.token_urlsafe(32)}"
                    key_hash = hashlib.sha256(demo_key.encode()).hexdigest()

                    cur.execute("""
                        INSERT INTO api_keys (client_id, key_hash, name, tier)
                        VALUES ('00000000-0000-0000-0000-000000000001', %s, 'Default Demo Key', 'demo')
                    """, (key_hash,))
                    conn.commit()
                    logger.info(f"Demo API key created (save this, shown only once): {demo_key}")
        
        logger.info("🚀 Cloud database is ready for Streamlit deployment!")
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        raise
    finally:
        store.close()

if __name__ == "__main__":
    load_dotenv()
    # Get connection string from CLI argument or environment variable.
    # NEVER hardcode database credentials in source code.
    if len(sys.argv) > 1:
        cloud_url = sys.argv[1]
    else:
        cloud_url = os.getenv("NEON_DATABASE_URL") or os.getenv("DATABASE_URL")

    if not cloud_url:
        logger.error(
            "No connection string provided. Pass it as a CLI argument or set "
            "NEON_DATABASE_URL / DATABASE_URL in your .env file."
        )
        sys.exit(1)

    migrate_to_cloud(cloud_url)
