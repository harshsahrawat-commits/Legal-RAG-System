"""
Create a Cyprus tenant with Greek language configuration.

Sets up:
- New client UUID for Cyprus
- Greek language config (voyage-multilingual-2 embeddings, Greek FTS)
- Greek FTS index in PostgreSQL
- API key for the tenant

Usage:
    python create_cyprus_tenant.py
"""

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CYPRUS_CLIENT_ID = "00000000-0000-0000-0000-000000000002"


def main():
    from execution.legal_rag.vector_store import VectorStore
    from execution.legal_rag.language_config import TenantLanguageConfig

    store = VectorStore()
    store.connect()

    # Ensure schemas exist
    store.initialize_schema()
    store.initialize_auth_schema()
    store.initialize_tenant_config_schema()

    # Set Greek language config for Cyprus tenant
    greek_config = TenantLanguageConfig.for_language("el")
    store.set_tenant_config(CYPRUS_CLIENT_ID, greek_config)
    logger.info(f"Tenant config set: client_id={CYPRUS_CLIENT_ID}, language=el")
    logger.info(f"  Embedding model: {greek_config.embedding_model}")
    logger.info(f"  FTS language: {greek_config.fts_language}")
    logger.info(f"  Reranker: {greek_config.reranker_model}")

    # Create Greek FTS index
    try:
        store.create_greek_fts_index()
        logger.info("Greek FTS index created")
    except Exception as e:
        logger.warning(f"Greek FTS index creation: {e}")

    # Create API key
    api_key = store.create_api_key(CYPRUS_CLIENT_ID, name="Cyprus Demo Key")

    print("\n" + "=" * 60)
    print("CYPRUS TENANT CREATED SUCCESSFULLY")
    print("=" * 60)
    print(f"Client ID:  {CYPRUS_CLIENT_ID}")
    print(f"Language:   Greek (el)")
    print(f"Embeddings: {greek_config.embedding_model}")
    print(f"FTS:        {greek_config.fts_language}")
    print(f"\nAPI Key (save this — it won't be shown again!):")
    print(f"  {api_key}")
    print("=" * 60)

    store.close()


if __name__ == "__main__":
    main()
