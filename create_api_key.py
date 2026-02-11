#!/usr/bin/env python3
"""Create an API key for the Legal RAG system."""

import os
import sys

# Load .env
from dotenv import load_dotenv
load_dotenv()

# Add execution dir to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "execution"))

from legal_rag.vector_store import VectorStore

# Demo tenant client ID (from CLAUDE.md)
DEMO_CLIENT_ID = "00000000-0000-0000-0000-000000000001"

def main():
    store = VectorStore()
    store.connect()

    # Ensure auth schema exists
    try:
        store.initialize_auth_schema()
        print("Auth schema initialized.")
    except Exception as e:
        print(f"Auth schema note: {e}")

    # Create the API key
    raw_key = store.create_api_key(
        client_id=DEMO_CLIENT_ID,
        name="dev-local",
        tier="default",
    )

    print("\n" + "=" * 50)
    print("API Key created successfully!")
    print("=" * 50)
    print(f"\n  {raw_key}\n")
    print("Save this key — it cannot be retrieved again.")
    print("=" * 50)

if __name__ == "__main__":
    main()
