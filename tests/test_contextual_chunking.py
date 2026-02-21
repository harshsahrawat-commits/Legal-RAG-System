#!/usr/bin/env python3
"""
Test script for contextual chunking (Anthropic's method).
Tests the contextualize_chunks() method with sample content.
"""

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(name)s - %(message)s')
logger = logging.getLogger(__name__)

# Sample legal document content
SAMPLE_DOCUMENT = """
UNITED STATES DISTRICT COURT
WESTERN DISTRICT OF CALIFORNIA

IN RE: PET FOOD PRODUCTS LIABILITY LITIGATION
MDL No. 1850

CLASS ACTION COMPLAINT

INTRODUCTION

1. Plaintiffs bring this class action complaint against Defendants Menu Foods Inc.
and related parties for manufacturing and selling contaminated pet food products
that caused illness and death to thousands of household pets.

2. Between March 2007 and April 2007, numerous brands of pet food were recalled
after reports of kidney failure and death in cats and dogs across North America.

FACTUAL BACKGROUND

3. Defendants Menu Foods Inc. is a Canadian company that manufactures pet food
products for various major brands including Iams, Eukanuba, and Hills Science Diet.

4. The contamination was caused by melamine, an industrial chemical, that was
illegally added to wheat gluten imported from China.

CAUSES OF ACTION

FIRST CAUSE OF ACTION
Breach of Implied Warranty of Merchantability

5. Defendants impliedly warranted that their pet food products were safe for
consumption by household pets.

6. The products were not safe and caused severe kidney damage and death.

SECOND CAUSE OF ACTION
Negligence

7. Defendants had a duty to exercise reasonable care in manufacturing their products.

8. Defendants breached this duty by failing to properly test ingredients.
"""


def test_contextualization():
    """Test the contextualize_chunks method."""
    from execution.legal_rag.chunker import LegalChunker, Chunk
    import uuid

    print("\n" + "="*70)
    print("TESTING CONTEXTUAL CHUNKING")
    print("="*70)

    # Check NVIDIA API key
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        print("❌ NVIDIA_API_KEY not set - cannot test contextualization")
        return False

    print(f"✅ NVIDIA_API_KEY found")

    # Create sample chunks manually
    chunker = LegalChunker()
    doc_id = str(uuid.uuid4())

    sample_chunks = [
        Chunk(
            chunk_id=str(uuid.uuid4()),
            document_id=doc_id,
            content="Plaintiffs bring this class action complaint against Defendants Menu Foods Inc. and related parties for manufacturing and selling contaminated pet food products that caused illness and death to thousands of household pets.",
            token_count=40,
            level=2,
            section_title="INTRODUCTION",
            hierarchy_path="Document/Introduction",
        ),
        Chunk(
            chunk_id=str(uuid.uuid4()),
            document_id=doc_id,
            content="The contamination was caused by melamine, an industrial chemical, that was illegally added to wheat gluten imported from China.",
            token_count=25,
            level=2,
            section_title="FACTUAL BACKGROUND",
            hierarchy_path="Document/Factual_Background",
        ),
        Chunk(
            chunk_id=str(uuid.uuid4()),
            document_id=doc_id,
            content="Defendants impliedly warranted that their pet food products were safe for consumption by household pets. The products were not safe and caused severe kidney damage and death.",
            token_count=35,
            level=2,
            section_title="FIRST CAUSE OF ACTION - Breach of Implied Warranty",
            hierarchy_path="Document/Causes_of_Action/First",
        ),
    ]

    print(f"\n📄 Created {len(sample_chunks)} sample chunks")
    print(f"   Document summary: {len(SAMPLE_DOCUMENT)} chars")

    # Test contextualization
    print("\n🔄 Calling contextualize_chunks()...")
    try:
        contextualized = chunker.contextualize_chunks(
            chunks=sample_chunks,
            document_summary=SAMPLE_DOCUMENT,
        )

        # Check results
        success_count = sum(1 for c in contextualized if c.contextualized)
        print(f"\n✅ Contextualized {success_count}/{len(contextualized)} chunks")

        # Show examples
        print("\n" + "-"*70)
        print("SAMPLE CONTEXTUALIZED CHUNKS:")
        print("-"*70)

        for i, chunk in enumerate(contextualized):
            if chunk.contextualized:
                print(f"\n📌 Chunk {i+1}: {chunk.section_title}")
                print(f"   Context prefix:")
                print(f"   \"{chunk.context_prefix}\"")
                print(f"\n   Original content (first 100 chars):")
                # Content now includes prefix, so show the part after it
                original_start = len(chunk.context_prefix) + 2 if chunk.context_prefix else 0
                print(f"   \"{chunk.content[original_start:original_start+100]}...\"")

        return success_count > 0

    except Exception as e:
        print(f"❌ Contextualization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_contextualization()
    print("\n" + "="*70)
    if success:
        print("✅ CONTEXTUAL CHUNKING TEST PASSED")
    else:
        print("❌ CONTEXTUAL CHUNKING TEST FAILED")
    print("="*70)
