"""
Legal RAG Demo Application

Streamlit-based demo for the Legal Document Intelligence system.
Upload PDFs, ask questions, get answers with citations.

Run with: streamlit run demo_app.py
"""

import os
import sys
import json
import logging
import tempfile
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
from dotenv import load_dotenv

# Load environment variables from project root
project_root = Path(__file__).parent.parent.parent
load_dotenv(project_root / ".env")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Demo mode client ID for fallback (when auth is disabled)
DEMO_CLIENT_ID = "00000000-0000-0000-0000-000000000001"

# Authentication mode: set to False to skip login for development
AUTH_ENABLED = os.getenv("AUTH_ENABLED", "false").lower() == "true"

# Page config
st.set_page_config(
    page_title="Legal Document Intelligence",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def get_current_client_id() -> str:
    """Get the current authenticated client ID."""
    if AUTH_ENABLED:
        return st.session_state.get("client_id", DEMO_CLIENT_ID)
    return DEMO_CLIENT_ID


def render_login_page():
    """Render the login page for authentication."""
    st.title("⚖️ Legal Document Intelligence")
    st.markdown("### Secure Login")
    st.markdown("---")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("Enter your API key to access the system.")

        api_key = st.text_input(
            "API Key",
            type="password",
            placeholder="lrag_xxxxxxxxxxxxx",
            help="Contact your administrator if you don't have an API key"
        )

        if st.button("Login", type="primary", use_container_width=True):
            if not api_key:
                st.error("Please enter an API key")
                return

            # Validate the API key
            try:
                from execution.legal_rag.vector_store import VectorStore

                store = VectorStore()
                store.connect()
                store.initialize_auth_schema()  # Ensure tables exist

                auth_result = store.validate_api_key(api_key)

                if auth_result:
                    st.session_state.authenticated = True
                    st.session_state.client_id = auth_result["client_id"]
                    st.session_state.client_tier = auth_result["tier"]
                    st.session_state.client_name = auth_result["name"]

                    # Log the login
                    store.log_audit(
                        client_id=auth_result["client_id"],
                        action="login",
                        details={"key_name": auth_result["name"]}
                    )

                    store.close()
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid API key. Please check and try again.")
                    store.close()

            except Exception as e:
                logger.error(f"Login error: {e}")
                st.error(f"Login failed: {str(e)}")

        st.markdown("---")
        st.markdown(
            "<small>For demo access, use the demo key or contact support.</small>",
            unsafe_allow_html=True
        )


def render_logout_button():
    """Render logout button in sidebar."""
    if AUTH_ENABLED and st.session_state.get("authenticated"):
        st.sidebar.markdown("---")
        col1, col2 = st.sidebar.columns([2, 1])
        with col1:
            st.write(f"👤 {st.session_state.get('client_name', 'User')}")
            st.caption(f"Tier: {st.session_state.get('client_tier', 'default')}")
        with col2:
            if st.button("Logout"):
                # Clear session
                for key in ["authenticated", "client_id", "client_tier", "client_name",
                           "initialized", "documents", "chat_history"]:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()


def init_services():
    """Initialize RAG services."""
    if "initialized" in st.session_state:
        return True

    try:
        from execution.legal_rag.document_parser import LegalDocumentParser
        from execution.legal_rag.chunker import LegalChunker
        from execution.legal_rag.embeddings import get_embedding_service
        from execution.legal_rag.vector_store import VectorStore
        from execution.legal_rag.retriever import HybridRetriever
        from execution.legal_rag.citation import CitationExtractor

        st.session_state.parser = LegalDocumentParser()
        st.session_state.chunker = LegalChunker()
        st.session_state.embeddings = get_embedding_service(provider="voyage")  # voyage-law-2
        st.session_state.store = VectorStore()
        st.session_state.store.connect()
        st.session_state.store.initialize_schema()
        st.session_state.retriever = HybridRetriever(
            st.session_state.store,
            st.session_state.embeddings,
        )
        st.session_state.citation_extractor = CitationExtractor()
        st.session_state.initialized = True
        st.session_state.documents = {}
        st.session_state.chat_history = []

        # Get the current client ID (from auth or demo mode)
        client_id = get_current_client_id()

        # Set tenant context for Row-Level Security (if RLS is enabled)
        try:
            st.session_state.store.set_tenant_context(client_id)
        except Exception as e:
            logger.debug(f"Tenant context not set (RLS may not be enabled): {e}")

        # Load existing documents from database for persistence (filtered by client)
        try:
            existing_docs = st.session_state.store.list_documents(client_id=client_id)
            for doc in existing_docs:
                doc_id = str(doc["id"])
                st.session_state.documents[doc_id] = {
                    "id": doc_id,
                    "title": doc["title"],
                    "type": doc["document_type"],
                    "jurisdiction": doc.get("jurisdiction"),
                    "pages": doc["page_count"],
                    "chunks": None,  # Unknown for existing docs
                    "uploaded_at": str(doc["created_at"]),
                }
                st.session_state.citation_extractor._document_titles[doc_id] = doc["title"]
            logger.info(f"Loaded {len(existing_docs)} existing documents from database")
        except Exception as e:
            logger.warning(f"Could not load existing documents: {e}")

        return True

    except Exception as e:
        st.error(f"Failed to initialize services: {e}")
        logger.exception("Initialization error")
        return False


def process_document(uploaded_file) -> dict:
    """Process an uploaded PDF document."""
    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        # Parse document
        with st.spinner("Parsing document structure..."):
            parsed = st.session_state.parser.parse(tmp_path)

        # Chunk document
        with st.spinner("Creating semantic chunks..."):
            chunks = st.session_state.chunker.chunk(parsed)

        # Add contextual prefixes (Anthropic's Contextual Retrieval method)
        # This prepends document context to each chunk for better retrieval
        with st.spinner("Adding document context to chunks..."):
            # Use first 2000 chars as document summary for context generation
            document_summary = parsed.raw_text[:2000] if parsed.raw_text else ""
            if document_summary:
                chunks = st.session_state.chunker.contextualize_chunks(
                    chunks=chunks,
                    document_summary=document_summary,
                )
                contextualized_count = sum(1 for c in chunks if c.contextualized)
                logger.info(f"Contextualized {contextualized_count}/{len(chunks)} chunks")

        # Generate embeddings
        with st.spinner(f"Generating embeddings for {len(chunks)} chunks..."):
            chunk_texts = [c.content for c in chunks]
            embeddings = st.session_state.embeddings.embed_documents(chunk_texts)

        # Store in vector DB
        with st.spinner("Storing in vector database..."):
            # Get current client ID for multi-tenant isolation
            client_id = get_current_client_id()

            # Store document metadata with client_id for multi-tenant isolation
            st.session_state.store.insert_document(
                document_id=parsed.metadata.document_id,
                title=parsed.metadata.title,
                document_type=parsed.metadata.document_type,
                client_id=client_id,
                jurisdiction=parsed.metadata.jurisdiction,
                file_path=tmp_path,
                page_count=parsed.metadata.page_count,
            )

            # Store chunks with embeddings and client_id
            chunk_dicts = [c.to_dict() for c in chunks]
            st.session_state.store.insert_chunks(chunk_dicts, embeddings, client_id=client_id)

            # Log the ingestion action for audit
            st.session_state.store.log_audit(
                client_id=client_id,
                action="ingest",
                resource_type="document",
                resource_id=parsed.metadata.document_id,
                details={"title": parsed.metadata.title, "chunks": len(chunks)}
            )

        # Store document info in session
        doc_info = {
            "id": parsed.metadata.document_id,
            "title": parsed.metadata.title,
            "type": parsed.metadata.document_type,
            "jurisdiction": parsed.metadata.jurisdiction,
            "pages": parsed.metadata.page_count,
            "chunks": len(chunks),
            "uploaded_at": datetime.now().isoformat(),
        }
        st.session_state.documents[parsed.metadata.document_id] = doc_info

        # Update citation extractor with document title
        st.session_state.citation_extractor._document_titles[
            parsed.metadata.document_id
        ] = parsed.metadata.title

        return doc_info

    finally:
        # Clean up temp file
        os.unlink(tmp_path)


def query_documents(query: str, document_id: str = None) -> dict:
    """
    Query documents and get cited response.

    Uses the detailed Llama 3.1 70B model for comprehensive, accurate answers.

    Args:
        query: User's question
        document_id: Optional filter to specific document

    Returns:
        dict with 'answer', 'sources', and 'latency_ms'
    """
    import time
    start_time = time.time()

    # Get current client ID for multi-tenant isolation
    client_id = get_current_client_id()

    # Log the query for audit
    st.session_state.store.log_audit(
        client_id=client_id,
        action="query",
        details={"query": query[:200]}
    )

    # Retrieve relevant chunks with client_id for multi-tenant isolation
    # Using top_k=10 to give enhanced retrieval (query expansion, HyDE) more room
    results = st.session_state.retriever.retrieve(
        query=query,
        client_id=client_id,
        document_id=document_id,
        top_k=10,
    )

    # Filter out document summary chunks (they show "Page N/A")
    results = [r for r in results if r.hierarchy_path != "Document"]

    retrieval_time = (time.time() - start_time) * 1000
    logger.info(f"Retrieved {len(results)} results in {retrieval_time:.0f}ms for query: {query[:50]}...")
    for i, r in enumerate(results[:5]):
        logger.info(f"  {i+1}. {r.section_title[:50]} (score: {r.score:.4f})")

    if not results:
        return {
            "answer": "No relevant information found in the uploaded documents.",
            "sources": [],
            "latency_ms": retrieval_time,
        }

    # Extract citations
    cited_contents = st.session_state.citation_extractor.extract(results)

    # Build context for LLM
    context = "\n\n---\n\n".join([
        f"**[{i+1}]** {cc.citation.short_format()}:\n{cc.content}"
        for i, cc in enumerate(cited_contents)
    ])

    # Generate response using NVIDIA NIM API (OpenAI-compatible)
    try:
        from openai import OpenAI

        client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=os.getenv("NVIDIA_API_KEY"),
        )

        # Use detailed model for comprehensive, accurate answers
        model = "meta/llama-3.1-70b-instruct"
        max_tokens = 1500
        system_prompt = """You are a legal research assistant. Answer the question based ONLY on the provided sources.

REQUIREMENTS:
1. Every claim must cite a source using the format [N] (e.g. [1], [2])
2. Group citations when possible (e.g. [1, 2])
3. If information is not in the sources, say "This information is not in the provided documents"
4. Be precise and accurate - this is for legal work
5. Quote exact language when relevant, using quotation marks
6. Note any limitations or caveats"""

        generation_start = time.time()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"""Based on the following sources, answer this question: {query}

SOURCES:
{context}

Provide a clear, well-cited answer."""
                }
            ],
            max_tokens=max_tokens,
            temperature=0.2,
        )

        answer = response.choices[0].message.content
        generation_time = (time.time() - generation_start) * 1000
        logger.info(f"Generated answer with {model} in {generation_time:.0f}ms")

    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        # Better fallback with formatted answer
        answer = "I found relevant information but couldn't generate a summary. Here are the sources:\n\n"
        for i, cc in enumerate(cited_contents):
            answer += f"**[Source {i+1}]** {cc.citation.short_format()}:\n> {cc.content[:300]}...\n\n"

    # Format sources
    sources = [cc.citation.to_dict() for cc in cited_contents]

    total_time = (time.time() - start_time) * 1000
    logger.info(f"Total query time: {total_time:.0f}ms")

    return {
        "answer": answer,
        "sources": sources,
        "latency_ms": total_time,
    }


def render_sidebar():
    """Render the sidebar with document upload."""
    with st.sidebar:
        st.title("⚖️ Legal RAG")
        st.markdown("---")

        # Document upload
        st.subheader("📄 Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload legal PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload contracts, statutes, case law, or other legal documents",
        )

        if uploaded_files:
            if st.button("Process Documents", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()

                results = []
                for i, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Processing {i+1}/{len(uploaded_files)}: {uploaded_file.name}")
                    doc_info = process_document(uploaded_file)
                    if doc_info:
                        results.append(doc_info)
                    progress_bar.progress((i + 1) / len(uploaded_files))

                if results:
                    st.success(f"✅ Processed {len(results)} document(s)")
                    total_chunks = sum(r['chunks'] for r in results)
                    st.info(f"Created {total_chunks} total searchable chunks")

        st.markdown("---")

        # Document list
        st.subheader("📚 Loaded Documents")
        if st.session_state.get("documents"):
            for doc_id, doc in st.session_state.documents.items():
                with st.expander(doc["title"][:30] + "..."):
                    st.write(f"**Type:** {doc['type']}")
                    st.write(f"**Pages:** {doc['pages']}")
                    st.write(f"**Chunks:** {doc['chunks']}")
                    if doc.get("jurisdiction"):
                        st.write(f"**Jurisdiction:** {doc['jurisdiction']}")
        else:
            st.info("No documents uploaded yet")

        st.markdown("---")

        # Settings
        st.subheader("⚙️ Settings")
        st.session_state.show_sources = st.checkbox(
            "Show source details",
            value=True,
        )


def render_chat():
    """Render the chat interface."""
    st.title("Legal Document Intelligence")
    st.markdown("Ask questions about your legal documents. Get answers with citations.")

    # Check if documents are loaded
    if not st.session_state.get("documents"):
        st.info("👈 Upload a legal document to get started")
        return

    # Chat history
    for idx, msg in enumerate(st.session_state.chat_history):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # Show latency and sources for assistant messages
            if msg["role"] == "assistant":
                latency = msg.get("latency_ms", 0)
                if latency > 0:
                    st.caption(f"Response time: {latency:.0f}ms")

                if msg.get("sources") and st.session_state.show_sources:
                    with st.expander("📚 Sources"):
                        for i, source in enumerate(msg["sources"]):
                            st.markdown(f"**[{i+1}]** {source['long_citation']}")

    # Query input
    query = st.chat_input("Ask a question about your documents...")

    if query:
        # Add user message
        st.session_state.chat_history.append({
            "role": "user",
            "content": query,
        })

        # Display user message
        with st.chat_message("user"):
            st.markdown(query)

        # Get response with detailed model
        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating answer..."):
                result = query_documents(query)

            st.markdown(result["answer"])

            # Show latency
            latency = result.get("latency_ms", 0)
            st.caption(f"Response time: {latency:.0f}ms")

            if result["sources"] and st.session_state.show_sources:
                with st.expander("📚 Sources"):
                    for i, source in enumerate(result["sources"]):
                        st.markdown(f"**[{i+1}]** {source['long_citation']}")

        # Add to history
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
            "latency_ms": latency,
        })


def main():
    """Main application entry point."""
    # Check authentication if enabled
    if AUTH_ENABLED and not st.session_state.get("authenticated"):
        render_login_page()
        return

    # Initialize services
    if not init_services():
        st.error("Failed to initialize. Check your configuration.")
        st.stop()

    # Render UI
    render_sidebar()
    render_logout_button()
    render_chat()


if __name__ == "__main__":
    main()
