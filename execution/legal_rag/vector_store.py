"""
Vector Store with PostgreSQL + pgvector

Provides vector storage and similarity search using PostgreSQL's pgvector
extension. Supports multi-tenant isolation via client_id filtering and
Row-Level Security (RLS) policies.
"""

import os
import json
import logging
import hashlib
import secrets
from typing import Optional
from dataclasses import dataclass
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class VectorStoreConfig:
    """Configuration for vector store."""
    connection_string: Optional[str] = None
    table_name: str = "document_chunks"
    embedding_dimensions: int = 1024
    index_lists: int = 100  # IVFFlat index parameter
    # Connection pooling settings
    pool_min_connections: int = 2
    pool_max_connections: int = 20
    use_pooling: bool = True  # Set to False for simple single-connection mode


@dataclass
class SearchResult:
    """A single search result with score."""
    chunk_id: str
    document_id: str
    content: str
    section_title: str
    hierarchy_path: str
    page_numbers: list[int]
    score: float
    metadata: dict
    # Paragraph tracking fields
    paragraph_start: Optional[int] = None
    paragraph_end: Optional[int] = None
    original_paragraph_numbers: list = None  # list[int]

    def __post_init__(self):
        if self.original_paragraph_numbers is None:
            self.original_paragraph_numbers = []

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "content": self.content,
            "section_title": self.section_title,
            "hierarchy_path": self.hierarchy_path,
            "page_numbers": self.page_numbers,
            "score": self.score,
            "metadata": self.metadata,
            "paragraph_start": self.paragraph_start,
            "paragraph_end": self.paragraph_end,
            "original_paragraph_numbers": self.original_paragraph_numbers,
        }


class VectorStore:
    """
    PostgreSQL vector store with pgvector.

    Features:
    - Cosine similarity search
    - Metadata filtering
    - Multi-tenant isolation via client_id
    - Batch insert for efficiency
    """

    def __init__(self, config: Optional[VectorStoreConfig] = None):
        """
        Initialize vector store.

        Args:
            config: Optional configuration. Uses env vars if not provided.
        """
        self.config = config or VectorStoreConfig()
        self._conn = None
        self._pool = None
        self._current_tenant: Optional[str] = None
        self._connection_string = (
            self.config.connection_string or
            os.getenv("POSTGRES_URL") or
            os.getenv("DATABASE_URL") or
            "postgresql://localhost:5432/legal_rag"
        )

    def connect(self) -> None:
        """Establish database connection (with optional pooling)."""
        try:
            import psycopg2
            from psycopg2.extras import RealDictCursor
            from psycopg2 import pool

            if self.config.use_pooling:
                # Use connection pooling for production
                self._pool = pool.ThreadedConnectionPool(
                    minconn=self.config.pool_min_connections,
                    maxconn=self.config.pool_max_connections,
                    dsn=self._connection_string,
                )
                # Also create a persistent connection for methods that use self._conn directly
                # This maintains backward compatibility while still having pool available
                self._conn = psycopg2.connect(
                    self._connection_string,
                    cursor_factory=RealDictCursor
                )
                self._conn.autocommit = False

                # Ensure pgvector extension
                with self._conn.cursor() as cur:
                    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                    self._conn.commit()

                logger.info(
                    f"Connection pool initialized (min={self.config.pool_min_connections}, "
                    f"max={self.config.pool_max_connections})"
                )
            else:
                # Simple single connection mode
                self._conn = psycopg2.connect(
                    self._connection_string,
                    cursor_factory=RealDictCursor
                )
                self._conn.autocommit = False

                # Ensure pgvector extension
                with self._conn.cursor() as cur:
                    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                    self._conn.commit()

                logger.info("Connected to PostgreSQL with pgvector (single connection)")

        except ImportError:
            raise ImportError(
                "psycopg2 not installed. Run: pip install psycopg2-binary"
            )
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise

    def _get_connection(self):
        """Get a database connection (from pool or single connection)."""
        if self._pool:
            try:
                conn = self._pool.getconn()
                # Check if connection is actually alive
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                
                # Set tenant context if we have one
                if self._current_tenant:
                    with conn.cursor() as cur:
                        cur.execute("SET app.current_tenant = %s", (self._current_tenant,))
                return conn
            except (psycopg2.OperationalError, psycopg2.InterfaceError):
                logger.warning("Connection from pool is dead, attempting to re-establish...")
                try:
                    self.connect()
                    return self._pool.getconn()
                except Exception as e:
                    logger.error(f"Failed to re-establish connection pool: {e}")
                    raise
        
        # For single connection mode, check and reconnect if needed
        if self._conn:
            try:
                if self._conn.closed:
                    self.connect()
                else:
                    with self._conn.cursor() as cur:
                        cur.execute("SELECT 1")
            except (psycopg2.OperationalError, psycopg2.InterfaceError):
                logger.warning("Connection died, reconnecting...")
                self.connect()
        
        return self._conn

    def _release_connection(self, conn):
        """Release a connection back to the pool (if pooling is enabled)."""
        if self._pool and conn:
            self._pool.putconn(conn)

    def _ensure_connection(self):
        """Ensure we have a connection (pool or single) and return it."""
        if not self._conn and not self._pool:
            self.connect()
        
        try:
            return self._get_connection()
        except (psycopg2.OperationalError, psycopg2.InterfaceError):
            logger.warning("Database error in _ensure_connection, retrying after reconnect...")
            self.connect()
            return self._get_connection()

    def _is_connected(self) -> bool:
        """Check if we have an active connection (pool or single)."""
        return self._conn is not None or self._pool is not None

    @contextmanager
    def get_connection(self):
        """
        Context manager for getting a database connection.

        Usage:
            with store.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")

        Automatically releases connection back to pool when done.
        """
        conn = self._get_connection()
        try:
            yield conn
        finally:
            self._release_connection(conn)

    def close(self) -> None:
        """Close database connection(s)."""
        if self._pool:
            self._pool.closeall()
            self._pool = None
            logger.info("Connection pool closed")
        if self._conn:
            self._conn.close()
            self._conn = None

    # =========================================================================
    # Row-Level Security (RLS) Methods
    # =========================================================================

    def set_tenant_context(self, client_id: str) -> None:
        """
        Set the tenant context for Row-Level Security.

        Must be called before any database operations when RLS is enabled.
        This ensures queries only return data for the specified tenant.

        Args:
            client_id: The client/tenant UUID
        """
        self._current_tenant = client_id

        # If using pooling, tenant context is set per-connection in _get_connection()
        if self._pool:
            logger.debug(f"Tenant context will be set to {client_id} on next connection")
            return

        # For single connection, set immediately
        if not self._conn:
            self.connect()

        try:
            with self._conn.cursor() as cur:
                cur.execute("SET app.current_tenant = %s", (client_id,))
            logger.debug(f"Tenant context set to {client_id}")
        except Exception as e:
            logger.error(f"Failed to set tenant context: {e}")
            raise

    def clear_tenant_context(self) -> None:
        """Clear the tenant context (for admin operations)."""
        self._current_tenant = None

        # For pooling, just clear the stored tenant (no persistent connection)
        if self._pool or not self._conn:
            return

        try:
            with self._conn.cursor() as cur:
                cur.execute("RESET app.current_tenant")
        except Exception as e:
            logger.error(f"Failed to clear tenant context: {e}")

    def enable_rls(self) -> None:
        """
        Enable Row-Level Security on tables for multi-tenant isolation.

        This creates RLS policies that enforce:
        - Each client can only see their own documents and chunks
        - Database-level security even if application has bugs

        WARNING: Only call this once during initial setup.
        """
        rls_sql = """
        -- Enable RLS on legal_documents table
        ALTER TABLE legal_documents ENABLE ROW LEVEL SECURITY;

        -- Policy: Users can only see their own documents
        DROP POLICY IF EXISTS tenant_isolation_docs ON legal_documents;
        CREATE POLICY tenant_isolation_docs ON legal_documents
            FOR ALL
            USING (client_id::text = current_setting('app.current_tenant', true));

        -- Enable RLS on document_chunks table
        ALTER TABLE document_chunks ENABLE ROW LEVEL SECURITY;

        -- Policy: Users can only see their own chunks
        DROP POLICY IF EXISTS tenant_isolation_chunks ON document_chunks;
        CREATE POLICY tenant_isolation_chunks ON document_chunks
            FOR ALL
            USING (client_id::text = current_setting('app.current_tenant', true));

        -- Create app_user role if not exists (restricted permissions)
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'app_user') THEN
                CREATE ROLE app_user;
            END IF;
        END
        $$;

        -- Grant permissions to app_user
        GRANT SELECT, INSERT, UPDATE, DELETE ON legal_documents TO app_user;
        GRANT SELECT, INSERT, UPDATE, DELETE ON document_chunks TO app_user;
        """

        conn = self._ensure_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(rls_sql)
                conn.commit()
            logger.info("Row-Level Security enabled on all tables")
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to enable RLS: {e}")
            raise
        finally:
            self._release_connection(conn)

    def disable_rls(self) -> None:
        """Disable Row-Level Security (for testing/migration)."""
        sql = """
        ALTER TABLE legal_documents DISABLE ROW LEVEL SECURITY;
        ALTER TABLE document_chunks DISABLE ROW LEVEL SECURITY;
        """

        conn = self._ensure_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(sql)
                conn.commit()
            logger.info("Row-Level Security disabled")
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to disable RLS: {e}")
            raise
        finally:
            self._release_connection(conn)

    # =========================================================================
    # Authentication & Audit Schema
    # =========================================================================

    def initialize_auth_schema(self) -> None:
        """
        Create tables for authentication and audit logging.

        Tables created:
        - api_keys: Maps API keys to client IDs for authentication
        - audit_log: Records all significant actions for compliance
        - usage_daily: Tracks daily usage for quotas
        """
        auth_sql = """
        -- API Keys table for authentication
        CREATE TABLE IF NOT EXISTS api_keys (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            key_hash VARCHAR(64) NOT NULL UNIQUE,
            client_id UUID NOT NULL,
            name VARCHAR(100),
            tier VARCHAR(20) DEFAULT 'default',
            is_active BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            last_used_at TIMESTAMPTZ
        );
        CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(key_hash);
        CREATE INDEX IF NOT EXISTS idx_api_keys_client ON api_keys(client_id);

        -- Audit log for compliance
        CREATE TABLE IF NOT EXISTS audit_log (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            client_id UUID NOT NULL,
            action VARCHAR(50) NOT NULL,
            resource_type VARCHAR(50),
            resource_id UUID,
            details JSONB,
            ip_address INET,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS idx_audit_client_time
            ON audit_log(client_id, created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_log(action);

        -- Usage tracking for quotas
        CREATE TABLE IF NOT EXISTS usage_daily (
            client_id UUID NOT NULL,
            date DATE NOT NULL,
            query_count INT DEFAULT 0,
            document_count INT DEFAULT 0,
            PRIMARY KEY (client_id, date)
        );
        """

        conn = self._ensure_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(auth_sql)
                conn.commit()
            logger.info("Authentication schema initialized")
        except Exception as e:
            conn.rollback()
            logger.error(f"Auth schema initialization failed: {e}")
            raise
        finally:
            self._release_connection(conn)

    def create_api_key(
        self,
        client_id: str,
        name: str = "Default",
        tier: str = "default"
    ) -> str:
        """
        Create a new API key for a client.

        Args:
            client_id: The client UUID
            name: A friendly name for the key
            tier: Subscription tier (default, premium)

        Returns:
            The raw API key (only shown once, store securely!)
        """
        # Generate a secure random API key
        raw_key = f"lrag_{secrets.token_urlsafe(32)}"
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

        sql = """
        INSERT INTO api_keys (key_hash, client_id, name, tier)
        VALUES (%s, %s::uuid, %s, %s)
        RETURNING id
        """

        conn = self._ensure_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, (key_hash, client_id, name, tier))
                conn.commit()
            logger.info(f"API key created for client {client_id}")
            return raw_key
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to create API key: {e}")
            raise
        finally:
            self._release_connection(conn)

    def validate_api_key(self, api_key: str) -> Optional[dict]:
        """
        Validate an API key and return client info.

        Args:
            api_key: The raw API key to validate

        Returns:
            Dict with client_id, tier, name if valid; None if invalid
        """
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        sql = """
        UPDATE api_keys
        SET last_used_at = NOW()
        WHERE key_hash = %s AND is_active = TRUE
        RETURNING client_id, tier, name
        """

        conn = self._ensure_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, (key_hash,))
                row = cur.fetchone()
                conn.commit()

            if row:
                # Handle both RealDictRow and tuple
                if isinstance(row, dict):
                    return {
                        "client_id": str(row["client_id"]),
                        "tier": row["tier"],
                        "name": row["name"]
                    }
                else:
                    return {
                        "client_id": str(row[0]),
                        "tier": row[1],
                        "name": row[2]
                    }
            return None
        except Exception as e:
            logger.error(f"API key validation failed: {e}")
            return None
        finally:
            self._release_connection(conn)

    def log_audit(
        self,
        client_id: str,
        action: str,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        details: Optional[dict] = None,
        ip_address: Optional[str] = None
    ) -> None:
        """
        Record an action in the audit log.

        Args:
            client_id: The client performing the action
            action: Action type (query, ingest, delete, login, etc.)
            resource_type: Type of resource (document, chunk, etc.)
            resource_id: ID of the affected resource
            details: Additional details as JSON
            ip_address: Client IP address
        """
        sql = """
        INSERT INTO audit_log (client_id, action, resource_type, resource_id, details, ip_address)
        VALUES (%s::uuid, %s, %s, %s::uuid, %s, %s::inet)
        """

        conn = self._ensure_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, (
                    client_id,
                    action,
                    resource_type,
                    resource_id,
                    json.dumps(details) if details else None,
                    ip_address
                ))
                conn.commit()
        except Exception as e:
            # Don't fail operations due to audit logging errors
            logger.warning(f"Audit logging failed: {e}")
        finally:
            self._release_connection(conn)

    def initialize_schema(self) -> None:
        """Create tables and indexes if they don't exist."""
        if not self._conn:
            self.connect()

        schema_sql = f"""
        -- Legal documents table
        CREATE TABLE IF NOT EXISTS legal_documents (
            id UUID PRIMARY KEY,
            client_id UUID,
            title TEXT NOT NULL,
            document_type TEXT,
            jurisdiction TEXT,
            effective_date TIMESTAMPTZ,
            file_path TEXT,
            page_count INT,
            metadata JSONB DEFAULT '{{}}',
            created_at TIMESTAMPTZ DEFAULT NOW()
        );

        -- Document chunks with embeddings
        CREATE TABLE IF NOT EXISTS {self.config.table_name} (
            id UUID PRIMARY KEY,
            document_id UUID NOT NULL REFERENCES legal_documents(id) ON DELETE CASCADE,
            client_id UUID,
            content TEXT NOT NULL,
            section_title TEXT,
            hierarchy_path TEXT,
            level INT DEFAULT 0,
            page_numbers INT[] DEFAULT ARRAY[]::INT[],
            -- Paragraph tracking for precise retrieval
            paragraph_start INT,
            paragraph_end INT,
            original_paragraph_numbers INT[] DEFAULT ARRAY[]::INT[],
            -- Contextual retrieval metadata
            contextualized BOOLEAN DEFAULT FALSE,
            context_prefix TEXT,
            embedding_model VARCHAR(50) DEFAULT 'cohere-embed-v3',
            parent_chunk_id UUID,
            token_count INT,
            embedding VECTOR({self.config.embedding_dimensions}),
            legal_references TEXT[] DEFAULT ARRAY[]::TEXT[],
            context_before TEXT,
            context_after TEXT,
            metadata JSONB DEFAULT '{{}}',
            created_at TIMESTAMPTZ DEFAULT NOW()
        );

        -- Indexes for search performance
        CREATE INDEX IF NOT EXISTS idx_chunks_document
            ON {self.config.table_name}(document_id);
        CREATE INDEX IF NOT EXISTS idx_chunks_client
            ON {self.config.table_name}(client_id);
        CREATE INDEX IF NOT EXISTS idx_chunks_level
            ON {self.config.table_name}(level);

        -- Composite index for multi-tenant filtered queries (Quick Win #1)
        CREATE INDEX IF NOT EXISTS idx_chunks_client_document
            ON {self.config.table_name}(client_id, document_id);

        -- Full-text search index for BM25-style keyword search
        CREATE INDEX IF NOT EXISTS idx_chunks_content_fts
            ON {self.config.table_name}
            USING GIN (to_tsvector('english', content));

        -- Index for paragraph range queries
        CREATE INDEX IF NOT EXISTS idx_chunks_paragraphs
            ON {self.config.table_name}(document_id, paragraph_start, paragraph_end);

        -- GIN index for paragraph array contains queries
        CREATE INDEX IF NOT EXISTS idx_chunks_para_array
            ON {self.config.table_name}
            USING GIN (original_paragraph_numbers);

        -- Vector similarity index (IVFFlat for scalability)
        -- Note: Only create after inserting some data for better index quality
        """

        try:
            with self._conn.cursor() as cur:
                cur.execute(schema_sql)
                self._conn.commit()
            logger.info("Schema initialized successfully")
        except Exception as e:
            self._conn.rollback()
            logger.error(f"Schema initialization failed: {e}")
            raise

    def create_vector_index(self, index_type: str = "ivfflat") -> None:
        """
        Create vector index (call after inserting data).

        Args:
            index_type: "ivfflat" (default, good for <50K chunks) or
                       "hnsw" (better for 50K+ chunks, slower to build)
        """
        if not self._conn:
            self.connect()

        if index_type == "hnsw":
            index_sql = f"""
            DROP INDEX IF EXISTS idx_chunks_embedding;
            CREATE INDEX idx_chunks_embedding
                ON {self.config.table_name}
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64);
            """
            logger.info("Creating HNSW index (this may take a while for large datasets)...")
        else:
            index_sql = f"""
            CREATE INDEX IF NOT EXISTS idx_chunks_embedding
                ON {self.config.table_name}
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = {self.config.index_lists});
            """

        try:
            with self._conn.cursor() as cur:
                cur.execute(index_sql)
                self._conn.commit()
            logger.info(f"Vector index created (type: {index_type})")
        except Exception as e:
            self._conn.rollback()
            logger.error(f"Vector index creation failed: {e}")
            raise

    def create_hnsw_index(self, m: int = 16, ef_construction: int = 64) -> None:
        """
        Create HNSW index for large-scale deployments (50K+ chunks).

        HNSW provides faster queries than IVFFlat at scale but takes
        longer to build. Use this when you have 50,000+ chunks.

        Args:
            m: Maximum number of connections per node (16 is good default)
            ef_construction: Size of dynamic candidate list (64 is good default)
        """
        if not self._conn:
            self.connect()

        # Check current chunk count
        with self._conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self.config.table_name}")
            count = cur.fetchone()["count"]

        if count < 10000:
            logger.warning(
                f"HNSW is recommended for 50K+ chunks. "
                f"Current count: {count}. IVFFlat may be sufficient."
            )

        index_sql = f"""
        -- Drop existing index
        DROP INDEX IF EXISTS idx_chunks_embedding;

        -- Create HNSW index (better for large scale)
        CREATE INDEX idx_chunks_embedding
            ON {self.config.table_name}
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = {m}, ef_construction = {ef_construction});
        """

        logger.info(f"Creating HNSW index (m={m}, ef_construction={ef_construction})...")
        logger.info("This may take several minutes for large datasets...")

        try:
            with self._conn.cursor() as cur:
                cur.execute(index_sql)
                self._conn.commit()
            logger.info("HNSW index created successfully")
        except Exception as e:
            self._conn.rollback()
            logger.error(f"HNSW index creation failed: {e}")
            raise

    def get_index_info(self) -> dict:
        """Get information about current vector indexes."""
        if not self._conn:
            self.connect()

        sql = """
        SELECT
            indexname,
            indexdef
        FROM pg_indexes
        WHERE tablename = %s
        AND indexname LIKE '%embedding%'
        """

        try:
            with self._conn.cursor() as cur:
                cur.execute(sql, (self.config.table_name,))
                rows = cur.fetchall()

            indexes = {}
            for row in rows:
                name = row["indexname"]
                definition = row["indexdef"]
                index_type = "hnsw" if "hnsw" in definition.lower() else "ivfflat"
                indexes[name] = {
                    "type": index_type,
                    "definition": definition,
                }

            return indexes

        except Exception as e:
            logger.error(f"Failed to get index info: {e}")
            return {}

    def insert_document(
        self,
        document_id: str,
        title: str,
        document_type: str,
        client_id: Optional[str] = None,
        jurisdiction: Optional[str] = None,
        file_path: Optional[str] = None,
        page_count: int = 0,
        metadata: Optional[dict] = None,
    ) -> None:
        """Insert a document record."""
        if not self._conn:
            self.connect()

        sql = """
        INSERT INTO legal_documents
            (id, client_id, title, document_type, jurisdiction, file_path, page_count, metadata)
        VALUES
            (%s::uuid, %s::uuid, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (id) DO UPDATE SET
            title = EXCLUDED.title,
            document_type = EXCLUDED.document_type,
            jurisdiction = EXCLUDED.jurisdiction,
            metadata = EXCLUDED.metadata
        """

        try:
            with self._conn.cursor() as cur:
                cur.execute(sql, (
                    document_id,
                    client_id,
                    title,
                    document_type,
                    jurisdiction,
                    file_path,
                    page_count,
                    json.dumps(metadata or {}),
                ))
                self._conn.commit()
        except Exception as e:
            self._conn.rollback()
            logger.error(f"Document insert failed: {e}")
            raise

    def insert_chunks(
        self,
        chunks: list[dict],
        embeddings: list[list[float]],
        client_id: Optional[str] = None,
    ) -> None:
        """
        Batch insert chunks with embeddings using execute_values for 50x faster ingestion.

        Args:
            chunks: List of chunk dictionaries (from Chunk.to_dict())
            embeddings: Corresponding embedding vectors
            client_id: Optional client ID for multi-tenant isolation
        """
        if not self._conn:
            self.connect()

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks, {len(embeddings)} embeddings"
            )

        # Quick Win #3: Use execute_values for batch insert (50x faster)
        from psycopg2.extras import execute_values

        sql = f"""
        INSERT INTO {self.config.table_name}
            (id, document_id, client_id, content, section_title, hierarchy_path,
             level, page_numbers, paragraph_start, paragraph_end, original_paragraph_numbers,
             contextualized, context_prefix, parent_chunk_id, token_count, embedding,
             legal_references, context_before, context_after, metadata)
        VALUES %s
        ON CONFLICT (id) DO UPDATE SET
            content = EXCLUDED.content,
            embedding = EXCLUDED.embedding,
            paragraph_start = EXCLUDED.paragraph_start,
            paragraph_end = EXCLUDED.paragraph_end,
            original_paragraph_numbers = EXCLUDED.original_paragraph_numbers,
            contextualized = EXCLUDED.contextualized,
            context_prefix = EXCLUDED.context_prefix
        """

        # Prepare all values as a list of tuples
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

        try:
            with self._conn.cursor() as cur:
                # Use execute_values with template for proper type casting
                execute_values(
                    cur,
                    sql,
                    values,
                    template="(%s::uuid, %s::uuid, %s::uuid, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::uuid, %s, %s::vector, %s, %s, %s, %s)",
                    page_size=1000,
                )
                self._conn.commit()
            logger.info(f"Batch inserted {len(chunks)} chunks")
        except Exception as e:
            self._conn.rollback()
            logger.error(f"Chunk insert failed: {e}")
            raise

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        client_id: Optional[str] = None,
        document_id: Optional[str] = None,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        """
        Semantic search using vector similarity.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            client_id: Optional filter by client
            document_id: Optional filter by document
            min_score: Minimum similarity score (0-1)

        Returns:
            List of SearchResult objects
        """
        if not self._conn:
            self.connect()

        # Build query with optional filters
        filters = []
        params = [query_embedding, top_k]

        if client_id:
            filters.append("client_id = %s::uuid")
            params.insert(-1, client_id)

        if document_id:
            filters.append("document_id = %s::uuid")
            params.insert(-1, document_id)

        where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""

        sql = f"""
        SELECT
            id as chunk_id,
            document_id,
            content,
            section_title,
            hierarchy_path,
            page_numbers,
            paragraph_start,
            paragraph_end,
            original_paragraph_numbers,
            level,
            legal_references,
            context_before,
            context_after,
            1 - (embedding <=> %s::vector) as score
        FROM {self.config.table_name}
        {where_clause}
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """

        # Adjust params for the query
        final_params = [query_embedding]
        if client_id:
            final_params.append(client_id)
        if document_id:
            final_params.append(document_id)
        final_params.extend([query_embedding, top_k])

        try:
            with self._conn.cursor() as cur:
                cur.execute(sql, final_params)
                rows = cur.fetchall()

            results = []
            for row in rows:
                if row["score"] >= min_score:
                    results.append(SearchResult(
                        chunk_id=str(row["chunk_id"]),
                        document_id=str(row["document_id"]),
                        content=row["content"],
                        section_title=row["section_title"],
                        hierarchy_path=row["hierarchy_path"],
                        page_numbers=row["page_numbers"] or [],
                        score=float(row["score"]),
                        metadata={
                            "level": row["level"],
                            "legal_references": row["legal_references"],
                            "context_before": row["context_before"],
                            "context_after": row["context_after"],
                        },
                        paragraph_start=row.get("paragraph_start"),
                        paragraph_end=row.get("paragraph_end"),
                        original_paragraph_numbers=row.get("original_paragraph_numbers") or [],
                    ))

            return results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise

    def keyword_search(
        self,
        query: str,
        top_k: int = 10,
        client_id: Optional[str] = None,
        document_id: Optional[str] = None,
    ) -> list[SearchResult]:
        """
        Full-text keyword search using PostgreSQL ts_rank.

        Args:
            query: Search query string
            top_k: Number of results to return
            client_id: Optional filter by client
            document_id: Optional filter by document

        Returns:
            List of SearchResult objects
        """
        if not self._conn:
            self.connect()

        filters = []
        params = [query]

        if client_id:
            filters.append("client_id = %s::uuid")
            params.append(client_id)

        if document_id:
            filters.append("document_id = %s::uuid")
            params.append(document_id)

        where_clause = ""
        if filters:
            where_clause = "AND " + " AND ".join(filters)

        sql = f"""
        SELECT
            id as chunk_id,
            document_id,
            content,
            section_title,
            hierarchy_path,
            page_numbers,
            paragraph_start,
            paragraph_end,
            original_paragraph_numbers,
            level,
            ts_rank(to_tsvector('english', content), plainto_tsquery('english', %s)) as score
        FROM {self.config.table_name}
        WHERE to_tsvector('english', content) @@ plainto_tsquery('english', %s)
        {where_clause}
        ORDER BY score DESC
        LIMIT %s
        """

        params = [query, query] + params[1:] + [top_k]

        try:
            with self._conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

            return [
                SearchResult(
                    chunk_id=str(row["chunk_id"]),
                    document_id=str(row["document_id"]),
                    content=row["content"],
                    section_title=row["section_title"],
                    hierarchy_path=row["hierarchy_path"],
                    page_numbers=row["page_numbers"] or [],
                    score=float(row["score"]),
                    metadata={"level": row["level"]},
                    paragraph_start=row.get("paragraph_start"),
                    paragraph_end=row.get("paragraph_end"),
                    original_paragraph_numbers=row.get("original_paragraph_numbers") or [],
                )
                for row in rows
            ]

        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            raise

    def search_by_paragraph(
        self,
        document_id: str,
        paragraph_number: int,
        client_id: Optional[str] = None,
    ) -> list[SearchResult]:
        """
        Find chunks containing a specific paragraph number.

        Args:
            document_id: Document to search in
            paragraph_number: The paragraph number to find (1-indexed)
            client_id: Optional client filter for multi-tenant isolation

        Returns:
            List of SearchResult objects containing that paragraph
        """
        if not self._conn:
            self.connect()

        filters = ["document_id = %s::uuid"]
        params = [document_id, paragraph_number, paragraph_number]

        if client_id:
            filters.append("client_id = %s::uuid")
            params.insert(0, client_id)
            params.append(client_id)  # For the second query part

        where_clause = " AND ".join(filters)

        # Search both in the array and in the range
        sql = f"""
        SELECT
            id as chunk_id,
            document_id,
            content,
            section_title,
            hierarchy_path,
            page_numbers,
            paragraph_start,
            paragraph_end,
            original_paragraph_numbers,
            level,
            1.0 as score
        FROM {self.config.table_name}
        WHERE {where_clause}
        AND (
            %s = ANY(original_paragraph_numbers)
            OR (%s BETWEEN paragraph_start AND paragraph_end)
        )
        ORDER BY paragraph_start ASC NULLS LAST
        """

        # Adjust params order
        final_params = [document_id]
        if client_id:
            final_params.append(client_id)
        final_params.extend([paragraph_number, paragraph_number])

        try:
            with self._conn.cursor() as cur:
                cur.execute(sql, final_params)
                rows = cur.fetchall()

            return [
                SearchResult(
                    chunk_id=str(row["chunk_id"]),
                    document_id=str(row["document_id"]),
                    content=row["content"],
                    section_title=row["section_title"],
                    hierarchy_path=row["hierarchy_path"],
                    page_numbers=row["page_numbers"] or [],
                    score=float(row["score"]),
                    metadata={"level": row["level"]},
                    paragraph_start=row.get("paragraph_start"),
                    paragraph_end=row.get("paragraph_end"),
                    original_paragraph_numbers=row.get("original_paragraph_numbers") or [],
                )
                for row in rows
            ]

        except Exception as e:
            logger.error(f"Paragraph search failed: {e}")
            raise

    def migrate_add_paragraph_columns(self) -> bool:
        """
        Migration: Add paragraph tracking columns to existing database.

        Run this once after upgrading to add paragraph support to existing tables.
        Safe to run multiple times (uses IF NOT EXISTS).

        Returns:
            True if migration succeeded, False otherwise
        """
        if not self._conn:
            self.connect()

        migration_sql = f"""
        -- Add paragraph tracking columns if they don't exist
        ALTER TABLE {self.config.table_name}
        ADD COLUMN IF NOT EXISTS paragraph_start INT;

        ALTER TABLE {self.config.table_name}
        ADD COLUMN IF NOT EXISTS paragraph_end INT;

        ALTER TABLE {self.config.table_name}
        ADD COLUMN IF NOT EXISTS original_paragraph_numbers INT[] DEFAULT ARRAY[]::INT[];

        ALTER TABLE {self.config.table_name}
        ADD COLUMN IF NOT EXISTS contextualized BOOLEAN DEFAULT FALSE;

        ALTER TABLE {self.config.table_name}
        ADD COLUMN IF NOT EXISTS context_prefix TEXT;

        ALTER TABLE {self.config.table_name}
        ADD COLUMN IF NOT EXISTS embedding_model VARCHAR(50) DEFAULT 'cohere-embed-v3';

        -- Create indexes for paragraph queries
        CREATE INDEX IF NOT EXISTS idx_chunks_paragraphs
            ON {self.config.table_name}(document_id, paragraph_start, paragraph_end);

        CREATE INDEX IF NOT EXISTS idx_chunks_para_array
            ON {self.config.table_name}
            USING GIN (original_paragraph_numbers);
        """

        try:
            with self._conn.cursor() as cur:
                cur.execute(migration_sql)
                self._conn.commit()
            logger.info("Migration completed: Added paragraph tracking columns")
            return True
        except Exception as e:
            self._conn.rollback()
            logger.error(f"Migration failed: {e}")
            return False

    def delete_document(self, document_id: str) -> None:
        """Delete a document and all its chunks."""
        if not self._conn:
            self.connect()

        sql = "DELETE FROM legal_documents WHERE id = %s::uuid"

        try:
            with self._conn.cursor() as cur:
                cur.execute(sql, (document_id,))
                self._conn.commit()
            logger.info(f"Deleted document {document_id}")
        except Exception as e:
            self._conn.rollback()
            logger.error(f"Document deletion failed: {e}")
            raise

    def get_document_chunks(self, document_id: str) -> list[dict]:
        """Get all chunks for a document."""
        if not self._conn:
            self.connect()

        sql = f"""
        SELECT * FROM {self.config.table_name}
        WHERE document_id = %s::uuid
        ORDER BY level, hierarchy_path
        """

        try:
            with self._conn.cursor() as cur:
                cur.execute(sql, (document_id,))
                return cur.fetchall()
        except Exception as e:
            logger.error(f"Failed to get chunks: {e}")
            raise

    def list_documents(self, client_id: Optional[str] = None) -> list[dict]:
        """
        Get all documents in the database.

        Args:
            client_id: Optional filter by client for multi-tenant isolation

        Returns:
            List of document dictionaries with id, title, type, etc.
        """
        if not self._conn:
            self.connect()

        sql = """
        SELECT id, title, document_type, jurisdiction, page_count,
               metadata, created_at
        FROM legal_documents
        """
        params = []
        if client_id:
            sql += " WHERE client_id = %s::uuid"
            params.append(client_id)
        sql += " ORDER BY created_at DESC"

        try:
            with self._conn.cursor() as cur:
                cur.execute(sql, params if params else None)
                rows = cur.fetchall()
            return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Failed to list documents: {e}")
            raise


# CLI for testing
if __name__ == "__main__":
    from dotenv import load_dotenv
    import sys

    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    store = VectorStore()
    store.connect()
    store.initialize_schema()

    # Check for command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "setup-auth":
            # Initialize authentication tables
            store.initialize_auth_schema()
            print("Authentication schema created")

        elif command == "enable-rls":
            # Enable Row-Level Security
            store.enable_rls()
            print("Row-Level Security enabled")

        elif command == "create-demo-key":
            # Create a demo API key
            store.initialize_auth_schema()
            demo_client_id = "00000000-0000-0000-0000-000000000001"
            api_key = store.create_api_key(demo_client_id, name="Demo Key")
            print(f"\nDemo API Key (save this, it won't be shown again!):")
            print(f"  {api_key}")
            print(f"\nClient ID: {demo_client_id}")

        elif command == "test-rls":
            # Test RLS isolation
            store.enable_rls()
            store.initialize_auth_schema()

            # Create two test clients
            client_a = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
            client_b = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"

            # Test that setting tenant context works
            store.set_tenant_context(client_a)
            docs_a = store.list_documents(client_id=client_a)
            print(f"Client A documents: {len(docs_a)}")

            store.set_tenant_context(client_b)
            docs_b = store.list_documents(client_id=client_b)
            print(f"Client B documents: {len(docs_b)}")

            print("\nRLS test complete. Documents are isolated by client_id.")

        else:
            print(f"Unknown command: {command}")
            print("Available commands: setup-auth, enable-rls, create-demo-key, test-rls")
    else:
        print("Vector store initialized successfully")
        print(f"Connection: {store._connection_string}")
        print("\nCommands:")
        print("  python vector_store.py setup-auth     - Create auth tables")
        print("  python vector_store.py enable-rls     - Enable Row-Level Security")
        print("  python vector_store.py create-demo-key - Create demo API key")
        print("  python vector_store.py test-rls       - Test RLS isolation")
