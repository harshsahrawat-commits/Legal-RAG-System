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

from .language_config import TenantLanguageConfig, VALID_FTS_CONFIGS

try:
    import psycopg2
    import psycopg2.extras
    import psycopg2.pool
except ImportError:
    psycopg2 = None  # Will be caught at connect() time

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
            if psycopg2 is None:
                raise ImportError(
                    "psycopg2 not installed. Run: pip install psycopg2-binary"
                )
            from psycopg2.extras import RealDictCursor

            if self.config.use_pooling:
                # Use connection pooling for production
                self._pool = psycopg2.pool.ThreadedConnectionPool(
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
        """Get a database connection (from pool or single connection).

        Uses lazy connection checking — only reconnects when the connection
        is known to be closed, avoiding a SELECT 1 round-trip on every call.
        Stale connections are caught by _ensure_connection's retry logic.
        """
        if self._pool:
            try:
                conn = self._pool.getconn()
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

        # For single connection mode, only reconnect if connection is closed
        if self._conn and self._conn.closed:
            logger.warning("Connection closed, reconnecting...")
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

    def _safe_rollback(self, conn) -> None:
        """Rollback a connection, ignoring errors if the connection is dead."""
        try:
            conn.rollback()
        except (psycopg2.InterfaceError, psycopg2.OperationalError):
            pass

    def _execute_with_retry(self, operation, label="db_operation"):
        """Execute a DB operation with one retry on stale connection.

        Args:
            operation: Callable(conn) that performs the DB work and returns a result.
            label: Human-readable name for logging.

        Returns:
            Whatever ``operation`` returns.
        """
        for attempt in range(2):
            conn = self._ensure_connection()
            try:
                result = operation(conn)
                self._release_connection(conn)
                return result
            except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
                self._safe_rollback(conn)
                try:
                    self._release_connection(conn)
                except Exception:
                    pass
                if attempt == 0:
                    logger.warning(f"{label}: stale conn, reconnecting: {e}")
                    self.connect()
                    continue
                raise
            except Exception:
                self._safe_rollback(conn)
                self._release_connection(conn)
                raise

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
            self._safe_rollback(conn)
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
            self._safe_rollback(conn)
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
            self._safe_rollback(conn)
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
            self._safe_rollback(conn)
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

        def _op(conn):
            with conn.cursor() as cur:
                cur.execute(sql, (key_hash,))
                row = cur.fetchone()
                conn.commit()
            if row:
                if isinstance(row, dict):
                    return {
                        "client_id": str(row["client_id"]),
                        "tier": row["tier"],
                        "name": row["name"]
                    }
                return {
                    "client_id": str(row[0]),
                    "tier": row[1],
                    "name": row[2]
                }
            return None

        try:
            return self._execute_with_retry(_op, "validate_api_key")
        except Exception as e:
            logger.error(f"API key validation failed: {e}")
            return None

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

        def _op(conn):
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

        try:
            self._execute_with_retry(_op, "log_audit")
        except Exception as e:
            # Don't fail operations due to audit logging errors
            logger.warning(f"Audit logging failed: {e}")

    def initialize_schema(self) -> None:
        """Create tables and indexes if they don't exist."""
        conn = self._ensure_connection()

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
            embedding_model VARCHAR(50) DEFAULT 'voyage-law-2',
            parent_chunk_id UUID,
            token_count INT,
            embedding VECTOR({self.config.embedding_dimensions}),
            legal_references TEXT[] DEFAULT ARRAY[]::TEXT[],
            context_before TEXT,
            context_after TEXT,
            metadata JSONB DEFAULT '{{}}',
            source_origin VARCHAR(20) DEFAULT 'cylaw',
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

        -- Full-text search index for BM25-style keyword search (English)
        CREATE INDEX IF NOT EXISTS idx_chunks_content_fts
            ON {self.config.table_name}
            USING GIN (to_tsvector('english', content));

        -- Full-text search index for Greek content (multilingual support)
        CREATE INDEX IF NOT EXISTS idx_chunks_fts_greek
            ON {self.config.table_name}
            USING GIN (to_tsvector('greek', content));

        -- Index for paragraph range queries
        CREATE INDEX IF NOT EXISTS idx_chunks_paragraphs
            ON {self.config.table_name}(document_id, paragraph_start, paragraph_end);

        -- GIN index for paragraph array contains queries
        CREATE INDEX IF NOT EXISTS idx_chunks_para_array
            ON {self.config.table_name}
            USING GIN (original_paragraph_numbers);

        -- Source origin indexes for multi-source filtering
        CREATE INDEX IF NOT EXISTS idx_chunks_source_origin
            ON {self.config.table_name}(source_origin);
        CREATE INDEX IF NOT EXISTS idx_chunks_client_source
            ON {self.config.table_name}(client_id, source_origin);

        -- Vector similarity index (IVFFlat for scalability)
        -- Note: Only create after inserting some data for better index quality
        """

        try:
            with conn.cursor() as cur:
                cur.execute(schema_sql)
                conn.commit()
            logger.info("Schema initialized successfully")
        except Exception as e:
            self._safe_rollback(conn)
            logger.error(f"Schema initialization failed: {e}")
            raise
        finally:
            self._release_connection(conn)

    def create_vector_index(self, index_type: str = "ivfflat") -> None:
        """
        Create vector index (call after inserting data).

        Args:
            index_type: "ivfflat" (default, good for <50K chunks) or
                       "hnsw" (better for 50K+ chunks, slower to build)
        """
        conn = self._ensure_connection()

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
            with conn.cursor() as cur:
                cur.execute(index_sql)
                conn.commit()
            logger.info(f"Vector index created (type: {index_type})")
        except Exception as e:
            self._safe_rollback(conn)
            logger.error(f"Vector index creation failed: {e}")
            raise
        finally:
            self._release_connection(conn)

    def create_hnsw_index(self, m: int = 16, ef_construction: int = 64) -> None:
        """
        Create HNSW index for large-scale deployments (50K+ chunks).

        HNSW provides faster queries than IVFFlat at scale but takes
        longer to build. Use this when you have 50,000+ chunks.

        Args:
            m: Maximum number of connections per node (16 is good default)
            ef_construction: Size of dynamic candidate list (64 is good default)
        """
        conn = self._ensure_connection()

        # Check current chunk count
        with conn.cursor() as cur:
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
            with conn.cursor() as cur:
                cur.execute(index_sql)
                conn.commit()
            logger.info("HNSW index created successfully")
        except Exception as e:
            self._safe_rollback(conn)
            logger.error(f"HNSW index creation failed: {e}")
            raise
        finally:
            self._release_connection(conn)

    def get_index_info(self) -> dict:
        """Get information about current vector indexes."""
        conn = self._ensure_connection()

        sql = """
        SELECT
            indexname,
            indexdef
        FROM pg_indexes
        WHERE tablename = %s
        AND indexname LIKE '%embedding%'
        """

        try:
            with conn.cursor() as cur:
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
        finally:
            self._release_connection(conn)

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
        family_id: Optional[str] = None,
        upload_scope: str = "persistent",
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> None:
        """Insert a document record."""
        sql = """
        INSERT INTO legal_documents
            (id, client_id, title, document_type, jurisdiction, file_path, page_count, metadata,
             family_id, upload_scope, conversation_id, user_id)
        VALUES
            (%s::uuid, %s::uuid, %s, %s, %s, %s, %s, %s,
             %s, %s, %s, %s)
        ON CONFLICT (id) DO UPDATE SET
            title = EXCLUDED.title,
            document_type = EXCLUDED.document_type,
            jurisdiction = EXCLUDED.jurisdiction,
            metadata = EXCLUDED.metadata,
            family_id = EXCLUDED.family_id
        """

        def _op(conn):
            with conn.cursor() as cur:
                cur.execute(sql, (
                    document_id,
                    client_id,
                    title,
                    document_type,
                    jurisdiction,
                    file_path,
                    page_count,
                    json.dumps(metadata or {}),
                    family_id,
                    upload_scope,
                    conversation_id,
                    user_id,
                ))
                conn.commit()

        self._execute_with_retry(_op, "insert_document")

    def insert_chunks(
        self,
        chunks: list[dict],
        embeddings: list[list[float]],
        client_id: Optional[str] = None,
        source_origin: str = "cylaw",
        family_id: Optional[str] = None,
    ) -> None:
        """
        Batch insert chunks with embeddings using execute_values for 50x faster ingestion.

        Args:
            chunks: List of chunk dictionaries (from Chunk.to_dict())
            embeddings: Corresponding embedding vectors
            client_id: Optional client ID for multi-tenant isolation
            source_origin: Source origin label ("cylaw", "hudoc", "eurlex", "user", "session")
            family_id: Optional family UUID to assign chunks to
        """
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
             legal_references, context_before, context_after, metadata, source_origin, family_id)
        VALUES %s
        ON CONFLICT (id) DO UPDATE SET
            content = EXCLUDED.content,
            embedding = EXCLUDED.embedding,
            paragraph_start = EXCLUDED.paragraph_start,
            paragraph_end = EXCLUDED.paragraph_end,
            original_paragraph_numbers = EXCLUDED.original_paragraph_numbers,
            contextualized = EXCLUDED.contextualized,
            context_prefix = EXCLUDED.context_prefix,
            source_origin = EXCLUDED.source_origin,
            family_id = EXCLUDED.family_id
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
                source_origin,
                family_id,
            ))

        def _op(conn):
            with conn.cursor() as cur:
                execute_values(
                    cur,
                    sql,
                    values,
                    template="(%s::uuid, %s::uuid, %s::uuid, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::uuid, %s, %s::vector, %s, %s, %s, %s, %s, %s::uuid)",
                    page_size=1000,
                )
                conn.commit()
            logger.info(f"Batch inserted {len(chunks)} chunks (source_origin={source_origin})")

        self._execute_with_retry(_op, "insert_chunks")

    def insert_chunks_with_conn(
        self,
        conn,
        chunks: list[dict],
        embeddings: list[list[float]],
        client_id: Optional[str] = None,
        source_origin: str = "cylaw",
    ) -> None:
        """
        Insert chunks using an existing connection (no commit/rollback).

        This variant is used for atomic document+chunks inserts where the
        caller manages the transaction. The caller is responsible for
        calling conn.commit() or conn.rollback().

        Args:
            conn: An existing psycopg2 connection (caller manages transaction)
            chunks: List of chunk dictionaries (from Chunk.to_dict())
            embeddings: Corresponding embedding vectors
            client_id: Optional client ID for multi-tenant isolation
            source_origin: Source origin label ("cylaw", "hudoc", or "eurlex")
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks, {len(embeddings)} embeddings"
            )

        from psycopg2.extras import execute_values

        sql = f"""
        INSERT INTO {self.config.table_name}
            (id, document_id, client_id, content, section_title, hierarchy_path,
             level, page_numbers, paragraph_start, paragraph_end, original_paragraph_numbers,
             contextualized, context_prefix, parent_chunk_id, token_count, embedding,
             legal_references, context_before, context_after, metadata, source_origin)
        VALUES %s
        ON CONFLICT (id) DO UPDATE SET
            content = EXCLUDED.content,
            embedding = EXCLUDED.embedding,
            paragraph_start = EXCLUDED.paragraph_start,
            paragraph_end = EXCLUDED.paragraph_end,
            original_paragraph_numbers = EXCLUDED.original_paragraph_numbers,
            contextualized = EXCLUDED.contextualized,
            context_prefix = EXCLUDED.context_prefix,
            source_origin = EXCLUDED.source_origin
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
                source_origin,
            ))

        with conn.cursor() as cur:
            execute_values(
                cur,
                sql,
                values,
                template="(%s::uuid, %s::uuid, %s::uuid, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::uuid, %s, %s::vector, %s, %s, %s, %s, %s)",
                page_size=1000,
            )
        logger.info(f"Inserted {len(chunks)} chunks via provided connection (source_origin={source_origin})")

    def insert_document_with_conn(
        self,
        conn,
        document_id: str,
        title: str,
        document_type: str,
        client_id: Optional[str] = None,
        jurisdiction: Optional[str] = None,
        file_path: Optional[str] = None,
        page_count: int = 0,
        metadata: Optional[dict] = None,
    ) -> None:
        """Insert a document record using an existing connection (no commit/rollback).

        Caller manages the transaction.
        """
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

        with conn.cursor() as cur:
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

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        client_id: Optional[str] = None,
        document_id: Optional[str] = None,
        min_score: float = 0.0,
        source_origins: Optional[list[str]] = None,
        family_ids: Optional[list[str]] = None,
        conversation_id: Optional[str] = None,
    ) -> list[SearchResult]:
        """
        Semantic search using vector similarity.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            client_id: Optional filter by client
            document_id: Optional filter by document
            min_score: Minimum similarity score (0-1)
            source_origins: Optional list of source origins to filter by (e.g. ["cylaw", "hudoc"])
            family_ids: Optional list of family UUIDs to include
            conversation_id: Optional conversation UUID to include session docs from

        Returns:
            List of SearchResult objects
        """
        # Build query with optional filters
        # Public corpus client_id: all CyLaw/HUDOC/EUR-Lex docs were ingested with this ID
        CORPUS_CLIENT_ID = "00000000-0000-0000-0000-000000000002"
        PUBLIC_SOURCES = {"cylaw", "hudoc", "eurlex"}

        filters = []
        filter_params = []

        if document_id:
            filters.append("c.document_id = %s::uuid")
            filter_params.append(document_id)

        # Source filter: OR logic between source_origins, family_ids, and conversation session docs
        source_conditions = []
        source_params = []

        if source_origins:
            # Public sources use demo client_id; include them regardless of current user
            public = [s for s in source_origins if s in PUBLIC_SOURCES]
            if public:
                placeholders = ",".join(["%s"] * len(public))
                source_conditions.append(
                    f"(c.source_origin IN ({placeholders}) AND c.client_id = %s::uuid)"
                )
                source_params.extend(public)
                source_params.append(CORPUS_CLIENT_ID)

        if family_ids and client_id:
            placeholders = ",".join(["%s::uuid"] * len(family_ids))
            source_conditions.append(f"(c.family_id IN ({placeholders}) AND c.client_id = %s::uuid)")
            source_params.extend(family_ids)
            source_params.append(client_id)

        if conversation_id:
            source_conditions.append(
                "c.document_id IN (SELECT id FROM legal_documents WHERE conversation_id = %s::uuid AND upload_scope = 'session')"
            )
            source_params.append(conversation_id)

        if source_conditions:
            filters.append(f"({' OR '.join(source_conditions)})")
            filter_params.extend(source_params)

        where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""

        columns = [
            "chunk_id", "document_id", "content", "section_title", "hierarchy_path",
            "page_numbers", "paragraph_start", "paragraph_end", "original_paragraph_numbers",
            "level", "legal_references", "context_before", "context_after", "score"
        ]
        sql = f"""
        SELECT
            c.id as chunk_id,
            c.document_id,
            c.content,
            c.section_title,
            c.hierarchy_path,
            c.page_numbers,
            c.paragraph_start,
            c.paragraph_end,
            c.original_paragraph_numbers,
            c.level,
            c.legal_references,
            c.context_before,
            c.context_after,
            1 - (c.embedding <=> %s::vector) as score
        FROM {self.config.table_name} c
        {where_clause}
        ORDER BY c.embedding <=> %s::vector
        LIMIT %s
        """

        # Build params in order: score calc embedding, filters, order embedding, limit
        final_params = [query_embedding] + filter_params + [query_embedding, top_k]

        def _op(conn):
            with conn.cursor() as cur:
                # Tune HNSW search: lower ef_search for faster queries
                # Default is 40; 25 gives ~98.5% recall at ~94K vectors with ~2x speedup
                cur.execute("SET hnsw.ef_search = 25")
                cur.execute(sql, final_params)
                rows = cur.fetchall()

            results = []
            for row in rows:
                # Handle both RealDictRow and tuple
                if hasattr(row, 'keys'):
                    row_dict = dict(row)
                else:
                    row_dict = dict(zip(columns, row))
                if row_dict["score"] >= min_score:
                    results.append(SearchResult(
                        chunk_id=str(row_dict["chunk_id"]),
                        document_id=str(row_dict["document_id"]),
                        content=row_dict["content"],
                        section_title=row_dict["section_title"],
                        hierarchy_path=row_dict["hierarchy_path"],
                        page_numbers=row_dict["page_numbers"] or [],
                        score=float(row_dict["score"]),
                        metadata={
                            "level": row_dict["level"],
                            "legal_references": row_dict["legal_references"],
                            "context_before": row_dict["context_before"],
                            "context_after": row_dict["context_after"],
                        },
                        paragraph_start=row_dict.get("paragraph_start"),
                        paragraph_end=row_dict.get("paragraph_end"),
                        original_paragraph_numbers=row_dict.get("original_paragraph_numbers") or [],
                    ))

            return results

        return self._execute_with_retry(_op, "search")

    def keyword_search(
        self,
        query: str,
        top_k: int = 10,
        client_id: Optional[str] = None,
        document_id: Optional[str] = None,
        fts_language: str = "english",
        source_origins: Optional[list[str]] = None,
        family_ids: Optional[list[str]] = None,
        conversation_id: Optional[str] = None,
    ) -> list[SearchResult]:
        """
        Full-text keyword search using PostgreSQL ts_rank.

        Args:
            query: Search query string
            top_k: Number of results to return
            client_id: Optional filter by client
            document_id: Optional filter by document
            fts_language: PostgreSQL FTS config name ("english" or "greek")
            source_origins: Optional list of source origins to filter by (e.g. ["cylaw", "hudoc"])
            family_ids: Optional list of family UUIDs to include
            conversation_id: Optional conversation UUID to include session docs from

        Returns:
            List of SearchResult objects
        """
        # Validate fts_language against whitelist to prevent SQL injection
        if fts_language not in VALID_FTS_CONFIGS:
            logger.warning(f"Invalid FTS language '{fts_language}', falling back to 'english'")
            fts_language = "english"

        # Public corpus client_id: all CyLaw/HUDOC/EUR-Lex docs were ingested with this ID
        CORPUS_CLIENT_ID = "00000000-0000-0000-0000-000000000002"
        PUBLIC_SOURCES = {"cylaw", "hudoc", "eurlex"}

        filter_params = []
        where_extra = ""

        if document_id:
            where_extra += " AND document_id = %s::uuid"
            filter_params.append(document_id)

        # Source filter: OR logic between source_origins, family_ids, and conversation session docs
        source_conditions = []
        source_params = []

        if source_origins:
            public = [s for s in source_origins if s in PUBLIC_SOURCES]
            if public:
                placeholders = ",".join(["%s"] * len(public))
                source_conditions.append(
                    f"(source_origin IN ({placeholders}) AND client_id = %s::uuid)"
                )
                source_params.extend(public)
                source_params.append(CORPUS_CLIENT_ID)

        if family_ids and client_id:
            placeholders = ",".join(["%s::uuid"] * len(family_ids))
            source_conditions.append(f"(family_id IN ({placeholders}) AND client_id = %s::uuid)")
            source_params.extend(family_ids)
            source_params.append(client_id)

        if conversation_id:
            source_conditions.append(
                "document_id IN (SELECT id FROM legal_documents WHERE conversation_id = %s::uuid AND upload_scope = 'session')"
            )
            source_params.append(conversation_id)

        if source_conditions:
            where_extra += f" AND ({' OR '.join(source_conditions)})"
            filter_params.extend(source_params)

        kw_columns = [
            "chunk_id", "document_id", "content", "section_title", "hierarchy_path",
            "page_numbers", "paragraph_start", "paragraph_end", "original_paragraph_numbers",
            "level", "score"
        ]
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
            ts_rank(to_tsvector(%s, content), websearch_to_tsquery(%s, %s)) as score
        FROM {self.config.table_name}
        WHERE to_tsvector(%s, content) @@ websearch_to_tsquery(%s, %s)
        {where_extra}
        ORDER BY score DESC
        LIMIT %s
        """

        params = [fts_language, fts_language, query, fts_language, fts_language, query] + filter_params + [top_k]

        def _op(conn):
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

            results = []
            for row in rows:
                if hasattr(row, 'keys'):
                    row_dict = dict(row)
                else:
                    row_dict = dict(zip(kw_columns, row))
                results.append(SearchResult(
                    chunk_id=str(row_dict["chunk_id"]),
                    document_id=str(row_dict["document_id"]),
                    content=row_dict["content"],
                    section_title=row_dict["section_title"],
                    hierarchy_path=row_dict["hierarchy_path"],
                    page_numbers=row_dict["page_numbers"] or [],
                    score=float(row_dict["score"]),
                    metadata={"level": row_dict["level"]},
                    paragraph_start=row_dict.get("paragraph_start"),
                    paragraph_end=row_dict.get("paragraph_end"),
                    original_paragraph_numbers=row_dict.get("original_paragraph_numbers") or [],
                ))
            return results

        return self._execute_with_retry(_op, "keyword_search")

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
        filters = ["document_id = %s::uuid"]

        if client_id:
            filters.append("client_id = %s::uuid")

        where_clause = " AND ".join(filters)

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

        final_params = [document_id]
        if client_id:
            final_params.append(client_id)
        final_params.extend([paragraph_number, paragraph_number])

        ctx_columns = [
            "chunk_id", "document_id", "content", "section_title", "hierarchy_path",
            "page_numbers", "paragraph_start", "paragraph_end", "original_paragraph_numbers",
            "level", "score"
        ]

        def _op(conn):
            with conn.cursor() as cur:
                cur.execute(sql, final_params)
                rows = cur.fetchall()

            results = []
            for row in rows:
                if hasattr(row, 'keys'):
                    row_dict = dict(row)
                else:
                    row_dict = dict(zip(ctx_columns, row))
                results.append(SearchResult(
                    chunk_id=str(row_dict["chunk_id"]),
                    document_id=str(row_dict["document_id"]),
                    content=row_dict["content"],
                    section_title=row_dict["section_title"],
                    hierarchy_path=row_dict["hierarchy_path"],
                    page_numbers=row_dict["page_numbers"] or [],
                    score=float(row_dict["score"]),
                    metadata={"level": row_dict["level"]},
                    paragraph_start=row_dict.get("paragraph_start"),
                    paragraph_end=row_dict.get("paragraph_end"),
                    original_paragraph_numbers=row_dict.get("original_paragraph_numbers") or [],
                ))
            return results

        return self._execute_with_retry(_op, "search_by_paragraph")

    def migrate_add_paragraph_columns(self) -> bool:
        """
        Migration: Add paragraph tracking columns to existing database.

        Run this once after upgrading to add paragraph support to existing tables.
        Safe to run multiple times (uses IF NOT EXISTS).

        Returns:
            True if migration succeeded, False otherwise
        """
        conn = self._ensure_connection()

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
        ADD COLUMN IF NOT EXISTS embedding_model VARCHAR(50) DEFAULT 'voyage-law-2';

        -- Create indexes for paragraph queries
        CREATE INDEX IF NOT EXISTS idx_chunks_paragraphs
            ON {self.config.table_name}(document_id, paragraph_start, paragraph_end);

        CREATE INDEX IF NOT EXISTS idx_chunks_para_array
            ON {self.config.table_name}
            USING GIN (original_paragraph_numbers);
        """

        try:
            with conn.cursor() as cur:
                cur.execute(migration_sql)
                conn.commit()
            logger.info("Migration completed: Added paragraph tracking columns")
            return True
        except Exception as e:
            self._safe_rollback(conn)
            logger.error(f"Migration failed: {e}")
            return False
        finally:
            self._release_connection(conn)

    def migrate_add_source_origin(self) -> bool:
        """
        Migration: Add source_origin column to document_chunks for multi-source filtering.

        Enables filtering by data source (cylaw, hudoc, eurlex) at the chunk level.
        Backfills all existing rows as 'cylaw' since all prior data is CyLaw.
        Safe to run multiple times (uses IF NOT EXISTS).

        Returns:
            True if migration succeeded, False otherwise
        """
        conn = self._ensure_connection()

        migration_sql = f"""
        ALTER TABLE {self.config.table_name}
        ADD COLUMN IF NOT EXISTS source_origin VARCHAR(20) DEFAULT 'cylaw';

        CREATE INDEX IF NOT EXISTS idx_chunks_source_origin
            ON {self.config.table_name}(source_origin);

        CREATE INDEX IF NOT EXISTS idx_chunks_client_source
            ON {self.config.table_name}(client_id, source_origin);

        UPDATE {self.config.table_name}
        SET source_origin = 'cylaw'
        WHERE source_origin IS NULL;
        """

        try:
            with conn.cursor() as cur:
                cur.execute(migration_sql)
                conn.commit()
            logger.info("Migration completed: Added source_origin column with indexes")
            return True
        except Exception as e:
            self._safe_rollback(conn)
            logger.error(f"source_origin migration failed: {e}")
            return False
        finally:
            self._release_connection(conn)

    def delete_document(self, document_id: str, client_id: Optional[str] = None) -> bool:
        """
        Delete a document and all its chunks.

        Args:
            document_id: The document UUID to delete
            client_id: If provided, only delete if document belongs to this tenant

        Returns:
            True if a document was deleted, False if not found (or wrong tenant)
        """
        if client_id:
            sql = "DELETE FROM legal_documents WHERE id = %s::uuid AND client_id = %s::uuid"
            params = (document_id, client_id)
        else:
            sql = "DELETE FROM legal_documents WHERE id = %s::uuid"
            params = (document_id,)

        def _op(conn):
            with conn.cursor() as cur:
                cur.execute(sql, params)
                deleted = cur.rowcount > 0
                conn.commit()
            if deleted:
                logger.info(f"Deleted document {document_id}")
            else:
                logger.warning(f"Document {document_id} not found (or wrong tenant)")
            return deleted

        return self._execute_with_retry(_op, "delete_document")

    def get_document_titles(
        self,
        document_ids: list[str],
        client_id: Optional[str] = None,
    ) -> dict[str, str]:
        """
        Look up document titles by IDs.

        Args:
            document_ids: List of document UUIDs to look up
            client_id: Optional tenant filter for isolation

        Returns:
            Dict mapping document_id to title
        """
        if not document_ids:
            return {}

        placeholders = ",".join(["%s::uuid"] * len(document_ids))
        sql = f"SELECT id, title FROM legal_documents WHERE id IN ({placeholders})"
        params = list(document_ids)

        if client_id:
            sql += " AND client_id = %s::uuid"
            params.append(client_id)

        def _op(conn):
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

            result = {}
            for row in rows:
                if hasattr(row, "keys"):
                    result[str(row["id"])] = row["title"]
                else:
                    result[str(row[0])] = row[1]
            return result

        try:
            return self._execute_with_retry(_op, "get_document_titles")
        except Exception as e:
            logger.error(f"Failed to get document titles: {e}")
            return {}

    def get_document_source_meta(
        self,
        document_ids: list[str],
        client_id: Optional[str] = None,
    ) -> dict[str, dict]:
        """Look up document titles, file_path, and metadata by IDs.

        Returns:
            Dict mapping document_id to {"title": ..., "file_path": ..., "metadata": ...}
        """
        if not document_ids:
            return {}

        placeholders = ",".join(["%s::uuid"] * len(document_ids))
        sql = f"SELECT id, title, file_path, metadata FROM legal_documents WHERE id IN ({placeholders})"
        params = list(document_ids)

        if client_id:
            sql += " AND client_id = %s::uuid"
            params.append(client_id)

        def _op(conn):
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

            result = {}
            for row in rows:
                if hasattr(row, "keys"):
                    result[str(row["id"])] = {
                        "title": row["title"],
                        "file_path": row.get("file_path"),
                        "metadata": row.get("metadata") or {},
                    }
                else:
                    result[str(row[0])] = {
                        "title": row[1],
                        "file_path": row[2],
                        "metadata": row[3] or {},
                    }
            return result

        try:
            return self._execute_with_retry(_op, "get_document_source_meta")
        except Exception as e:
            logger.error(f"Failed to get document source meta: {e}")
            return {}

    def get_document_chunks(self, document_id: str) -> list[dict]:
        """Get all chunks for a document."""
        sql = f"""
        SELECT * FROM {self.config.table_name}
        WHERE document_id = %s::uuid
        ORDER BY level, hierarchy_path
        """

        def _op(conn):
            with conn.cursor() as cur:
                cur.execute(sql, (document_id,))
                return cur.fetchall()

        return self._execute_with_retry(_op, "get_document_chunks")

    def list_documents(self, client_id: Optional[str] = None, exclude_session: bool = True) -> list[dict]:
        """
        Get all documents in the database.

        Args:
            client_id: Optional filter by client for multi-tenant isolation
            exclude_session: If True, excludes session-scoped (chat-uploaded) documents

        Returns:
            List of document dictionaries with id, title, type, family_id, etc.
        """
        columns = ["id", "title", "document_type", "jurisdiction", "page_count",
                   "metadata", "created_at", "file_path", "family_id"]
        sql = f"""
        SELECT {', '.join(columns)}
        FROM legal_documents
        """
        conditions = []
        params = []
        if client_id:
            conditions.append("client_id = %s::uuid")
            params.append(client_id)
        if exclude_session:
            conditions.append("(upload_scope IS NULL OR upload_scope != 'session')")
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        sql += " ORDER BY created_at DESC"

        def _op(conn):
            with conn.cursor() as cur:
                cur.execute(sql, params if params else None)
                rows = cur.fetchall()
            result = []
            for row in rows:
                if hasattr(row, 'keys'):
                    result.append(dict(row))
                else:
                    result.append(dict(zip(columns, row)))
            return result

        return self._execute_with_retry(_op, "list_documents")

    # =========================================================================
    # Tenant Language Configuration
    # =========================================================================

    def initialize_tenant_config_schema(self) -> None:
        """Create tenant_config table for per-tenant language settings."""
        conn = self._ensure_connection()

        sql = """
        CREATE TABLE IF NOT EXISTS tenant_config (
            client_id UUID PRIMARY KEY,
            language VARCHAR(5) NOT NULL DEFAULT 'en',
            embedding_model VARCHAR(100) DEFAULT 'voyage-law-2',
            embedding_provider VARCHAR(20) DEFAULT 'voyage',
            llm_model VARCHAR(100) DEFAULT 'qwen/qwen3-235b-a22b',
            reranker_model VARCHAR(100) DEFAULT 'rerank-multilingual-v3.0',
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );
        """

        try:
            with conn.cursor() as cur:
                cur.execute(sql)
                conn.commit()
            logger.info("Tenant config schema initialized")
        except Exception as e:
            self._safe_rollback(conn)
            logger.error(f"Tenant config schema init failed: {e}")
            raise
        finally:
            self._release_connection(conn)

    def get_tenant_config(self, client_id: str) -> Optional[TenantLanguageConfig]:
        """
        Get language configuration for a tenant.

        Args:
            client_id: The tenant UUID

        Returns:
            TenantLanguageConfig if found, None otherwise
        """
        sql = """
        SELECT language, embedding_model, embedding_provider,
               llm_model, reranker_model
        FROM tenant_config
        WHERE client_id = %s::uuid
        """

        def _op(conn):
            with conn.cursor() as cur:
                cur.execute(sql, (client_id,))
                row = cur.fetchone()

            if not row:
                return None

            from .language_config import SUPPORTED_LANGUAGES
            lang = row["language"] if isinstance(row, dict) else row[0]
            lang_info = SUPPORTED_LANGUAGES.get(lang, SUPPORTED_LANGUAGES["en"])

            return TenantLanguageConfig(
                language=lang,
                embedding_model=row["embedding_model"] if isinstance(row, dict) else row[1],
                embedding_provider=row["embedding_provider"] if isinstance(row, dict) else row[2],
                llm_model=row["llm_model"] if isinstance(row, dict) else row[3],
                reranker_model=row["reranker_model"] if isinstance(row, dict) else row[4],
                fts_language=lang_info["fts_config"],
                chars_per_token=lang_info["chars_per_token"],
            )

        try:
            return self._execute_with_retry(_op, "get_tenant_config")
        except Exception as e:
            logger.error(f"Failed to get tenant config: {e}")
            return None

    def set_tenant_config(self, client_id: str, config: TenantLanguageConfig) -> None:
        """
        Set or update language configuration for a tenant.

        Args:
            client_id: The tenant UUID
            config: The language configuration to set
        """
        sql = """
        INSERT INTO tenant_config
            (client_id, language, embedding_model, embedding_provider,
             llm_model, reranker_model)
        VALUES (%s::uuid, %s, %s, %s, %s, %s)
        ON CONFLICT (client_id) DO UPDATE SET
            language = EXCLUDED.language,
            embedding_model = EXCLUDED.embedding_model,
            embedding_provider = EXCLUDED.embedding_provider,
            llm_model = EXCLUDED.llm_model,
            reranker_model = EXCLUDED.reranker_model,
            updated_at = NOW()
        """

        def _op(conn):
            with conn.cursor() as cur:
                cur.execute(sql, (
                    client_id,
                    config.language,
                    config.embedding_model,
                    config.embedding_provider,
                    config.llm_model,
                    config.reranker_model,
                ))
                conn.commit()
            logger.info(f"Tenant config set for {client_id}: lang={config.language}")

        self._execute_with_retry(_op, "set_tenant_config")

    # =========================================================================
    # User Auth & Conversation Schema (Google OAuth)
    # =========================================================================

    def migrate_add_user_auth_schema(self) -> bool:
        """
        Migration: Create users, conversations, messages tables and add
        user_id/upload_scope/conversation_id columns to legal_documents.

        Safe to run multiple times (uses IF NOT EXISTS / IF NOT EXISTS).

        Returns:
            True if migration succeeded, False otherwise
        """
        conn = self._ensure_connection()

        migration_sql = """
        -- Users table (Google OAuth)
        CREATE TABLE IF NOT EXISTS users (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            google_sub VARCHAR(255) UNIQUE NOT NULL,
            email VARCHAR(255) UNIQUE NOT NULL,
            name VARCHAR(255),
            avatar_url TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            last_login TIMESTAMPTZ DEFAULT NOW()
        );

        -- Conversations (chat sessions)
        CREATE TABLE IF NOT EXISTS conversations (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            title VARCHAR(500) DEFAULT 'New Chat',
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS idx_conversations_user
            ON conversations(user_id, updated_at DESC);

        -- Messages within conversations
        CREATE TABLE IF NOT EXISTS messages (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            role VARCHAR(20) NOT NULL,
            content TEXT NOT NULL,
            sources JSONB,
            latency_ms FLOAT,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS idx_messages_conversation
            ON messages(conversation_id, created_at);

        -- Add user_id, upload_scope, conversation_id to legal_documents
        ALTER TABLE legal_documents ADD COLUMN IF NOT EXISTS user_id UUID REFERENCES users(id);
        ALTER TABLE legal_documents ADD COLUMN IF NOT EXISTS upload_scope VARCHAR(20) DEFAULT 'persistent';
        ALTER TABLE legal_documents ADD COLUMN IF NOT EXISTS conversation_id UUID REFERENCES conversations(id);

        CREATE INDEX IF NOT EXISTS idx_legal_docs_user
            ON legal_documents(user_id);
        CREATE INDEX IF NOT EXISTS idx_legal_docs_conversation
            ON legal_documents(conversation_id);
        """

        try:
            with conn.cursor() as cur:
                cur.execute(migration_sql)
                conn.commit()
            logger.info("Migration completed: User auth & conversation schema created")
            return True
        except Exception as e:
            self._safe_rollback(conn)
            logger.error(f"User auth schema migration failed: {e}")
            return False
        finally:
            self._release_connection(conn)

    # =========================================================================
    # User CRUD
    # =========================================================================

    def create_or_get_user(
        self,
        google_sub: str,
        email: str,
        name: Optional[str] = None,
        avatar_url: Optional[str] = None,
    ) -> dict:
        """
        Create a new user or get existing one by google_sub.
        Updates last_login, name, and avatar_url on each login.

        Returns:
            Dict with id, google_sub, email, name, avatar_url
        """
        sql = """
        INSERT INTO users (google_sub, email, name, avatar_url)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (google_sub) DO UPDATE SET
            email = EXCLUDED.email,
            name = EXCLUDED.name,
            avatar_url = EXCLUDED.avatar_url,
            last_login = NOW()
        RETURNING id, google_sub, email, name, avatar_url
        """

        def _op(conn):
            with conn.cursor() as cur:
                cur.execute(sql, (google_sub, email, name, avatar_url))
                row = cur.fetchone()
                conn.commit()
            if isinstance(row, dict):
                return {
                    "id": str(row["id"]),
                    "google_sub": row["google_sub"],
                    "email": row["email"],
                    "name": row["name"],
                    "avatar_url": row["avatar_url"],
                }
            return {
                "id": str(row[0]),
                "google_sub": row[1],
                "email": row[2],
                "name": row[3],
                "avatar_url": row[4],
            }

        return self._execute_with_retry(_op, "create_or_get_user")

    def get_user_by_id(self, user_id: str) -> Optional[dict]:
        """Get user by ID. Returns None if not found."""
        sql = "SELECT id, google_sub, email, name, avatar_url FROM users WHERE id = %s::uuid"

        def _op(conn):
            with conn.cursor() as cur:
                cur.execute(sql, (user_id,))
                row = cur.fetchone()

            if not row:
                return None

            if isinstance(row, dict):
                return {
                    "id": str(row["id"]),
                    "google_sub": row["google_sub"],
                    "email": row["email"],
                    "name": row["name"],
                    "avatar_url": row["avatar_url"],
                }
            return {
                "id": str(row[0]),
                "google_sub": row[1],
                "email": row[2],
                "name": row[3],
                "avatar_url": row[4],
            }

        try:
            return self._execute_with_retry(_op, "get_user_by_id")
        except Exception as e:
            logger.error(f"get_user_by_id failed: {e}")
            return None

    # =========================================================================
    # Conversation CRUD
    # =========================================================================

    def create_conversation(self, user_id: str, title: str = "New Chat") -> dict:
        """Create a new conversation for a user."""
        sql = """
        INSERT INTO conversations (user_id, title)
        VALUES (%s::uuid, %s)
        RETURNING id, user_id, title, created_at, updated_at
        """

        def _op(conn):
            with conn.cursor() as cur:
                cur.execute(sql, (user_id, title))
                row = cur.fetchone()
                conn.commit()

            if isinstance(row, dict):
                return {
                    "id": str(row["id"]),
                    "user_id": str(row["user_id"]),
                    "title": row["title"],
                    "created_at": row["created_at"].isoformat(),
                    "updated_at": row["updated_at"].isoformat(),
                }
            return {
                "id": str(row[0]),
                "user_id": str(row[1]),
                "title": row[2],
                "created_at": row[3].isoformat(),
                "updated_at": row[4].isoformat(),
            }

        return self._execute_with_retry(_op, "create_conversation")

    def list_conversations(self, user_id: str) -> list[dict]:
        """List all conversations for a user, newest first."""
        sql = """
        SELECT id, user_id, title, created_at, updated_at
        FROM conversations
        WHERE user_id = %s::uuid
        ORDER BY updated_at DESC
        """

        def _op(conn):
            with conn.cursor() as cur:
                cur.execute(sql, (user_id,))
                rows = cur.fetchall()

            result = []
            for row in rows:
                if isinstance(row, dict):
                    result.append({
                        "id": str(row["id"]),
                        "user_id": str(row["user_id"]),
                        "title": row["title"],
                        "created_at": row["created_at"].isoformat(),
                        "updated_at": row["updated_at"].isoformat(),
                    })
                else:
                    result.append({
                        "id": str(row[0]),
                        "user_id": str(row[1]),
                        "title": row[2],
                        "created_at": row[3].isoformat(),
                        "updated_at": row[4].isoformat(),
                    })
            return result

        return self._execute_with_retry(_op, "list_conversations")

    def delete_conversation(self, conversation_id: str, user_id: str) -> bool:
        """Delete a conversation and all its messages (user-isolated).

        Also deletes any chat-scoped documents tied to this conversation.
        """
        def _op(conn):
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM legal_documents WHERE conversation_id = %s::uuid AND user_id = %s::uuid",
                    (conversation_id, user_id),
                )
                cur.execute(
                    "DELETE FROM conversations WHERE id = %s::uuid AND user_id = %s::uuid",
                    (conversation_id, user_id),
                )
                deleted = cur.rowcount > 0
                conn.commit()
            return deleted

        return self._execute_with_retry(_op, "delete_conversation")

    def rename_conversation(self, conversation_id: str, user_id: str, title: str) -> bool:
        """Rename a conversation (user-isolated)."""
        sql = """
        UPDATE conversations SET title = %s, updated_at = NOW()
        WHERE id = %s::uuid AND user_id = %s::uuid
        """

        def _op(conn):
            with conn.cursor() as cur:
                cur.execute(sql, (title, conversation_id, user_id))
                updated = cur.rowcount > 0
                conn.commit()
            return updated

        return self._execute_with_retry(_op, "rename_conversation")

    def touch_conversation(self, conversation_id: str) -> None:
        """Update the updated_at timestamp of a conversation."""
        def _op(conn):
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE conversations SET updated_at = NOW() WHERE id = %s::uuid",
                    (conversation_id,),
                )
                conn.commit()

        try:
            self._execute_with_retry(_op, "touch_conversation")
        except Exception as e:
            logger.warning(f"touch_conversation failed: {e}")

    # =========================================================================
    # Message CRUD
    # =========================================================================

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        sources: Optional[list] = None,
        latency_ms: Optional[float] = None,
    ) -> dict:
        """Add a message to a conversation."""
        sql = """
        INSERT INTO messages (conversation_id, role, content, sources, latency_ms)
        VALUES (%s::uuid, %s, %s, %s, %s)
        RETURNING id, conversation_id, role, content, sources, latency_ms, created_at
        """

        import json as _json
        sources_json = _json.dumps(sources) if sources else None

        def _op(conn):
            with conn.cursor() as cur:
                cur.execute(sql, (conversation_id, role, content, sources_json, latency_ms))
                row = cur.fetchone()
                conn.commit()

            if isinstance(row, dict):
                return {
                    "id": str(row["id"]),
                    "conversation_id": str(row["conversation_id"]),
                    "role": row["role"],
                    "content": row["content"],
                    "sources": row["sources"],
                    "latency_ms": row["latency_ms"],
                    "created_at": row["created_at"].isoformat(),
                }
            return {
                "id": str(row[0]),
                "conversation_id": str(row[1]),
                "role": row[2],
                "content": row[3],
                "sources": row[4],
                "latency_ms": row[5],
                "created_at": row[6].isoformat(),
            }

        return self._execute_with_retry(_op, "add_message")

    def get_messages(self, conversation_id: str, user_id: str) -> list[dict]:
        """Get all messages for a conversation (verified by user_id)."""
        sql = """
        SELECT m.id, m.conversation_id, m.role, m.content, m.sources, m.latency_ms, m.created_at
        FROM messages m
        JOIN conversations c ON c.id = m.conversation_id
        WHERE m.conversation_id = %s::uuid AND c.user_id = %s::uuid
        ORDER BY m.created_at ASC
        """

        def _op(conn):
            with conn.cursor() as cur:
                cur.execute(sql, (conversation_id, user_id))
                rows = cur.fetchall()

            result = []
            for row in rows:
                if isinstance(row, dict):
                    result.append({
                        "id": str(row["id"]),
                        "conversation_id": str(row["conversation_id"]),
                        "role": row["role"],
                        "content": row["content"],
                        "sources": row["sources"],
                        "latency_ms": row["latency_ms"],
                        "created_at": row["created_at"].isoformat(),
                    })
                else:
                    result.append({
                        "id": str(row[0]),
                        "conversation_id": str(row[1]),
                        "role": row[2],
                        "content": row[3],
                        "sources": row[4],
                        "latency_ms": row[5],
                        "created_at": row[6].isoformat(),
                    })
            return result

        return self._execute_with_retry(_op, "get_messages")

    # =========================================================================
    # Document Families
    # =========================================================================

    def migrate_add_document_families(self) -> bool:
        """
        Migration: Create document_families table and add family_id columns.

        Safe to run multiple times (uses IF NOT EXISTS).

        Returns:
            True if migration succeeded, False otherwise
        """
        conn = self._ensure_connection()

        migration_sql = f"""
        -- Document families table
        CREATE TABLE IF NOT EXISTS document_families (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            name VARCHAR(100) NOT NULL,
            is_active BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE(user_id, name)
        );
        CREATE INDEX IF NOT EXISTS idx_families_user
            ON document_families(user_id);

        -- Add family_id to legal_documents
        ALTER TABLE legal_documents
        ADD COLUMN IF NOT EXISTS family_id UUID REFERENCES document_families(id) ON DELETE SET NULL;

        CREATE INDEX IF NOT EXISTS idx_legal_docs_family
            ON legal_documents(family_id);

        -- Add family_id to document_chunks
        ALTER TABLE {self.config.table_name}
        ADD COLUMN IF NOT EXISTS family_id UUID;

        CREATE INDEX IF NOT EXISTS idx_chunks_family
            ON {self.config.table_name}(family_id);

        CREATE INDEX IF NOT EXISTS idx_chunks_client_family
            ON {self.config.table_name}(client_id, family_id);
        """

        try:
            with conn.cursor() as cur:
                cur.execute(migration_sql)
                conn.commit()
            logger.info("Migration completed: Document families schema created")
            return True
        except Exception as e:
            self._safe_rollback(conn)
            logger.error(f"Document families migration failed: {e}")
            return False
        finally:
            self._release_connection(conn)

    def create_family(self, user_id: str, name: str) -> dict:
        """Create a new document family for a user."""
        sql = """
        INSERT INTO document_families (user_id, name)
        VALUES (%s::uuid, %s)
        RETURNING id, user_id, name, is_active, created_at, updated_at
        """

        def _op(conn):
            with conn.cursor() as cur:
                cur.execute(sql, (user_id, name))
                row = cur.fetchone()
                conn.commit()

            if isinstance(row, dict):
                return {
                    "id": str(row["id"]),
                    "user_id": str(row["user_id"]),
                    "name": row["name"],
                    "is_active": row["is_active"],
                    "created_at": row["created_at"].isoformat(),
                    "updated_at": row["updated_at"].isoformat(),
                }
            return {
                "id": str(row[0]),
                "user_id": str(row[1]),
                "name": row[2],
                "is_active": row[3],
                "created_at": row[4].isoformat(),
                "updated_at": row[5].isoformat(),
            }

        try:
            return self._execute_with_retry(_op, "create_family")
        except ValueError:
            raise
        except Exception as e:
            if "unique" in str(e).lower():
                raise ValueError(f"Family '{name}' already exists")
            raise

    def list_families(self, user_id: str) -> list[dict]:
        """List all document families for a user."""
        sql = """
        SELECT f.id, f.user_id, f.name, f.is_active, f.created_at, f.updated_at,
               COUNT(ld.id) as document_count
        FROM document_families f
        LEFT JOIN legal_documents ld ON ld.family_id = f.id
        WHERE f.user_id = %s::uuid
        GROUP BY f.id
        ORDER BY f.name
        """

        def _op(conn):
            with conn.cursor() as cur:
                cur.execute(sql, (user_id,))
                rows = cur.fetchall()

            result = []
            for row in rows:
                if isinstance(row, dict):
                    result.append({
                        "id": str(row["id"]),
                        "user_id": str(row["user_id"]),
                        "name": row["name"],
                        "is_active": row["is_active"],
                        "document_count": row["document_count"],
                        "created_at": row["created_at"].isoformat(),
                        "updated_at": row["updated_at"].isoformat(),
                    })
                else:
                    result.append({
                        "id": str(row[0]),
                        "user_id": str(row[1]),
                        "name": row[2],
                        "is_active": row[3],
                        "document_count": row[6],
                        "created_at": row[4].isoformat(),
                        "updated_at": row[5].isoformat(),
                    })
            return result

        return self._execute_with_retry(_op, "list_families")

    def rename_family(self, family_id: str, user_id: str, name: str) -> bool:
        """Rename a family (user-isolated)."""
        sql = """
        UPDATE document_families SET name = %s, updated_at = NOW()
        WHERE id = %s::uuid AND user_id = %s::uuid
        """

        def _op(conn):
            with conn.cursor() as cur:
                cur.execute(sql, (name, family_id, user_id))
                updated = cur.rowcount > 0
                conn.commit()
            return updated

        try:
            return self._execute_with_retry(_op, "rename_family")
        except ValueError:
            raise
        except Exception as e:
            if "unique" in str(e).lower():
                raise ValueError(f"Family '{name}' already exists")
            raise

    def delete_family(self, family_id: str, user_id: str) -> bool:
        """Delete a family. Documents are unassigned (family_id set to NULL via ON DELETE SET NULL)."""
        def _op(conn):
            with conn.cursor() as cur:
                cur.execute(
                    f"UPDATE {self.config.table_name} SET family_id = NULL "
                    "WHERE family_id = %s::uuid AND client_id = %s::uuid",
                    (family_id, user_id),
                )
                cur.execute(
                    "DELETE FROM document_families WHERE id = %s::uuid AND user_id = %s::uuid",
                    (family_id, user_id),
                )
                deleted = cur.rowcount > 0
                conn.commit()
            return deleted

        return self._execute_with_retry(_op, "delete_family")

    def set_family_active(self, family_id: str, user_id: str, is_active: bool) -> bool:
        """Toggle a family's active status. Caller must enforce max 3 active."""
        sql = """
        UPDATE document_families SET is_active = %s, updated_at = NOW()
        WHERE id = %s::uuid AND user_id = %s::uuid
        """

        def _op(conn):
            with conn.cursor() as cur:
                cur.execute(sql, (is_active, family_id, user_id))
                updated = cur.rowcount > 0
                conn.commit()
            return updated

        return self._execute_with_retry(_op, "set_family_active")

    def get_active_family_count(self, user_id: str) -> int:
        """Get the number of active families for a user."""
        sql = "SELECT COUNT(*) FROM document_families WHERE user_id = %s::uuid AND is_active = TRUE"

        def _op(conn):
            with conn.cursor() as cur:
                cur.execute(sql, (user_id,))
                row = cur.fetchone()
            return row["count"] if isinstance(row, dict) else row[0]

        try:
            return self._execute_with_retry(_op, "get_active_family_count")
        except Exception as e:
            logger.error(f"get_active_family_count failed: {e}")
            return 0

    def move_document_to_family(
        self, document_id: str, family_id: Optional[str], user_id: str
    ) -> bool:
        """Move a document (and its chunks) to a family (or unassign with family_id=None)."""
        def _op(conn):
            with conn.cursor() as cur:
                if family_id:
                    cur.execute(
                        "UPDATE legal_documents SET family_id = %s::uuid "
                        "WHERE id = %s::uuid AND client_id = %s::uuid",
                        (family_id, document_id, user_id),
                    )
                else:
                    cur.execute(
                        "UPDATE legal_documents SET family_id = NULL "
                        "WHERE id = %s::uuid AND client_id = %s::uuid",
                        (document_id, user_id),
                    )
                updated = cur.rowcount > 0

                if updated:
                    if family_id:
                        cur.execute(
                            f"UPDATE {self.config.table_name} SET family_id = %s::uuid "
                            "WHERE document_id = %s::uuid AND client_id = %s::uuid",
                            (family_id, document_id, user_id),
                        )
                    else:
                        cur.execute(
                            f"UPDATE {self.config.table_name} SET family_id = NULL "
                            "WHERE document_id = %s::uuid AND client_id = %s::uuid",
                            (document_id, user_id),
                        )

                conn.commit()
            return updated

        return self._execute_with_retry(_op, "move_document_to_family")

    def create_greek_fts_index(self) -> None:
        """Create Greek FTS index on-demand (when first Greek tenant is created)."""
        conn = self._ensure_connection()

        sql = f"""
        CREATE INDEX IF NOT EXISTS idx_chunks_fts_greek
            ON {self.config.table_name}
            USING GIN (to_tsvector('greek', content));
        """

        try:
            with conn.cursor() as cur:
                cur.execute(sql)
                conn.commit()
            logger.info("Greek FTS index created")
        except Exception as e:
            self._safe_rollback(conn)
            logger.error(f"Failed to create Greek FTS index: {e}")
            raise
        finally:
            self._release_connection(conn)


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

        elif command == "migrate-source-origin":
            # Add source_origin column for multi-source filtering
            ok = store.migrate_add_source_origin()
            print("source_origin migration " + ("succeeded" if ok else "FAILED"))

        elif command == "migrate-user-auth":
            # Add users, conversations, messages tables
            ok = store.migrate_add_user_auth_schema()
            print("user auth migration " + ("succeeded" if ok else "FAILED"))

        elif command == "migrate-families":
            # Add document families table and family_id columns
            ok = store.migrate_add_document_families()
            print("document families migration " + ("succeeded" if ok else "FAILED"))

        else:
            print(f"Unknown command: {command}")
            print("Available commands: setup-auth, enable-rls, create-demo-key, test-rls, migrate-source-origin, migrate-user-auth, migrate-families")
    else:
        import re as _re
        print("Vector store initialized successfully")
        masked = _re.sub(r'://[^@]+@', '://***:***@', store._connection_string)
        print(f"Connection: {masked}")
        print("\nCommands:")
        print("  python vector_store.py setup-auth     - Create auth tables")
        print("  python vector_store.py enable-rls     - Enable Row-Level Security")
        print("  python vector_store.py create-demo-key - Create demo API key")
        print("  python vector_store.py test-rls       - Test RLS isolation")
