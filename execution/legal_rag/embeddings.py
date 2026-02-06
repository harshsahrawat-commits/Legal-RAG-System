"""
Embedding Service for Legal RAG

Provides embeddings via Voyage AI (voyage-law-2) or Cohere with fallback options.
Supports batching, caching, and different input types (documents vs queries).

voyage-law-2: Optimized for legal documents, 6-10% better retrieval on legal benchmarks.
"""

import os
import json
import hashlib
import logging
from typing import Optional
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Configuration for embedding service."""
    # Default to voyage-law-2 for legal documents
    provider: str = "voyage"  # "voyage" or "cohere"
    model: str = "voyage-law-2"  # voyage-law-2 or embed-english-v3.0
    dimensions: int = 1024
    batch_size: int = 128  # Voyage supports up to 128
    cache_dir: Optional[str] = None
    use_cache: bool = True


class EmbeddingService:
    """
    Generates embeddings for legal document chunks.

    Uses Cohere's embed-v3 model which provides:
    - 1024-dimensional embeddings
    - Different input types for documents vs queries
    - High quality for legal/professional text
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """
        Initialize embedding service.

        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        self.config = config or EmbeddingConfig()
        self._client = None
        self._cache = {}

        # Set up cache directory
        if self.config.cache_dir:
            self._cache_path = Path(self.config.cache_dir)
            self._cache_path.mkdir(parents=True, exist_ok=True)
        else:
            self._cache_path = None

        # Initialize Cohere client
        self._init_client()

    def _init_client(self):
        """Initialize the Cohere client."""
        api_key = os.getenv("COHERE_API_KEY")

        if not api_key:
            logger.warning(
                "COHERE_API_KEY not found. Embeddings will fail. "
                "Set the environment variable or use a different provider."
            )
            return

        try:
            import cohere
            self._client = cohere.Client(api_key)
            logger.info(f"Cohere client initialized with model {self.config.model}")
        except ImportError:
            logger.error("Cohere package not installed. Run: pip install cohere")
            raise

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for document chunks.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        if not self._client:
            raise RuntimeError("Cohere client not initialized. Check COHERE_API_KEY.")

        embeddings = []
        total_batches = (len(texts) + self.config.batch_size - 1) // self.config.batch_size

        logger.info(f"Embedding {len(texts)} documents in {total_batches} batches")

        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]
            batch_embeddings = self._embed_batch(batch, input_type="search_document")
            embeddings.extend(batch_embeddings)

            if (i // self.config.batch_size + 1) % 10 == 0:
                logger.info(f"Processed batch {i // self.config.batch_size + 1}/{total_batches}")

        return embeddings

    def embed_query(self, query: str) -> list[float]:
        """
        Generate embedding for a search query.

        Uses different input_type for better query-document matching.

        Args:
            query: Search query string

        Returns:
            Embedding vector
        """
        if not self._client:
            raise RuntimeError("Cohere client not initialized. Check COHERE_API_KEY.")

        # Check cache
        cache_key = self._get_cache_key(query, "query")
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        result = self._embed_batch([query], input_type="search_query")

        # Cache result
        if result:
            self._set_cached(cache_key, result[0])
            return result[0]

        return []

    def _embed_batch(
        self,
        texts: list[str],
        input_type: str = "search_document"
    ) -> list[list[float]]:
        """Embed a batch of texts."""
        # Check cache for each text
        results = []
        uncached_texts = []
        uncached_indices = []

        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text, input_type)
            cached = self._get_cached(cache_key)
            if cached is not None:
                results.append((i, cached))
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        # Get embeddings for uncached texts
        if uncached_texts:
            try:
                response = self._client.embed(
                    texts=uncached_texts,
                    model=self.config.model,
                    input_type=input_type,
                )

                # Cache and collect results
                for idx, embedding in zip(uncached_indices, response.embeddings):
                    text = texts[idx]
                    cache_key = self._get_cache_key(text, input_type)
                    self._set_cached(cache_key, embedding)
                    results.append((idx, embedding))

            except Exception as e:
                logger.error(f"Embedding failed: {e}")
                raise

        # Sort by original index and return embeddings
        results.sort(key=lambda x: x[0])
        return [emb for _, emb in results]

    def _get_cache_key(self, text: str, input_type: str) -> str:
        """Generate cache key for text."""
        content = f"{self.config.model}:{input_type}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()[:32]

    def _get_cached(self, key: str) -> Optional[list[float]]:
        """Get cached embedding."""
        if not self.config.use_cache:
            return None

        # Check memory cache
        if key in self._cache:
            return self._cache[key]

        # Check file cache
        if self._cache_path:
            cache_file = self._cache_path / f"{key}.json"
            if cache_file.exists():
                try:
                    with open(cache_file) as f:
                        embedding = json.load(f)
                        self._cache[key] = embedding
                        return embedding
                except Exception:
                    pass

        return None

    def _set_cached(self, key: str, embedding: list[float]) -> None:
        """Cache an embedding."""
        if not self.config.use_cache:
            return

        # Memory cache
        self._cache[key] = embedding

        # File cache
        if self._cache_path:
            cache_file = self._cache_path / f"{key}.json"
            try:
                with open(cache_file, 'w') as f:
                    json.dump(embedding, f)
            except Exception as e:
                logger.warning(f"Failed to cache embedding: {e}")

    @property
    def dimensions(self) -> int:
        """Return embedding dimensions."""
        return self.config.dimensions


class VoyageEmbeddingService:
    """
    Embedding service using Voyage AI's voyage-law-2 model.

    voyage-law-2 provides:
    - 1024-dimensional embeddings
    - 6-10% better retrieval on legal benchmarks vs general models
    - 50M free tokens, then $0.22/1M tokens
    - Different input types for documents vs queries
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """Initialize Voyage AI embedding service."""
        self.config = config or EmbeddingConfig()
        self._client = None
        self._cache = {}

        # Set up cache directory
        if self.config.cache_dir:
            self._cache_path = Path(self.config.cache_dir)
            self._cache_path.mkdir(parents=True, exist_ok=True)
        else:
            self._cache_path = None

        self._init_client()

    def _init_client(self):
        """Initialize the Voyage AI client."""
        api_key = os.getenv("VOYAGE_API_KEY")

        if not api_key:
            logger.warning(
                "VOYAGE_API_KEY not found. Embeddings will fail. "
                "Get your free API key at https://dash.voyageai.com/"
            )
            return

        try:
            import voyageai
            self._client = voyageai.Client(api_key=api_key)
            logger.info(f"Voyage AI client initialized with model {self.config.model}")
        except ImportError:
            logger.error("voyageai package not installed. Run: pip install voyageai")
            raise

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for document chunks.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        if not self._client:
            raise RuntimeError("Voyage AI client not initialized. Check VOYAGE_API_KEY.")

        embeddings = []
        total_batches = (len(texts) + self.config.batch_size - 1) // self.config.batch_size

        logger.info(f"Embedding {len(texts)} documents in {total_batches} batches with voyage-law-2")

        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]
            batch_embeddings = self._embed_batch(batch, input_type="document")
            embeddings.extend(batch_embeddings)

            if (i // self.config.batch_size + 1) % 10 == 0:
                logger.info(f"Processed batch {i // self.config.batch_size + 1}/{total_batches}")

        return embeddings

    def embed_query(self, query: str) -> list[float]:
        """
        Generate embedding for a search query.

        Uses different input_type for better query-document matching.

        Args:
            query: Search query string

        Returns:
            Embedding vector
        """
        if not self._client:
            raise RuntimeError("Voyage AI client not initialized. Check VOYAGE_API_KEY.")

        # Check cache
        cache_key = self._get_cache_key(query, "query")
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        result = self._embed_batch([query], input_type="query")

        # Cache result
        if result:
            self._set_cached(cache_key, result[0])
            return result[0]

        return []

    def _embed_batch(
        self,
        texts: list[str],
        input_type: str = "document"
    ) -> list[list[float]]:
        """Embed a batch of texts using Voyage AI."""
        # Check cache for each text
        results = []
        uncached_texts = []
        uncached_indices = []

        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text, input_type)
            cached = self._get_cached(cache_key)
            if cached is not None:
                results.append((i, cached))
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        # Get embeddings for uncached texts
        if uncached_texts:
            try:
                response = self._client.embed(
                    texts=uncached_texts,
                    model=self.config.model,
                    input_type=input_type,
                )

                # Cache and collect results
                for idx, embedding in zip(uncached_indices, response.embeddings):
                    text = texts[idx]
                    cache_key = self._get_cache_key(text, input_type)
                    self._set_cached(cache_key, embedding)
                    results.append((idx, embedding))

            except Exception as e:
                logger.error(f"Voyage embedding failed: {e}")
                raise

        # Sort by original index and return embeddings
        results.sort(key=lambda x: x[0])
        return [emb for _, emb in results]

    def _get_cache_key(self, text: str, input_type: str) -> str:
        """Generate cache key for text."""
        content = f"{self.config.model}:{input_type}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()[:32]

    def _get_cached(self, key: str) -> Optional[list[float]]:
        """Get cached embedding."""
        if not self.config.use_cache:
            return None

        # Check memory cache
        if key in self._cache:
            return self._cache[key]

        # Check file cache
        if self._cache_path:
            cache_file = self._cache_path / f"{key}.json"
            if cache_file.exists():
                try:
                    with open(cache_file) as f:
                        embedding = json.load(f)
                        self._cache[key] = embedding
                        return embedding
                except Exception:
                    pass

        return None

    def _set_cached(self, key: str, embedding: list[float]) -> None:
        """Cache an embedding."""
        if not self.config.use_cache:
            return

        # Memory cache
        self._cache[key] = embedding

        # File cache
        if self._cache_path:
            cache_file = self._cache_path / f"{key}.json"
            try:
                with open(cache_file, 'w') as f:
                    json.dump(embedding, f)
            except Exception as e:
                logger.warning(f"Failed to cache embedding: {e}")

    @property
    def dimensions(self) -> int:
        """Return embedding dimensions."""
        return self.config.dimensions


class LocalEmbeddingService:
    """
    Alternative embedding service using local models.

    Uses sentence-transformers with BGE-M3 for cost-free embeddings.
    Good for development or high-volume batch processing.
    """

    def __init__(self, model_name: str = "BAAI/bge-m3"):
        """Initialize with a local model."""
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name)
            self._dimensions = self._model.get_sentence_embedding_dimension()
            logger.info(f"Local embedding model loaded: {model_name}")
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed documents using local model."""
        embeddings = self._model.encode(texts, show_progress_bar=True)
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Embed a query using local model."""
        embedding = self._model.encode([query])
        return embedding[0].tolist()

    @property
    def dimensions(self) -> int:
        """Return embedding dimensions."""
        return self._dimensions


def get_embedding_service(
    provider: str = "voyage",
    use_local: bool = False
) -> VoyageEmbeddingService | EmbeddingService | LocalEmbeddingService:
    """
    Factory function to get appropriate embedding service.

    Args:
        provider: "voyage" (default, best for legal) or "cohere"
        use_local: If True, use local BGE-M3 model instead of API

    Returns:
        Configured embedding service
    """
    if use_local:
        return LocalEmbeddingService()

    if provider == "voyage":
        config = EmbeddingConfig(
            provider="voyage",
            model="voyage-law-2",
            dimensions=1024,
            batch_size=128,
        )
        return VoyageEmbeddingService(config)

    # Fallback to Cohere
    config = EmbeddingConfig(
        provider="cohere",
        model="embed-english-v3.0",
        dimensions=1024,
        batch_size=96,
    )
    return EmbeddingService(config)


# CLI for testing
if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    # Use voyage-law-2 by default (best for legal documents)
    provider = os.getenv("EMBEDDING_PROVIDER", "voyage")
    print(f"Using embedding provider: {provider}")

    service = get_embedding_service(provider=provider)

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "What are the termination clauses in this contract?"

    print(f"Query: {query}")
    embedding = service.embed_query(query)
    print(f"Embedding dimensions: {len(embedding)}")
    print(f"First 10 values: {embedding[:10]}")
