"""
Embedding Service for Legal RAG

Provides embeddings via Voyage AI (voyage-law-2) or Cohere with fallback options.
Supports batching, caching, and different input types (documents vs queries).

voyage-law-2: Optimized for legal documents, 6-10% better retrieval on legal benchmarks.

Architecture:
    BaseEmbeddingService  -- shared caching, batching, embed_documents, embed_query
        EmbeddingService          -- Cohere embed-v3 provider
        VoyageEmbeddingService    -- Voyage AI voyage-law-2 provider
    LocalEmbeddingService -- local sentence-transformers (no caching needed)
"""

import os
import json
import hashlib
import logging
from typing import Optional, Union
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
    max_tokens_per_batch: int = 100000  # Conservative limit (Voyage max: 120K)
    chars_per_token: float = 4.0  # ~4 for English, ~3 for Greek
    cache_dir: Optional[str] = None
    use_cache: bool = True


class BaseEmbeddingService:
    """
    Base class for API-based embedding services.

    Provides shared functionality:
    - Batched embedding with progress logging
    - Memory and file-based caching
    - Cache key generation
    - Document vs query input type distinction

    Subclasses only need to implement:
    - _init_client(): Initialize the provider-specific API client

    And set these class attributes:
    - _provider_name: Human-readable provider name for error messages
    - _env_var_name: Environment variable name for the API key
    - _doc_input_type: Input type string for document embeddings
    - _query_input_type: Input type string for query embeddings
    """

    # Subclasses must override these
    _provider_name: str = "Base"
    _env_var_name: str = ""
    _doc_input_type: str = "document"
    _query_input_type: str = "query"

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

        # Initialize provider-specific client
        self._init_client()

    def _init_client(self):
        """Initialize the provider-specific API client. Must be overridden by subclasses."""
        raise NotImplementedError("Subclasses must implement _init_client()")

    def _create_batches(self, texts: list[str]) -> list[list[str]]:
        """Split texts into batches respecting both item count and token limits."""
        batches = []
        current_batch = []
        current_tokens = 0
        cpt = self.config.chars_per_token

        for text in texts:
            est_tokens = len(text) / cpt
            # Start new batch if adding this text would exceed limits
            if current_batch and (
                len(current_batch) >= self.config.batch_size
                or current_tokens + est_tokens > self.config.max_tokens_per_batch
            ):
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0
            current_batch.append(text)
            current_tokens += est_tokens

        if current_batch:
            batches.append(current_batch)

        return batches

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
            raise RuntimeError(
                f"{self._provider_name} client not initialized. "
                f"Check {self._env_var_name}."
            )

        batches = self._create_batches(texts)

        logger.info(
            f"Embedding {len(texts)} documents in {len(batches)} batches"
            f" with {self._provider_name}"
        )

        embeddings = []
        for batch_idx, batch in enumerate(batches):
            batch_embeddings = self._embed_batch(batch, input_type=self._doc_input_type)
            embeddings.extend(batch_embeddings)

            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Processed batch {batch_idx + 1}/{len(batches)}")

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
            raise RuntimeError(
                f"{self._provider_name} client not initialized. "
                f"Check {self._env_var_name}."
            )

        # Check cache
        cache_key = self._get_cache_key(query, "query")
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        result = self._embed_batch([query], input_type=self._query_input_type)

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
        """Embed a batch of texts using the provider API."""
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
                logger.error(f"{self._provider_name} embedding failed: {e}")
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
                except Exception as e:
                    logger.debug(f"Failed to read embedding cache file {cache_file}: {e}")

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


class EmbeddingService(BaseEmbeddingService):
    """
    Generates embeddings using Cohere's embed-v3 model.

    Cohere embed-v3 provides:
    - 1024-dimensional embeddings
    - Different input types for documents vs queries
    - High quality for legal/professional text
    """

    _provider_name = "Cohere"
    _env_var_name = "COHERE_API_KEY"
    _doc_input_type = "search_document"
    _query_input_type = "search_query"

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


class VoyageEmbeddingService(BaseEmbeddingService):
    """
    Embedding service using Voyage AI's voyage-law-2 model.

    voyage-law-2 provides:
    - 1024-dimensional embeddings
    - 6-10% better retrieval on legal benchmarks vs general models
    - 50M free tokens, then $0.22/1M tokens
    - Different input types for documents vs queries
    """

    _provider_name = "Voyage AI"
    _env_var_name = "VOYAGE_API_KEY"
    _doc_input_type = "document"
    _query_input_type = "query"

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
    use_local: bool = False,
    language_config=None,
) -> Union[VoyageEmbeddingService, EmbeddingService, LocalEmbeddingService]:
    """
    Factory function to get appropriate embedding service.

    Args:
        provider: "voyage" (default, best for legal) or "cohere"
        use_local: If True, use local BGE-M3 model instead of API
        language_config: Optional TenantLanguageConfig for multilingual support

    Returns:
        Configured embedding service
    """
    if use_local:
        return LocalEmbeddingService()

    # If language config provided, use its model and provider settings
    if language_config is not None:
        model = language_config.embedding_model
        prov = language_config.embedding_provider
    else:
        model = None
        prov = provider

    if prov == "voyage":
        # Voyage tokenizer is more aggressive than LLM tokenizers:
        # Greek can be ~1.0-1.2 chars/token, English ~2-3. Use conservative estimates.
        cpt = 1.0 if (language_config and language_config.chars_per_token <= 3) else 2.0
        config = EmbeddingConfig(
            provider="voyage",
            model=model or "voyage-law-2",
            dimensions=1024,
            batch_size=128,
            chars_per_token=cpt,
        )
        return VoyageEmbeddingService(config)

    # Fallback to Cohere
    config = EmbeddingConfig(
        provider="cohere",
        model=model or "embed-english-v3.0",
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
