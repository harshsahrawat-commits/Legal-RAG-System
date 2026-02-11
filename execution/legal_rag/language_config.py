"""
Language Configuration for Multilingual Legal RAG

Provides per-tenant language configuration supporting English and Greek.
Each tenant can have different language settings for LLM, embeddings,
reranking, and full-text search.
"""

from dataclasses import dataclass
from typing import Optional


# Supported languages with their PostgreSQL FTS config names and token ratios
SUPPORTED_LANGUAGES = {
    "en": {
        "name": "English",
        "fts_config": "english",
        "chars_per_token": 4,
    },
    "el": {
        "name": "Greek",
        "fts_config": "greek",
        "chars_per_token": 3,
    },
}

# Whitelist of valid FTS language configs (for SQL injection prevention)
VALID_FTS_CONFIGS = frozenset(lang["fts_config"] for lang in SUPPORTED_LANGUAGES.values())


@dataclass
class TenantLanguageConfig:
    """Per-tenant language and model configuration."""
    language: str = "en"
    embedding_model: str = "voyage-law-2"
    embedding_provider: str = "voyage"
    llm_model: str = "qwen/qwen3-235b-a22b"
    reranker_model: str = "rerank-multilingual-v3.0"
    fts_language: str = "english"
    chars_per_token: int = 4

    @classmethod
    def for_language(cls, language: str) -> "TenantLanguageConfig":
        """
        Factory method returning defaults for a given language.

        Args:
            language: ISO 639-1 code ("en" or "el")

        Returns:
            TenantLanguageConfig with appropriate defaults
        """
        if language == "el":
            return cls(
                language="el",
                embedding_model="voyage-multilingual-2",
                embedding_provider="voyage",
                llm_model="qwen/qwen3-235b-a22b",
                reranker_model="rerank-multilingual-v3.0",
                fts_language="greek",
                chars_per_token=3,
            )

        # Default: English
        if language not in SUPPORTED_LANGUAGES:
            language = "en"

        return cls(
            language="en",
            embedding_model="voyage-law-2",
            embedding_provider="voyage",
            llm_model="qwen/qwen3-235b-a22b",
            reranker_model="rerank-multilingual-v3.0",
            fts_language="english",
            chars_per_token=4,
        )

    def validate_fts_language(self) -> bool:
        """Check that fts_language is in the whitelist."""
        return self.fts_language in VALID_FTS_CONFIGS
