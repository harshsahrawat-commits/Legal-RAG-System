"""Tests for TenantLanguageConfig and language configuration."""

import pytest

from execution.legal_rag.language_config import (
    TenantLanguageConfig,
    SUPPORTED_LANGUAGES,
    VALID_FTS_CONFIGS,
)


class TestTenantLanguageConfig:
    """Tests for the TenantLanguageConfig dataclass and factory."""

    def test_english_defaults(self):
        config = TenantLanguageConfig.for_language("en")
        assert config.language == "en"
        assert config.embedding_model == "voyage-law-2"
        assert config.embedding_provider == "voyage"
        assert config.llm_model == "qwen/qwen3-235b-a22b"
        assert config.reranker_model == "rerank-multilingual-v3.0"
        assert config.fts_language == "english"
        assert config.chars_per_token == 4

    def test_greek_defaults(self):
        config = TenantLanguageConfig.for_language("el")
        assert config.language == "el"
        assert config.embedding_model == "voyage-multilingual-2"
        assert config.embedding_provider == "voyage"
        assert config.llm_model == "qwen/qwen3-235b-a22b"
        assert config.reranker_model == "rerank-multilingual-v3.0"
        assert config.fts_language == "greek"
        assert config.chars_per_token == 3

    def test_unsupported_language_falls_back_to_english(self):
        config = TenantLanguageConfig.for_language("xx")
        assert config.language == "en"
        assert config.fts_language == "english"

    def test_validate_fts_language_valid(self):
        config = TenantLanguageConfig.for_language("en")
        assert config.validate_fts_language() is True

    def test_validate_fts_language_invalid(self):
        config = TenantLanguageConfig(fts_language="french")
        assert config.validate_fts_language() is False


class TestSupportedLanguages:
    """Tests for the SUPPORTED_LANGUAGES constant."""

    def test_english_in_supported(self):
        assert "en" in SUPPORTED_LANGUAGES
        assert SUPPORTED_LANGUAGES["en"]["fts_config"] == "english"

    def test_greek_in_supported(self):
        assert "el" in SUPPORTED_LANGUAGES
        assert SUPPORTED_LANGUAGES["el"]["fts_config"] == "greek"

    def test_valid_fts_configs_frozenset(self):
        assert isinstance(VALID_FTS_CONFIGS, frozenset)
        assert "english" in VALID_FTS_CONFIGS
        assert "greek" in VALID_FTS_CONFIGS
