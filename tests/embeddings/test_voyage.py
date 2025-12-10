"""Tests for Voyage AI embedding generator."""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

from codesearch.embeddings.voyage import (
    DEFAULT_DIMENSIONS,
    DEFAULT_MAX_LENGTH,
    DEFAULT_MODEL,
    VoyageAPIError,
    VoyageEmbeddingGenerator,
)


@pytest.fixture
def mock_voyageai():
    """Mock the voyageai module for tests."""
    mock_module = MagicMock()
    mock_client = MagicMock()
    mock_module.Client.return_value = mock_client

    with patch.dict(sys.modules, {"voyageai": mock_module}):
        yield mock_module, mock_client


class TestVoyageEmbeddingGeneratorInit:
    """Tests for VoyageEmbeddingGenerator initialization."""

    def test_init_with_api_key(self, mock_voyageai):
        """Test initialization with explicit API key."""
        mock_module, mock_client = mock_voyageai

        generator = VoyageEmbeddingGenerator(api_key="test-key")

        assert generator.api_key == "test-key"
        assert generator.model == DEFAULT_MODEL
        assert generator.dimensions == DEFAULT_DIMENSIONS
        mock_module.Client.assert_called_once_with(api_key="test-key")

    def test_init_from_env_var(self, mock_voyageai):
        """Test initialization from VOYAGE_API_KEY environment variable."""
        with patch.dict(os.environ, {"VOYAGE_API_KEY": "env-key"}):
            generator = VoyageEmbeddingGenerator()
            assert generator.api_key == "env-key"

    def test_init_no_api_key_raises(self):
        """Test that missing API key raises ValueError."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError) as exc_info:
                VoyageEmbeddingGenerator()
            assert "Voyage API key required" in str(exc_info.value)

    def test_init_missing_package_raises(self):
        """Test that missing voyageai package raises ImportError."""
        # Temporarily remove voyageai from modules if present
        with patch.dict(os.environ, {"VOYAGE_API_KEY": "test-key"}):
            with patch.dict(sys.modules, {"voyageai": None}):
                with pytest.raises(ImportError) as exc_info:
                    VoyageEmbeddingGenerator(api_key="test-key")
                assert "voyageai package required" in str(exc_info.value)

    def test_init_custom_model(self, mock_voyageai):
        """Test initialization with custom model."""
        generator = VoyageEmbeddingGenerator(
            api_key="test-key",
            model="custom-model",
            dimensions=512,
        )

        assert generator.model == "custom-model"
        assert generator.dimensions == 512


class TestVoyageEmbeddingGeneratorEmbed:
    """Tests for embedding generation."""

    def test_embed_code_single(self, mock_voyageai):
        """Test embedding a single code snippet."""
        mock_module, mock_client = mock_voyageai

        # Mock the embed response
        mock_result = MagicMock()
        mock_result.embeddings = [[0.1, 0.2, 0.3]]
        mock_result.total_tokens = 10
        mock_client.embed.return_value = mock_result

        generator = VoyageEmbeddingGenerator(api_key="test-key")
        result = generator.embed_code("def hello(): pass")

        assert result == [0.1, 0.2, 0.3]
        mock_client.embed.assert_called_once_with(
            ["def hello(): pass"],
            model=DEFAULT_MODEL,
            input_type="document",
        )

    def test_embed_batch(self, mock_voyageai):
        """Test batch embedding."""
        mock_module, mock_client = mock_voyageai

        mock_result = MagicMock()
        mock_result.embeddings = [[0.1, 0.2], [0.3, 0.4]]
        mock_result.total_tokens = 20
        mock_client.embed.return_value = mock_result

        generator = VoyageEmbeddingGenerator(api_key="test-key")
        result = generator.embed_batch(["code1", "code2"])

        assert result == [[0.1, 0.2], [0.3, 0.4]]

    def test_embed_batch_empty(self, mock_voyageai):
        """Test batch embedding with empty list."""
        generator = VoyageEmbeddingGenerator(api_key="test-key")
        result = generator.embed_batch([])
        assert result == []

    def test_embed_with_retry_on_rate_limit(self, mock_voyageai):
        """Test exponential backoff on rate limit errors."""
        mock_module, mock_client = mock_voyageai

        # First call raises rate limit, second succeeds
        mock_result = MagicMock()
        mock_result.embeddings = [[0.1]]
        mock_result.total_tokens = 5
        mock_client.embed.side_effect = [
            Exception("rate limit exceeded"),
            mock_result,
        ]

        generator = VoyageEmbeddingGenerator(api_key="test-key")

        with patch("time.sleep"):  # Don't actually sleep
            result = generator.embed_code("test")

        assert result == [0.1]
        assert mock_client.embed.call_count == 2

    def test_embed_retry_exhausted(self, mock_voyageai):
        """Test that retries are exhausted after max attempts."""
        mock_module, mock_client = mock_voyageai
        mock_client.embed.side_effect = Exception("rate limit 429")

        generator = VoyageEmbeddingGenerator(api_key="test-key", max_retries=3)

        with patch("time.sleep"):
            with pytest.raises(VoyageAPIError) as exc_info:
                generator.embed_code("test")

        assert "Failed after 3 retries" in str(exc_info.value)

    def test_non_rate_limit_error_raises_immediately(self, mock_voyageai):
        """Test that non-rate-limit errors raise immediately."""
        mock_module, mock_client = mock_voyageai
        mock_client.embed.side_effect = Exception("invalid input")

        generator = VoyageEmbeddingGenerator(api_key="test-key")

        with pytest.raises(VoyageAPIError) as exc_info:
            generator.embed_code("test")

        assert "Voyage API error" in str(exc_info.value)
        assert mock_client.embed.call_count == 1  # No retry


class TestVoyageEmbeddingGeneratorInfo:
    """Tests for model info and usage tracking."""

    def test_get_model_info(self, mock_voyageai):
        """Test getting model metadata."""
        generator = VoyageEmbeddingGenerator(api_key="test-key")
        info = generator.get_model_info()

        assert info["name"] == DEFAULT_MODEL
        assert info["dimensions"] == DEFAULT_DIMENSIONS
        assert info["max_length"] == DEFAULT_MAX_LENGTH
        assert info["device"] == "api"
        assert info["provider"] == "voyage"

    def test_get_usage_stats_initial(self, mock_voyageai):
        """Test usage stats before any requests."""
        generator = VoyageEmbeddingGenerator(api_key="test-key")
        stats = generator.get_usage_stats()

        assert stats["total_tokens"] == 0
        assert stats["total_requests"] == 0
        assert stats["estimated_cost_usd"] == 0.0

    def test_usage_tracking(self, mock_voyageai):
        """Test that usage is tracked after requests."""
        mock_module, mock_client = mock_voyageai

        mock_result = MagicMock()
        mock_result.embeddings = [[0.1]]
        mock_result.total_tokens = 100
        mock_client.embed.return_value = mock_result

        generator = VoyageEmbeddingGenerator(api_key="test-key")
        generator.embed_code("test")

        stats = generator.get_usage_stats()
        assert stats["total_tokens"] == 100
        assert stats["total_requests"] == 1

    def test_estimate_cost(self, mock_voyageai):
        """Test cost estimation."""
        generator = VoyageEmbeddingGenerator(api_key="test-key")
        estimate = generator.estimate_cost(["short", "longer code snippet here"])

        assert estimate["text_count"] == 2
        assert estimate["total_characters"] == 5 + 24  # 29
        assert estimate["estimated_tokens"] == 7  # 29 // 4


class TestVoyageModelRegistry:
    """Tests for Voyage model in registry."""

    def test_voyage_code_3_in_registry(self):
        """Test that voyage-code-3 is in the model registry."""
        from codesearch.embeddings.config import MODEL_REGISTRY

        assert "voyage-code-3" in MODEL_REGISTRY
        config = MODEL_REGISTRY["voyage-code-3"]
        assert config.dimensions == 1024
        assert config.max_length == 16000
        assert config.device == "api"

    def test_get_voyage_config(self):
        """Test getting voyage config via get_model_config."""
        from codesearch.embeddings.config import get_model_config

        config = get_model_config("voyage-code-3")

        assert config.model_name == "voyage-code-3"
        assert config.api_endpoint == "https://api.voyageai.com/v1/embeddings"


class TestEmbeddingGeneratorVoyageIntegration:
    """Tests for EmbeddingGenerator with Voyage model."""

    @pytest.fixture
    def mock_voyage_env(self, mock_voyageai):
        """Set up Voyage environment with mocked module and API key."""
        mock_module, mock_client = mock_voyageai
        with patch.dict(os.environ, {"VOYAGE_API_KEY": "test-api-key"}):
            yield mock_module, mock_client

    def test_generator_detects_api_model(self, mock_voyage_env):
        """Test that EmbeddingGenerator detects API model and delegates."""
        from codesearch.embeddings.generator import EmbeddingGenerator

        generator = EmbeddingGenerator(model_config="voyage-code-3")

        assert generator._api_generator is not None
        assert generator.device == "api"

    def test_generator_delegates_embed_code(self, mock_voyage_env):
        """Test that embed_code delegates to API generator."""
        mock_module, mock_client = mock_voyage_env

        mock_result = MagicMock()
        mock_result.embeddings = [[0.5] * 1024]
        mock_result.total_tokens = 10
        mock_client.embed.return_value = mock_result

        from codesearch.embeddings.generator import EmbeddingGenerator

        generator = EmbeddingGenerator(model_config="voyage-code-3")
        result = generator.embed_code("def test(): pass")

        assert len(result) == 1024
        mock_client.embed.assert_called_once()

    def test_generator_delegates_get_model_info(self, mock_voyage_env):
        """Test that get_model_info delegates to API generator."""
        from codesearch.embeddings.generator import EmbeddingGenerator

        generator = EmbeddingGenerator(model_config="voyage-code-3")
        info = generator.get_model_info()

        assert info["provider"] == "voyage"
        assert info["device"] == "api"
