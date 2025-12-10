"""Tests for embedding configuration system."""

import os
from unittest.mock import Mock, patch

import pytest

from codesearch.embeddings.config import (
    DEFAULT_MODEL_NAME,
    MODEL_REGISTRY,
    EmbeddingConfig,
    PoolingStrategy,
    get_available_models,
    get_embedding_config,
    get_model_config,
)


class TestEmbeddingConfig:
    """Tests for EmbeddingConfig dataclass."""

    def test_create_basic_config(self):
        """Test creating a basic config."""
        config = EmbeddingConfig(
            model_name="test-model",
            model_path="test/model",
            dimensions=768,
            max_length=512,
        )
        assert config.model_name == "test-model"
        assert config.model_path == "test/model"
        assert config.dimensions == 768
        assert config.max_length == 512
        assert config.device == "auto"
        assert config.pooling == PoolingStrategy.MEAN
        assert config.api_key is None

    def test_config_with_all_options(self):
        """Test creating a config with all options."""
        config = EmbeddingConfig(
            model_name="test-model",
            model_path="test/model",
            dimensions=256,
            max_length=1024,
            device="cuda",
            pooling=PoolingStrategy.CLS,
            api_key="test-key",
            api_endpoint="https://api.test.com",
        )
        assert config.device == "cuda"
        assert config.pooling == PoolingStrategy.CLS
        assert config.api_key == "test-key"
        assert config.api_endpoint == "https://api.test.com"

    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = EmbeddingConfig(
            model_name="test-model",
            model_path="test/model",
            dimensions=768,
            max_length=512,
            pooling=PoolingStrategy.MEAN,
        )
        result = config.to_dict()
        assert result["model_name"] == "test-model"
        assert result["model_path"] == "test/model"
        assert result["dimensions"] == 768
        assert result["pooling"] == "mean"

    def test_from_dict(self):
        """Test creating config from dictionary."""
        data = {
            "model_name": "test-model",
            "model_path": "test/model",
            "dimensions": 768,
            "max_length": 512,
            "device": "cpu",
            "pooling": "cls",
        }
        config = EmbeddingConfig.from_dict(data)
        assert config.model_name == "test-model"
        assert config.device == "cpu"
        assert config.pooling == PoolingStrategy.CLS


class TestModelRegistry:
    """Tests for model registry."""

    def test_codebert_in_registry(self):
        """Test that CodeBERT is in the registry."""
        assert "codebert" in MODEL_REGISTRY
        config = MODEL_REGISTRY["codebert"]
        assert config.model_path == "microsoft/codebert-base"
        assert config.dimensions == 768

    def test_unixcoder_in_registry(self):
        """Test that UniXcoder is in the registry."""
        assert "unixcoder" in MODEL_REGISTRY
        config = MODEL_REGISTRY["unixcoder"]
        assert config.model_path == "microsoft/unixcoder-base"

    def test_codet5p_in_registry(self):
        """Test that CodeT5+ models are in the registry."""
        assert "codet5p-110m" in MODEL_REGISTRY
        assert "codet5p-220m" in MODEL_REGISTRY

    def test_get_available_models(self):
        """Test getting list of available models."""
        models = get_available_models()
        assert "codebert" in models
        assert "unixcoder" in models
        assert len(models) >= 4  # At least our core models


class TestGetModelConfig:
    """Tests for get_model_config function."""

    def test_get_valid_model(self):
        """Test getting a valid model config."""
        config = get_model_config("codebert")
        assert config.model_name == "codebert"
        assert config.model_path == "microsoft/codebert-base"

    def test_get_invalid_model_raises(self):
        """Test that invalid model raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            get_model_config("nonexistent-model")
        assert "Unknown model" in str(exc_info.value)
        assert "nonexistent-model" in str(exc_info.value)


class TestGetEmbeddingConfig:
    """Tests for get_embedding_config function."""

    def test_default_model(self):
        """Test that default model is used when nothing specified."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("codesearch.embeddings.config.get_config_file", return_value=None):
                config = get_embedding_config()
                assert config.model_name == DEFAULT_MODEL_NAME

    def test_explicit_model_argument(self):
        """Test that explicit model argument takes priority."""
        config = get_embedding_config(model="unixcoder")
        assert config.model_name == "unixcoder"
        assert config.model_path == "microsoft/unixcoder-base"

    def test_explicit_device_argument(self):
        """Test that explicit device argument takes priority."""
        config = get_embedding_config(model="codebert", device="cuda")
        assert config.device == "cuda"

    def test_env_var_model(self):
        """Test that CODESEARCH_MODEL environment variable works."""
        with patch.dict(os.environ, {"CODESEARCH_MODEL": "unixcoder"}, clear=True):
            with patch("codesearch.embeddings.config.get_config_file", return_value=None):
                config = get_embedding_config()
                assert config.model_name == "unixcoder"

    def test_env_var_device(self):
        """Test that CODESEARCH_EMBEDDING_DEVICE environment variable works."""
        with patch.dict(os.environ, {"CODESEARCH_EMBEDDING_DEVICE": "cpu"}, clear=True):
            with patch("codesearch.embeddings.config.get_config_file", return_value=None):
                config = get_embedding_config()
                assert config.device == "cpu"

    def test_explicit_overrides_env_var(self):
        """Test that explicit argument overrides environment variable."""
        with patch.dict(os.environ, {"CODESEARCH_MODEL": "codebert"}, clear=True):
            config = get_embedding_config(model="unixcoder")
            assert config.model_name == "unixcoder"

    def test_config_file_model(self):
        """Test that config file model setting works."""
        mock_config = {"embedding": {"model": "graphcodebert"}}
        with patch.dict(os.environ, {}, clear=True):
            with patch("codesearch.embeddings.config.get_config_file", return_value="/fake/path"):
                with patch("codesearch.embeddings.config.load_config_file", return_value=mock_config):
                    config = get_embedding_config()
                    assert config.model_name == "graphcodebert"

    def test_env_var_overrides_config_file(self):
        """Test that environment variable overrides config file."""
        mock_config = {"embedding": {"model": "graphcodebert"}}
        with patch.dict(os.environ, {"CODESEARCH_MODEL": "unixcoder"}, clear=True):
            with patch("codesearch.embeddings.config.get_config_file", return_value="/fake/path"):
                with patch("codesearch.embeddings.config.load_config_file", return_value=mock_config):
                    config = get_embedding_config()
                    assert config.model_name == "unixcoder"

    def test_api_key_from_env(self):
        """Test that API key is read from environment variable."""
        with patch.dict(
            os.environ,
            {"CODESEARCH_EMBEDDING_API_KEY": "secret-key"},
            clear=True
        ):
            with patch("codesearch.embeddings.config.get_config_file", return_value=None):
                config = get_embedding_config()
                assert config.api_key == "secret-key"


class TestPoolingStrategy:
    """Tests for PoolingStrategy enum."""

    def test_mean_value(self):
        """Test MEAN pooling value."""
        assert PoolingStrategy.MEAN.value == "mean"

    def test_cls_value(self):
        """Test CLS pooling value."""
        assert PoolingStrategy.CLS.value == "cls"

    def test_from_string(self):
        """Test creating from string."""
        assert PoolingStrategy("mean") == PoolingStrategy.MEAN
        assert PoolingStrategy("cls") == PoolingStrategy.CLS
