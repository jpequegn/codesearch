"""Tests for embedding model and generator."""

import pytest
from codesearch.models import EmbeddingModel
from codesearch.embeddings.generator import EmbeddingGenerator


def test_embedding_model_creation():
    """Test creating an EmbeddingModel configuration."""
    model = EmbeddingModel(
        name="codebert-base",
        model_name="microsoft/codebert-base",
        dimensions=768,
        max_length=512,
    )

    assert model.name == "codebert-base"
    assert model.model_name == "microsoft/codebert-base"
    assert model.dimensions == 768
    assert model.max_length == 512
    assert model.device == "auto"  # Default value


def test_embedding_model_custom_device():
    """Test creating an EmbeddingModel with custom device."""
    model = EmbeddingModel(
        name="codebert-base",
        model_name="microsoft/codebert-base",
        dimensions=768,
        max_length=512,
        device="cpu",
    )

    assert model.device == "cpu"


def test_embedding_generator_initialization():
    """Test initializing EmbeddingGenerator with default CodeBERT model."""
    generator = EmbeddingGenerator()

    assert generator is not None
    assert generator.model is not None
    assert generator.tokenizer is not None
    assert generator.device is not None


def test_embedding_generator_with_custom_model():
    """Test initializing EmbeddingGenerator with custom model config."""
    model_config = EmbeddingModel(
        name="codebert-base",
        model_name="microsoft/codebert-base",
        dimensions=768,
        max_length=512,
        device="cpu",
    )

    generator = EmbeddingGenerator(model_config)
    assert generator.model_config == model_config
    assert generator.device == "cpu"
