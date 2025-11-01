"""Tests for embedding model and generator."""

import pytest
from codesearch.models import EmbeddingModel


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
