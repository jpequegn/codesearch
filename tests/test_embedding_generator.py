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


def test_embed_code_simple():
    """Test embedding a simple code snippet."""
    generator = EmbeddingGenerator()

    code = "def hello():\n    return 'world'"
    embedding = generator.embed_code(code)

    # Should return 768-dimensional vector
    assert isinstance(embedding, list)
    assert len(embedding) == 768
    # Values should be floats in roughly [-1, 1] range after normalization
    assert all(isinstance(x, float) for x in embedding)


def test_embed_code_consistency():
    """Test that embedding same code twice gives same result."""
    generator = EmbeddingGenerator()

    code = "def add(a, b):\n    return a + b"
    embedding1 = generator.embed_code(code)
    embedding2 = generator.embed_code(code)

    # Should be identical (deterministic)
    assert embedding1 == embedding2


def test_embed_code_different_inputs():
    """Test that different code produces different embeddings."""
    generator = EmbeddingGenerator()

    code1 = "def add(a, b):\n    return a + b"
    code2 = "def multiply(a, b):\n    return a * b"

    embedding1 = generator.embed_code(code1)
    embedding2 = generator.embed_code(code2)

    # Should be different
    assert embedding1 != embedding2


def test_embed_code_empty():
    """Test embedding empty code string."""
    generator = EmbeddingGenerator()

    embedding = generator.embed_code("")

    # Should still return valid embedding
    assert isinstance(embedding, list)
    assert len(embedding) == 768


def test_embed_code_long_truncation():
    """Test that very long code gets truncated to max_length."""
    generator = EmbeddingGenerator()

    # Create code longer than max_length tokens
    long_code = "def func():\n    " + "x = 1\n    " * 500

    # Should not raise error, should truncate
    embedding = generator.embed_code(long_code)
    assert isinstance(embedding, list)
    assert len(embedding) == 768


def test_embed_batch_basic():
    """Test batch embedding generation."""
    generator = EmbeddingGenerator()

    codes = [
        "def add(a, b):\n    return a + b",
        "def subtract(a, b):\n    return a - b",
        "def multiply(a, b):\n    return a * b",
    ]

    embeddings = generator.embed_batch(codes)

    # Should return list of embeddings
    assert isinstance(embeddings, list)
    assert len(embeddings) == 3
    # Each embedding should be 768-dimensional
    for emb in embeddings:
        assert isinstance(emb, list)
        assert len(emb) == 768


def test_embed_batch_single_item():
    """Test batch embedding with single item."""
    generator = EmbeddingGenerator()

    codes = ["def foo(): pass"]
    embeddings = generator.embed_batch(codes)

    assert len(embeddings) == 1
    assert len(embeddings[0]) == 768


def test_embed_batch_empty_list():
    """Test batch embedding with empty list."""
    generator = EmbeddingGenerator()

    embeddings = generator.embed_batch([])

    assert embeddings == []


def test_embed_batch_consistency_with_single():
    """Test that batch results match single embedding results."""
    generator = EmbeddingGenerator()

    code = "def hello(): return 'world'"

    # Get single embedding
    single_emb = generator.embed_code(code)

    # Get batch embedding
    batch_embs = generator.embed_batch([code])

    # Should match
    assert single_emb == batch_embs[0]
