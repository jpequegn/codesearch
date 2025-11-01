"""Tests for batch embedding generator."""

import pytest
from codesearch.models import Function, Class
from codesearch.embeddings.generator import EmbeddingGenerator
from codesearch.embeddings.text_preparator import TextPreparator
from codesearch.embeddings.batch_generator import BatchEmbeddingGenerator


def test_batch_generator_initialization():
    """Test creating a BatchEmbeddingGenerator."""
    generator = EmbeddingGenerator()
    preparator = TextPreparator(generator.tokenizer, max_tokens=512)
    batch_gen = BatchEmbeddingGenerator(generator, preparator)

    assert batch_gen is not None
    assert batch_gen.embedding_generator is not None
    assert batch_gen.text_preparator is not None
    assert isinstance(batch_gen.cache, dict)


def test_batch_generator_has_required_methods():
    """Test that BatchEmbeddingGenerator has all required methods."""
    generator = EmbeddingGenerator()
    preparator = TextPreparator(generator.tokenizer, max_tokens=512)
    batch_gen = BatchEmbeddingGenerator(generator, preparator)

    # Verify all methods exist
    assert hasattr(batch_gen, 'process_functions')
    assert hasattr(batch_gen, 'process_classes')
    assert hasattr(batch_gen, 'process_batch')
    assert hasattr(batch_gen, '_load_cache')
    assert hasattr(batch_gen, '_save_cache')
    assert hasattr(batch_gen, '_get_cache_key')
