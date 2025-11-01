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


def test_load_cache_creates_metadata():
    """Test that _load_cache initializes metadata."""
    generator = EmbeddingGenerator()
    preparator = TextPreparator(generator.tokenizer, max_tokens=512)
    batch_gen = BatchEmbeddingGenerator(generator, preparator)

    batch_gen._load_cache()

    # Should have metadata even if cache file doesn't exist
    assert batch_gen.metadata is not None
    assert isinstance(batch_gen.metadata, dict)


def test_save_cache_creates_file():
    """Test that _save_cache persists cache to disk."""
    import tempfile
    import json
    import os

    generator = EmbeddingGenerator()
    preparator = TextPreparator(generator.tokenizer, max_tokens=512)

    with tempfile.TemporaryDirectory() as tmpdir:
        batch_gen = BatchEmbeddingGenerator(
            generator, preparator,
            cache_dir=tmpdir
        )

        # Add something to cache
        batch_gen.cache["/test.py:1"] = {
            "name": "test_func",
            "embedding": [0.1, 0.2, 0.3],
            "timestamp": "2025-11-01T00:00:00Z",
            "model_version": "1.0"
        }
        batch_gen.metadata = {
            "model_name": "codebert-base",
            "model_version": "1.0"
        }

        batch_gen._save_cache()

        # Verify file exists
        assert os.path.exists(batch_gen.cache_path)

        # Verify file contains expected data
        with open(batch_gen.cache_path, 'r') as f:
            data = json.load(f)

        assert "embeddings" in data
        assert "/test.py:1" in data["embeddings"]


def test_load_existing_cache():
    """Test that _load_cache reads existing cache file."""
    import tempfile
    import json
    import os

    generator = EmbeddingGenerator()
    preparator = TextPreparator(generator.tokenizer, max_tokens=512)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create cache file
        cache_path = os.path.join(tmpdir, "cache.json")
        cache_data = {
            "metadata": {
                "model_name": "codebert-base",
                "model_version": "1.0"
            },
            "embeddings": {
                "/existing.py:10": {
                    "name": "existing_func",
                    "embedding": [0.5, 0.6, 0.7],
                    "timestamp": "2025-11-01T00:00:00Z",
                    "model_version": "1.0"
                }
            }
        }

        with open(cache_path, 'w') as f:
            json.dump(cache_data, f)

        batch_gen = BatchEmbeddingGenerator(
            generator, preparator,
            cache_dir=tmpdir
        )

        batch_gen._load_cache()

        # Verify cache loaded
        assert "/existing.py:10" in batch_gen.cache
        assert batch_gen.cache["/existing.py:10"]["name"] == "existing_func"
