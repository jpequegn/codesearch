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


def test_process_single_function():
    """Test processing a single function."""
    generator = EmbeddingGenerator()
    preparator = TextPreparator(generator.tokenizer, max_tokens=512)
    batch_gen = BatchEmbeddingGenerator(generator, preparator)

    func = Function(
        name="add",
        file_path="/test.py",
        language="python",
        source_code="def add(a, b):\n    return a + b",
        docstring="Add two numbers.",
        line_number=1,
    )

    result = batch_gen.process_functions([func])

    # Should return dict with summary and embeddings
    assert "summary" in result
    assert "embeddings" in result
    assert result["summary"]["total"] == 1
    assert result["summary"]["success"] == 1
    assert "/test.py:1" in result["embeddings"]


def test_process_multiple_functions():
    """Test processing multiple functions."""
    generator = EmbeddingGenerator()
    preparator = TextPreparator(generator.tokenizer, max_tokens=512)
    batch_gen = BatchEmbeddingGenerator(generator, preparator)

    functions = [
        Function(
            name="add", file_path="/test.py", language="python",
            source_code="def add(a, b):\n    return a + b",
            docstring="Add numbers.", line_number=1,
        ),
        Function(
            name="sub", file_path="/test.py", language="python",
            source_code="def sub(a, b):\n    return a - b",
            docstring="Subtract numbers.", line_number=5,
        ),
    ]

    result = batch_gen.process_functions(functions)

    assert result["summary"]["total"] == 2
    assert result["summary"]["success"] == 2
    assert len(result["embeddings"]) == 2


def test_process_classes():
    """Test processing classes."""
    generator = EmbeddingGenerator()
    preparator = TextPreparator(generator.tokenizer, max_tokens=512)
    batch_gen = BatchEmbeddingGenerator(generator, preparator)

    cls = Class(
        name="Calculator",
        file_path="/test.py",
        language="python",
        source_code="class Calculator:\n    pass",
        docstring="A calculator class.",
        line_number=10,
    )

    result = batch_gen.process_classes([cls])

    assert result["summary"]["total"] == 1
    assert result["summary"]["success"] == 1
    assert "/test.py:10" in result["embeddings"]


def test_cache_hit():
    """Test that cached embeddings are reused."""
    import tempfile

    generator = EmbeddingGenerator()
    preparator = TextPreparator(generator.tokenizer, max_tokens=512)

    with tempfile.TemporaryDirectory() as tmpdir:
        batch_gen = BatchEmbeddingGenerator(generator, preparator, cache_dir=tmpdir)

        func = Function(
            name="test", file_path="/test.py", language="python",
            source_code="def test(): pass",
            docstring="Test function.", line_number=1,
        )

        # First run - should compute
        result1 = batch_gen.process_functions([func])
        assert result1["summary"]["newly_embedded"] == 1
        assert result1["summary"]["cached"] == 0

        # Second run - should use cache
        batch_gen2 = BatchEmbeddingGenerator(generator, preparator, cache_dir=tmpdir)
        result2 = batch_gen2.process_functions([func])
        assert result2["summary"]["cached"] == 1
        assert result2["summary"]["newly_embedded"] == 0

        # Embeddings should match
        assert result1["embeddings"]["/test.py:1"] == result2["embeddings"]["/test.py:1"]


def test_mixed_cache_hit_miss():
    """Test batch with both cached and new embeddings."""
    import tempfile

    generator = EmbeddingGenerator()
    preparator = TextPreparator(generator.tokenizer, max_tokens=512)

    with tempfile.TemporaryDirectory() as tmpdir:
        batch_gen = BatchEmbeddingGenerator(generator, preparator, cache_dir=tmpdir)

        func1 = Function(
            name="cached", file_path="/test.py", language="python",
            source_code="def cached(): pass",
            docstring="Cached.", line_number=1,
        )

        # First run
        batch_gen.process_functions([func1])

        # Second run with new function
        func2 = Function(
            name="new", file_path="/test.py", language="python",
            source_code="def new(): pass",
            docstring="New.", line_number=10,
        )

        batch_gen2 = BatchEmbeddingGenerator(generator, preparator, cache_dir=tmpdir)
        result = batch_gen2.process_functions([func1, func2])

        assert result["summary"]["total"] == 2
        assert result["summary"]["cached"] == 1
        assert result["summary"]["newly_embedded"] == 1
        assert result["summary"]["success"] == 2
