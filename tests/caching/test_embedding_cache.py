"""Tests for embedding caching."""

import tempfile
from pathlib import Path
import pytest

from codesearch.caching.embedding_cache import EmbeddingCache


class TestEmbeddingCache:
    """Tests for embedding cache."""

    def test_embedding_cache_initialization(self):
        """Test embedding cache initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = EmbeddingCache(cache_dir=Path(tmpdir))

            assert cache is not None
            assert cache.CURRENT_MODEL_VERSION is not None

    def test_set_and_get_embedding(self):
        """Test storing and retrieving embedding."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = EmbeddingCache(cache_dir=Path(tmpdir))

            code = "def hello(): print('world')"
            embedding = [0.1, 0.2, 0.3, 0.4, 0.5]

            cache.set_embedding(code, embedding)

            retrieved = cache.get_embedding(code)

            assert retrieved is not None
            assert retrieved == embedding

    def test_embedding_cache_miss(self):
        """Test embedding cache miss."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = EmbeddingCache(cache_dir=Path(tmpdir))

            code = "def hello(): print('world')"
            retrieved = cache.get_embedding(code)

            assert retrieved is None

    def test_embedding_batch_operations(self):
        """Test batch embedding operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = EmbeddingCache(cache_dir=Path(tmpdir))

            codes = ["code1", "code2", "code3"]
            embeddings = [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
            ]

            cache.set_batch_embeddings(codes, embeddings)

            uncached, retrieved = cache.get_batch_embeddings(codes)

            assert len(uncached) == 0
            assert all(e is not None for e in retrieved)

    def test_embedding_batch_partial_cache(self):
        """Test batch operations with partial cache hit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = EmbeddingCache(cache_dir=Path(tmpdir))

            # Cache first two codes
            cache.set_embedding("code1", [0.1, 0.2])
            cache.set_embedding("code2", [0.3, 0.4])

            # Try to get three codes (only two cached)
            codes = ["code1", "code2", "code3"]
            uncached, retrieved = cache.get_batch_embeddings(codes)

            assert len(uncached) == 1
            assert uncached[0] == 2
            assert retrieved[0] is not None  # code1
            assert retrieved[1] is not None  # code2
            assert retrieved[2] is None  # code3

    def test_embedding_model_version_tracking(self):
        """Test embedding version tracking."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = EmbeddingCache(cache_dir=Path(tmpdir))

            code = "def hello(): pass"
            embedding = [0.1, 0.2, 0.3]

            # Store with current model version
            cache.set_embedding(code, embedding)

            # Should retrieve with same version
            retrieved = cache.get_embedding(code)
            assert retrieved is not None

            # Change model version
            old_version = cache.CURRENT_MODEL_VERSION
            cache.CURRENT_MODEL_VERSION = "new-model-v2"

            # Should not retrieve with different version
            retrieved = cache.get_embedding(code)
            assert retrieved is None

            # Restore for cleanup
            cache.CURRENT_MODEL_VERSION = old_version

    def test_embedding_invalidate_model_version(self):
        """Test invalidating embeddings by model version."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = EmbeddingCache(cache_dir=Path(tmpdir))

            cache.set_embedding("code1", [0.1, 0.2], model_version="v1")
            cache.set_embedding("code2", [0.3, 0.4], model_version="v2")

            count = cache.invalidate_model_version("v1")

            assert count >= 1  # At least one invalidated

    def test_embedding_mismatch_error(self):
        """Test error on code/embedding mismatch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = EmbeddingCache(cache_dir=Path(tmpdir))

            with pytest.raises(ValueError):
                cache.set_batch_embeddings(
                    ["code1", "code2"], [[0.1, 0.2]]  # 2 codes, 1 embedding
                )
