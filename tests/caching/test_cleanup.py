"""Tests for cache cleanup and maintenance."""

import tempfile
from pathlib import Path
import pickle
import time
from datetime import datetime, timedelta
import pytest

from codesearch.caching.cleanup import CacheCleanup
from codesearch.caching.cache import CacheEntry


class TestCacheCleanup:
    """Tests for cache cleanup and maintenance."""

    def test_cleanup_initialization(self):
        """Test cleanup manager initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cleanup = CacheCleanup(cache_dir=Path(tmpdir))

            assert cleanup.cache_dir == Path(tmpdir)

    def test_cleanup_by_age(self):
        """Test removing cache entries older than specified days."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            cache_dir.mkdir(exist_ok=True)

            # Create old and new cache files
            old_file = cache_dir / "old.cache"
            new_file = cache_dir / "new.cache"

            entry_old = CacheEntry("key1", "value1")
            entry_new = CacheEntry("key2", "value2")

            with open(old_file, "wb") as f:
                pickle.dump(entry_old, f)
            with open(new_file, "wb") as f:
                pickle.dump(entry_new, f)

            # Make old file appear old
            old_time = (datetime.now() - timedelta(days=35)).timestamp()
            old_file.touch()
            import os

            os.utime(str(old_file), (old_time, old_time))

            cleanup = CacheCleanup(cache_dir=cache_dir)
            removed = cleanup.cleanup_by_age(days=30)

            assert removed == 1
            assert not old_file.exists()
            assert new_file.exists()

    def test_cleanup_expired(self):
        """Test removing expired cache entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            cache_dir.mkdir(exist_ok=True)

            # Create expired and valid cache files
            expired_entry = CacheEntry(
                "key1",
                "value1",
                expires_at=datetime.now() - timedelta(seconds=1),
            )
            valid_entry = CacheEntry(
                "key2",
                "value2",
                expires_at=datetime.now() + timedelta(seconds=60),
            )

            expired_file = cache_dir / "expired.cache"
            valid_file = cache_dir / "valid.cache"

            with open(expired_file, "wb") as f:
                pickle.dump(expired_entry, f)
            with open(valid_file, "wb") as f:
                pickle.dump(valid_entry, f)

            cleanup = CacheCleanup(cache_dir=cache_dir)
            removed = cleanup.cleanup_expired()

            assert removed == 1
            assert not expired_file.exists()
            assert valid_file.exists()

    def test_cleanup_by_size(self):
        """Test removing cache entries until under size limit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            cache_dir.mkdir(exist_ok=True)

            # Create multiple cache files
            for i in range(5):
                cache_file = cache_dir / f"cache_{i}.cache"
                entry = CacheEntry(f"key{i}", "x" * 1000)  # ~1KB each
                with open(cache_file, "wb") as f:
                    pickle.dump(entry, f)

            cleanup = CacheCleanup(cache_dir=cache_dir)
            # Max size 3KB should remove oldest files
            removed = cleanup.cleanup_by_size(max_size_mb=0.000003)  # ~3KB

            # Should have removed at least 2 files to get under limit
            assert removed >= 2

    def test_clear_all(self):
        """Test clearing entire cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            cache_dir.mkdir(exist_ok=True)

            # Create some cache files
            for i in range(3):
                cache_file = cache_dir / f"cache_{i}.cache"
                entry = CacheEntry(f"key{i}", f"value{i}")
                with open(cache_file, "wb") as f:
                    pickle.dump(entry, f)

            cleanup = CacheCleanup(cache_dir=cache_dir)
            cleanup.clear_all()

            # Directory should be empty
            assert len(list(cache_dir.iterdir())) == 0

    def test_clear_category(self):
        """Test clearing cache for specific category."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)

            # Create category directories
            ast_dir = cache_dir / "ast"
            embeddings_dir = cache_dir / "embeddings"
            ast_dir.mkdir(parents=True, exist_ok=True)
            embeddings_dir.mkdir(parents=True, exist_ok=True)

            # Create cache files in each category
            for i in range(2):
                ast_file = ast_dir / f"ast_{i}.cache"
                entry = CacheEntry(f"ast_key{i}", f"ast_value{i}")
                with open(ast_file, "wb") as f:
                    pickle.dump(entry, f)

                emb_file = embeddings_dir / f"emb_{i}.cache"
                entry = CacheEntry(f"emb_key{i}", f"emb_value{i}")
                with open(emb_file, "wb") as f:
                    pickle.dump(entry, f)

            cleanup = CacheCleanup(cache_dir=cache_dir)
            removed = cleanup.clear_category("ast")

            assert removed == 2
            assert not ast_dir.exists()
            assert embeddings_dir.exists()
            assert len(list(embeddings_dir.iterdir())) == 2

    def test_get_stats_empty_cache(self):
        """Test getting stats for empty cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cleanup = CacheCleanup(cache_dir=Path(tmpdir))
            stats = cleanup.get_stats()

            assert stats["total_size_mb"] == 0
            assert stats["total_files"] == 0

    def test_get_stats_with_files(self):
        """Test getting stats with cache files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)

            # Create category directories with files
            ast_dir = cache_dir / "ast"
            ast_dir.mkdir(parents=True, exist_ok=True)

            for i in range(3):
                cache_file = ast_dir / f"ast_{i}.cache"
                entry = CacheEntry(f"key{i}", "x" * 100)
                with open(cache_file, "wb") as f:
                    pickle.dump(entry, f)

            cleanup = CacheCleanup(cache_dir=cache_dir)
            stats = cleanup.get_stats()

            assert stats["total_files"] == 3
            assert stats["total_size_mb"] > 0
            assert "ast" in stats["categories"]
            assert stats["categories"]["ast"]["files"] == 3

    def test_cleanup_nonexistent_directory(self):
        """Test cleanup on nonexistent cache directory."""
        cleanup = CacheCleanup(cache_dir=Path("/nonexistent/cache"))

        assert cleanup.cleanup_expired() == 0
        assert cleanup.cleanup_by_age(days=30) == 0
        assert cleanup.cleanup_by_size(max_size_mb=100) == 0

    def test_get_stats_nonexistent_directory(self):
        """Test getting stats for nonexistent directory."""
        cleanup = CacheCleanup(cache_dir=Path("/nonexistent/cache"))
        stats = cleanup.get_stats()

        assert stats["total_size_mb"] == 0
        assert stats["total_files"] == 0

    def test_cleanup_handles_corrupted_files(self):
        """Test that cleanup handles corrupted cache files gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            cache_dir.mkdir(exist_ok=True)

            # Create a corrupted cache file
            corrupted_file = cache_dir / "corrupted.cache"
            corrupted_file.write_bytes(b"corrupted data")

            # Create a valid cache file
            valid_file = cache_dir / "valid.cache"
            entry = CacheEntry("key1", "value1")
            with open(valid_file, "wb") as f:
                pickle.dump(entry, f)

            cleanup = CacheCleanup(cache_dir=cache_dir)
            # Should handle corrupted file gracefully
            removed = cleanup.cleanup_expired()

            # Valid file should still exist
            assert valid_file.exists()

    def test_cleanup_multiple_categories(self):
        """Test cleanup with multiple cache categories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)

            # Create multiple category directories
            for category in ["ast", "embeddings", "misc"]:
                cat_dir = cache_dir / category
                cat_dir.mkdir(parents=True, exist_ok=True)

                for i in range(2):
                    cache_file = cat_dir / f"{category}_{i}.cache"
                    entry = CacheEntry(f"{category}_key{i}", f"{category}_value{i}")
                    with open(cache_file, "wb") as f:
                        pickle.dump(entry, f)

            cleanup = CacheCleanup(cache_dir=cache_dir)
            stats = cleanup.get_stats()

            assert stats["total_files"] == 6
            assert len(stats["categories"]) == 3
            for category in ["ast", "embeddings", "misc"]:
                assert category in stats["categories"]
                assert stats["categories"][category]["files"] == 2
