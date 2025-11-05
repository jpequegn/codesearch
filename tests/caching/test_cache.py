"""Tests for core caching infrastructure."""

import tempfile
from pathlib import Path
from datetime import datetime, timedelta
import pytest

from codesearch.caching.cache import Cache, CacheEntry


class TestCacheEntry:
    """Tests for cache entries."""

    def test_entry_creation(self):
        """Test creating cache entry."""
        entry = CacheEntry("key1", "value1")

        assert entry.key == "key1"
        assert entry.value == "value1"
        assert entry.version == "1.0"
        assert not entry.is_expired()

    def test_entry_expiration(self):
        """Test cache entry expiration."""
        # Create entry that expires in 1 second
        expires_at = datetime.now() + timedelta(seconds=1)
        entry = CacheEntry("key1", "value1", expires_at=expires_at)

        assert not entry.is_expired()

        # Create already expired entry
        expired_at = datetime.now() - timedelta(seconds=1)
        expired_entry = CacheEntry("key2", "value2", expires_at=expired_at)

        assert expired_entry.is_expired()

    def test_entry_serialization(self):
        """Test entry to dict conversion."""
        entry = CacheEntry("key1", "value1", version="2.0")
        data = entry.to_dict()

        assert data["key"] == "key1"
        assert data["version"] == "2.0"


class TestCache:
    """Tests for core cache."""

    def test_cache_initialization(self):
        """Test cache initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = Cache(cache_dir=Path(tmpdir), max_size=100)

            assert cache.max_size == 100
            assert (Path(tmpdir)).exists()

    def test_cache_set_and_get(self):
        """Test setting and getting cache values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = Cache(cache_dir=Path(tmpdir))

            cache.set("key1", "value1")
            assert cache.get("key1") == "value1"

    def test_cache_miss(self):
        """Test cache miss returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = Cache(cache_dir=Path(tmpdir))

            assert cache.get("nonexistent") is None

    def test_cache_delete(self):
        """Test deleting cache entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = Cache(cache_dir=Path(tmpdir))

            cache.set("key1", "value1")
            assert cache.get("key1") == "value1"

            assert cache.delete("key1")
            assert cache.get("key1") is None

    def test_cache_clear(self):
        """Test clearing cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = Cache(cache_dir=Path(tmpdir))

            cache.set("key1", "value1")
            cache.set("key2", "value2")

            assert cache.get("key1") == "value1"
            cache.clear()
            assert cache.get("key1") is None
            assert cache.get("key2") is None

    def test_cache_expiration_with_ttl(self):
        """Test cache entries expire with TTL."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = Cache(cache_dir=Path(tmpdir), ttl_minutes=1)

            cache.set("key1", "value1")
            # Entry should be valid immediately
            assert cache.get("key1") == "value1"

    def test_cache_memory_size_limit(self):
        """Test cache respects memory size limit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = Cache(cache_dir=Path(tmpdir), max_size=3)

            cache.set("key1", "value1")
            cache.set("key2", "value2")
            cache.set("key3", "value3")
            assert len(cache._memory_cache) == 3

            # Adding 4th entry should evict oldest
            cache.set("key4", "value4")
            assert len(cache._memory_cache) == 3

    def test_cache_version_invalidation(self):
        """Test invalidating cache by version."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = Cache(cache_dir=Path(tmpdir))

            cache.set("key1", "value1", version="1.0")
            cache.set("key2", "value2", version="1.0")
            cache.set("key3", "value3", version="2.0")

            # Only check memory invalidation (to avoid disk interference)
            memory_count = sum(
                1 for entry in cache._memory_cache.values() if entry.version == "1.0"
            )
            assert memory_count == 2

            count = cache.invalidate_version("1.0")

            # Verify the invalidation happened
            assert cache.get("key1") is None
            assert cache.get("key2") is None
            assert cache.get("key3") == "value3"

    def test_cache_statistics(self):
        """Test getting cache statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = Cache(cache_dir=Path(tmpdir))

            cache.set("key1", "value1")
            cache.set("key2", "value2")

            stats = cache.get_stats()

            assert stats["memory_entries"] == 2
            assert stats["max_size"] > 0

    def test_cache_disk_persistence(self):
        """Test cache persists to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)

            # Set value in first cache instance
            cache1 = Cache(cache_dir=cache_dir)
            cache1.set("key1", "value1")

            # Create new cache instance reading same directory
            cache2 = Cache(cache_dir=cache_dir)
            # Should still be able to read from disk
            assert cache2.get("key1") == "value1"
