"""Core caching infrastructure."""

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Generic, TypeVar
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CacheEntry(Generic[T]):
    """Represents a cached entry with metadata."""

    def __init__(
        self,
        key: str,
        value: T,
        version: str = "1.0",
        created_at: Optional[datetime] = None,
        expires_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize cache entry.

        Args:
            key: Unique cache key
            value: Cached value
            version: Version string for invalidation
            created_at: Creation timestamp
            expires_at: Expiration timestamp
            metadata: Additional metadata
        """
        self.key = key
        self.value = value
        self.version = version
        self.created_at = created_at or datetime.now()
        self.expires_at = expires_at
        self.metadata = metadata or {}

    def is_expired(self) -> bool:
        """Check if entry has expired.

        Returns:
            True if entry has expired
        """
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation
        """
        return {
            "key": self.key,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "metadata": self.metadata,
        }


class Cache(Generic[T]):
    """Generic in-memory cache with optional disk persistence."""

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        max_size: int = 1000,
        ttl_minutes: Optional[int] = None,
    ):
        """Initialize cache.

        Args:
            cache_dir: Directory for disk cache (optional)
            max_size: Maximum number of entries
            ttl_minutes: Time-to-live in minutes (None = no expiration)
        """
        self.cache_dir = cache_dir or (
            Path.home() / ".codesearch" / "cache"
        )
        self.max_size = max_size
        self.ttl_minutes = ttl_minutes
        self._memory_cache: Dict[str, CacheEntry[T]] = {}

        # Create cache directory if needed
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Initialized cache at {self.cache_dir}")

    def get(self, key: str) -> Optional[T]:
        """Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        # Try memory cache first
        if key in self._memory_cache:
            entry = self._memory_cache[key]
            if not entry.is_expired():
                logger.debug(f"Cache hit (memory): {key}")
                return entry.value
            else:
                del self._memory_cache[key]
                logger.debug(f"Cache expired (memory): {key}")

        # Try disk cache
        if self.cache_dir:
            return self._get_from_disk(key)

        return None

    def set(self, key: str, value: T, version: str = "1.0") -> None:
        """Store value in cache.

        Args:
            key: Cache key
            value: Value to cache
            version: Version string for invalidation
        """
        # Calculate expiration
        expires_at = None
        if self.ttl_minutes:
            expires_at = datetime.now() + timedelta(minutes=self.ttl_minutes)

        entry = CacheEntry(key=key, value=value, version=version, expires_at=expires_at)

        # Store in memory cache
        if len(self._memory_cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = min(
                self._memory_cache.keys(),
                key=lambda k: self._memory_cache[k].created_at,
            )
            del self._memory_cache[oldest_key]
            logger.debug(f"Evicted cache entry: {oldest_key}")

        self._memory_cache[key] = entry

        # Store on disk if configured
        if self.cache_dir:
            self._set_on_disk(key, entry)

        logger.debug(f"Cached value: {key}")

    def delete(self, key: str) -> bool:
        """Delete value from cache.

        Args:
            key: Cache key

        Returns:
            True if entry was deleted
        """
        deleted = False

        if key in self._memory_cache:
            del self._memory_cache[key]
            deleted = True

        if self.cache_dir:
            deleted = self._delete_from_disk(key) or deleted

        if deleted:
            logger.debug(f"Deleted cache entry: {key}")

        return deleted

    def clear(self) -> None:
        """Clear all cached entries."""
        self._memory_cache.clear()

        if self.cache_dir:
            self._clear_disk()

        logger.info("Cleared all cache entries")

    def invalidate_version(self, version: str) -> int:
        """Invalidate all entries of a specific version.

        Args:
            version: Version string to invalidate

        Returns:
            Number of entries invalidated
        """
        count = 0

        # Invalidate in memory
        keys_to_delete = [
            k
            for k, v in self._memory_cache.items()
            if v.version == version
        ]
        for key in keys_to_delete:
            del self._memory_cache[key]
            count += 1

        # Invalidate on disk
        if self.cache_dir:
            for cache_file in self.cache_dir.glob("*.cache"):
                try:
                    entry = self._load_from_file(cache_file)
                    if entry.version == version:
                        cache_file.unlink()
                        count += 1
                except Exception as e:
                    logger.warning(f"Failed to check cache file {cache_file}: {e}")

        logger.info(f"Invalidated {count} cache entries for version {version}")
        return count

    def _get_cache_path(self, key: str) -> Path:
        """Get disk cache path for key.

        Args:
            key: Cache key

        Returns:
            Path to cache file
        """
        # Create safe filename from key
        safe_key = hashlib.sha256(key.encode()).hexdigest()
        return self.cache_dir / f"{safe_key}.cache"

    def _get_from_disk(self, key: str) -> Optional[T]:
        """Get value from disk cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None
        """
        try:
            cache_path = self._get_cache_path(key)
            if not cache_path.exists():
                return None

            entry = self._load_from_file(cache_path)

            if entry.is_expired():
                cache_path.unlink()
                logger.debug(f"Cache expired (disk): {key}")
                return None

            logger.debug(f"Cache hit (disk): {key}")
            return entry.value

        except Exception as e:
            logger.warning(f"Failed to load from disk cache: {e}")
            return None

    def _set_on_disk(self, key: str, entry: CacheEntry[T]) -> None:
        """Store entry on disk.

        Args:
            key: Cache key
            entry: Cache entry to store
        """
        try:
            cache_path = self._get_cache_path(key)
            with open(cache_path, "wb") as f:
                pickle.dump(entry, f)
        except Exception as e:
            logger.warning(f"Failed to write to disk cache: {e}")

    def _delete_from_disk(self, key: str) -> bool:
        """Delete entry from disk cache.

        Args:
            key: Cache key

        Returns:
            True if file was deleted
        """
        try:
            cache_path = self._get_cache_path(key)
            if cache_path.exists():
                cache_path.unlink()
                return True
        except Exception as e:
            logger.warning(f"Failed to delete disk cache: {e}")

        return False

    def _clear_disk(self) -> None:
        """Clear all disk cache files."""
        try:
            for cache_file in self.cache_dir.glob("*.cache"):
                cache_file.unlink()
        except Exception as e:
            logger.warning(f"Failed to clear disk cache: {e}")

    def _load_from_file(self, path: Path) -> CacheEntry[T]:
        """Load cache entry from file.

        Args:
            path: Path to cache file

        Returns:
            Cache entry
        """
        with open(path, "rb") as f:
            return pickle.load(f)

    def get_stats(self) -> dict:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        disk_files = 0
        disk_size = 0

        if self.cache_dir and self.cache_dir.exists():
            for cache_file in self.cache_dir.glob("*.cache"):
                disk_files += 1
                disk_size += cache_file.stat().st_size

        return {
            "memory_entries": len(self._memory_cache),
            "disk_entries": disk_files,
            "disk_size_bytes": disk_size,
            "max_size": self.max_size,
            "ttl_minutes": self.ttl_minutes,
        }
