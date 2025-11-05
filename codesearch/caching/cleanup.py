"""Cache cleanup and maintenance utilities."""

import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class CacheCleanup:
    """Manages cache cleanup and maintenance."""

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize cleanup manager.

        Args:
            cache_dir: Root cache directory
        """
        self.cache_dir = cache_dir or (Path.home() / ".codesearch" / "cache")

    def cleanup_expired(self) -> int:
        """Remove expired cache entries.

        Returns:
            Number of entries removed
        """
        import pickle

        removed_count = 0

        if not self.cache_dir.exists():
            return 0

        # Recursively check all cache files
        for cache_file in self.cache_dir.rglob("*.cache"):
            try:
                with open(cache_file, "rb") as f:
                    entry = pickle.load(f)

                if entry.is_expired():
                    cache_file.unlink()
                    removed_count += 1
                    logger.debug(f"Removed expired cache entry: {cache_file.name}")

            except Exception as e:
                logger.warning(f"Failed to check cache file {cache_file}: {e}")

        logger.info(f"Removed {removed_count} expired cache entries")
        return removed_count

    def cleanup_by_size(self, max_size_mb: int = 100) -> int:
        """Remove oldest cache entries until size is under limit.

        Args:
            max_size_mb: Maximum cache size in MB

        Returns:
            Number of entries removed
        """
        if not self.cache_dir.exists():
            return 0

        # Calculate current size
        current_size = self._get_directory_size(self.cache_dir)
        current_size_mb = current_size / (1024 * 1024)

        if current_size_mb <= max_size_mb:
            logger.debug(f"Cache size {current_size_mb:.1f}MB is under limit")
            return 0

        # Remove oldest files until under limit
        cache_files = []
        for cache_file in self.cache_dir.rglob("*.cache"):
            stat = cache_file.stat()
            cache_files.append((cache_file, stat.st_mtime))

        # Sort by modification time (oldest first)
        cache_files.sort(key=lambda x: x[1])

        removed_count = 0
        for cache_file, _ in cache_files:
            if current_size_mb <= max_size_mb:
                break

            try:
                file_size = cache_file.stat().st_size
                cache_file.unlink()
                current_size -= file_size
                current_size_mb = current_size / (1024 * 1024)
                removed_count += 1
                logger.debug(f"Removed cache file: {cache_file.name}")

            except Exception as e:
                logger.warning(f"Failed to remove cache file {cache_file}: {e}")

        logger.info(f"Removed {removed_count} cache entries for size management")
        return removed_count

    def cleanup_by_age(self, days: int = 30) -> int:
        """Remove cache entries older than specified days.

        Args:
            days: Age threshold in days

        Returns:
            Number of entries removed
        """
        if not self.cache_dir.exists():
            return 0

        cutoff_time = datetime.now() - timedelta(days=days)
        cutoff_timestamp = cutoff_time.timestamp()

        removed_count = 0

        for cache_file in self.cache_dir.rglob("*.cache"):
            stat = cache_file.stat()
            file_mtime = datetime.fromtimestamp(stat.st_mtime)

            if file_mtime < cutoff_time:
                try:
                    cache_file.unlink()
                    removed_count += 1
                    logger.debug(f"Removed old cache file: {cache_file.name}")
                except Exception as e:
                    logger.warning(f"Failed to remove cache file {cache_file}: {e}")

        logger.info(f"Removed {removed_count} cache entries older than {days} days")
        return removed_count

    def clear_all(self) -> None:
        """Clear entire cache directory."""
        if self.cache_dir.exists():
            try:
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                logger.info("Cleared entire cache")
            except Exception as e:
                logger.error(f"Failed to clear cache: {e}")

    def clear_category(self, category: str) -> int:
        """Clear cache for a specific category (ast, embeddings, etc).

        Args:
            category: Cache category name

        Returns:
            Number of files removed
        """
        category_dir = self.cache_dir / category

        if not category_dir.exists():
            return 0

        removed_count = 0

        for cache_file in category_dir.rglob("*.cache"):
            try:
                cache_file.unlink()
                removed_count += 1
            except Exception as e:
                logger.warning(f"Failed to remove cache file: {e}")

        # Remove category directory if empty
        try:
            if not list(category_dir.iterdir()):
                category_dir.rmdir()
        except Exception as e:
            logger.debug(f"Failed to remove category directory: {e}")

        logger.info(f"Removed {removed_count} cache entries from {category}")
        return removed_count

    def get_stats(self) -> dict:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        if not self.cache_dir.exists():
            return {
                "total_size_mb": 0,
                "total_files": 0,
                "cache_dir": str(self.cache_dir),
            }

        total_size = self._get_directory_size(self.cache_dir)
        total_files = len(list(self.cache_dir.rglob("*.cache")))

        # Get per-category stats
        categories = {}
        for subdir in self.cache_dir.iterdir():
            if subdir.is_dir():
                size = self._get_directory_size(subdir)
                files = len(list(subdir.rglob("*.cache")))
                categories[subdir.name] = {
                    "size_mb": size / (1024 * 1024),
                    "files": files,
                }

        return {
            "total_size_mb": total_size / (1024 * 1024),
            "total_files": total_files,
            "cache_dir": str(self.cache_dir),
            "categories": categories,
        }

    def _get_directory_size(self, directory: Path) -> int:
        """Get total size of directory in bytes.

        Args:
            directory: Directory path

        Returns:
            Size in bytes
        """
        total_size = 0

        try:
            for file_path in directory.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        except Exception as e:
            logger.warning(f"Failed to calculate directory size: {e}")

        return total_size
