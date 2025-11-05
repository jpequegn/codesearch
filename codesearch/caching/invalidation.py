"""Cache invalidation based on file changes."""

import hashlib
from pathlib import Path
from typing import Optional, Set
import logging

logger = logging.getLogger(__name__)


class CacheInvalidator:
    """Manages cache invalidation on file changes."""

    def __init__(self):
        """Initialize cache invalidator."""
        self._file_hashes: dict[str, str] = {}

    def get_file_hash(self, file_path: str) -> str:
        """Compute SHA256 hash of file contents.

        Args:
            file_path: Path to file

        Returns:
            SHA256 hash of file contents
        """
        try:
            with open(file_path, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception as e:
            logger.warning(f"Failed to hash file {file_path}: {e}")
            return ""

    def track_file(self, file_path: str) -> str:
        """Track file and store its hash.

        Args:
            file_path: Path to file

        Returns:
            File hash
        """
        file_hash = self.get_file_hash(file_path)
        self._file_hashes[str(file_path)] = file_hash
        return file_hash

    def has_changed(self, file_path: str) -> bool:
        """Check if file has changed since last tracking.

        Args:
            file_path: Path to file

        Returns:
            True if file has changed
        """
        file_path_str = str(file_path)

        if file_path_str not in self._file_hashes:
            # Never tracked before - consider it changed
            return True

        current_hash = self.get_file_hash(file_path_str)
        previous_hash = self._file_hashes[file_path_str]

        has_changed = current_hash != previous_hash

        if has_changed:
            logger.debug(f"File changed: {file_path}")
            self._file_hashes[file_path_str] = current_hash

        return has_changed

    def get_changed_files(self, file_paths: list[str]) -> Set[str]:
        """Get list of files that have changed.

        Args:
            file_paths: List of file paths to check

        Returns:
            Set of paths that have changed
        """
        changed = set()

        for file_path in file_paths:
            if self.has_changed(file_path):
                changed.add(file_path)

        return changed

    def invalidate_file(self, file_path: str) -> bool:
        """Invalidate cache for a file.

        Args:
            file_path: Path to file

        Returns:
            True if file was being tracked
        """
        file_path_str = str(file_path)

        if file_path_str in self._file_hashes:
            del self._file_hashes[file_path_str]
            logger.debug(f"Invalidated tracking for {file_path}")
            return True

        return False

    def clear_tracking(self) -> None:
        """Clear all tracked files."""
        self._file_hashes.clear()
        logger.debug("Cleared file tracking")

    def get_stats(self) -> dict:
        """Get invalidator statistics.

        Returns:
            Dictionary with stats
        """
        return {
            "tracked_files": len(self._file_hashes),
        }
