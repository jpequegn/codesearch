"""AST (Abstract Syntax Tree) caching for parsed code."""

import hashlib
from pathlib import Path
from typing import List, Optional
import logging

from codesearch.caching.cache import Cache
from codesearch.models import Function

logger = logging.getLogger(__name__)


class ASTCache(Cache[List[Function]]):
    """Cache for parsed AST structures."""

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize AST cache.

        Args:
            cache_dir: Directory for cache storage
        """
        super().__init__(
            cache_dir=cache_dir or (
                Path.home() / ".codesearch" / "cache" / "ast"
            ),
            max_size=500,
            ttl_minutes=None,  # AST cache doesn't expire
        )
        logger.debug("Initialized AST cache")

    def get_for_file(self, file_path: str, file_hash: str) -> Optional[List[Function]]:
        """Get cached AST for a file.

        Args:
            file_path: Path to source file
            file_hash: SHA256 hash of file contents (for validation)

        Returns:
            List of Function objects or None if not cached/invalid
        """
        key = self._make_key(file_path, file_hash)
        return self.get(key)

    def set_for_file(
        self,
        file_path: str,
        file_hash: str,
        functions: List[Function],
    ) -> None:
        """Cache AST for a file.

        Args:
            file_path: Path to source file
            file_hash: SHA256 hash of file contents
            functions: Parsed functions from file
        """
        key = self._make_key(file_path, file_hash)
        self.set(key, functions, version=self._get_version())

        logger.debug(f"Cached AST for {file_path}: {len(functions)} functions")

    def invalidate_for_file(self, file_path: str) -> bool:
        """Invalidate all cached ASTs for a file (all versions).

        Args:
            file_path: Path to source file

        Returns:
            True if any entries were invalidated
        """
        invalidated = False

        # Try common hash values (this is conservative but safe)
        for i in range(5):
            key = f"ast:{file_path}:*"  # Pattern match would be ideal
            if key in self._memory_cache:
                del self._memory_cache[key]
                invalidated = True

        # Clear disk cache for this file
        if self.cache_dir:
            for cache_file in self.cache_dir.glob("*.cache"):
                try:
                    entry = self._load_from_file(cache_file)
                    if entry.key.startswith(f"ast:{file_path}:"):
                        cache_file.unlink()
                        invalidated = True
                except Exception as e:
                    logger.debug(f"Error checking cache file: {e}")

        if invalidated:
            logger.debug(f"Invalidated cached AST for {file_path}")

        return invalidated

    def _make_key(self, file_path: str, file_hash: str) -> str:
        """Create cache key from file path and hash.

        Args:
            file_path: Path to source file
            file_hash: SHA256 hash of contents

        Returns:
            Cache key
        """
        return f"ast:{file_path}:{file_hash}"

    def _get_version(self) -> str:
        """Get current cache version.

        Returns:
            Version string
        """
        # Version based on parser version - increment if parser changes
        return "1.0"
