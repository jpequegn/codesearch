"""Embedding caching with versioning support."""

from pathlib import Path
from typing import List, Optional, Tuple
import logging

from codesearch.caching.cache import Cache

logger = logging.getLogger(__name__)


class EmbeddingCache(Cache[List[float]]):
    """Cache for code embeddings with version tracking."""

    # Current embedding model version - increment when model changes
    CURRENT_MODEL_VERSION = "codebert-base-v1"

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize embedding cache.

        Args:
            cache_dir: Directory for cache storage
        """
        super().__init__(
            cache_dir=cache_dir or (
                Path.home() / ".codesearch" / "cache" / "embeddings"
            ),
            max_size=10000,
            ttl_minutes=None,  # Embeddings don't expire
        )
        logger.debug("Initialized embedding cache")

    def get_embedding(
        self, code_text: str, model_version: Optional[str] = None
    ) -> Optional[List[float]]:
        """Get cached embedding for code.

        Args:
            code_text: Code snippet to get embedding for
            model_version: Model version (uses current if not specified)

        Returns:
            Embedding vector or None if not cached/invalid
        """
        model_version = model_version or self.CURRENT_MODEL_VERSION
        key = self._make_key(code_text, model_version)
        cached = self.get(key)

        if cached and self._get_cached_version(key) != model_version:
            logger.debug(f"Embedding cache stale for model {model_version}")
            return None

        return cached

    def set_embedding(
        self,
        code_text: str,
        embedding: List[float],
        model_version: Optional[str] = None,
    ) -> None:
        """Cache embedding for code.

        Args:
            code_text: Code snippet
            embedding: Embedding vector
            model_version: Model version (uses current if not specified)
        """
        model_version = model_version or self.CURRENT_MODEL_VERSION
        key = self._make_key(code_text, model_version)
        self.set(key, embedding, version=model_version)

        logger.debug(f"Cached embedding for {len(code_text)} char code")

    def get_batch_embeddings(
        self,
        code_texts: List[str],
        model_version: Optional[str] = None,
    ) -> Tuple[List[int], List[List[float]]]:
        """Get cached embeddings for multiple code snippets.

        Args:
            code_texts: List of code snippets
            model_version: Model version

        Returns:
            Tuple of (uncached_indices, cached_embeddings)
            - uncached_indices: List of indices that need embedding
            - cached_embeddings: List where entry is embedding or None
        """
        model_version = model_version or self.CURRENT_MODEL_VERSION
        uncached_indices = []
        cached_embeddings = []

        for i, code_text in enumerate(code_texts):
            embedding = self.get_embedding(code_text, model_version)
            if embedding is None:
                uncached_indices.append(i)
                cached_embeddings.append(None)
            else:
                cached_embeddings.append(embedding)

        return uncached_indices, cached_embeddings

    def set_batch_embeddings(
        self,
        code_texts: List[str],
        embeddings: List[List[float]],
        model_version: Optional[str] = None,
    ) -> None:
        """Cache embeddings for multiple code snippets.

        Args:
            code_texts: List of code snippets
            embeddings: List of embedding vectors (same length as code_texts)
            model_version: Model version
        """
        if len(code_texts) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(code_texts)} texts vs {len(embeddings)} embeddings"
            )

        model_version = model_version or self.CURRENT_MODEL_VERSION

        for code_text, embedding in zip(code_texts, embeddings):
            self.set_embedding(code_text, embedding, model_version)

        logger.debug(f"Cached {len(code_texts)} embeddings")

    def invalidate_model_version(self, model_version: str) -> int:
        """Invalidate all embeddings for a specific model version.

        Args:
            model_version: Model version to invalidate

        Returns:
            Number of embeddings invalidated
        """
        count = self.invalidate_version(model_version)
        logger.info(f"Invalidated {count} embeddings for model {model_version}")
        return count

    def invalidate_current_model(self) -> int:
        """Invalidate all embeddings for the current model version.

        Returns:
            Number of embeddings invalidated
        """
        return self.invalidate_model_version(self.CURRENT_MODEL_VERSION)

    def _make_key(self, code_text: str, model_version: str) -> str:
        """Create cache key from code and model version.

        Args:
            code_text: Code snippet
            model_version: Model version

        Returns:
            Cache key
        """
        import hashlib

        code_hash = hashlib.sha256(code_text.encode()).hexdigest()
        return f"embedding:{code_hash}:{model_version}"

    def _get_cached_version(self, key: str) -> Optional[str]:
        """Get the model version for a cached embedding.

        Args:
            key: Cache key

        Returns:
            Model version or None
        """
        if key in self._memory_cache:
            return self._memory_cache[key].version
        return None
