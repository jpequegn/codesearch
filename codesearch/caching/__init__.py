"""Caching system for ASTs, embeddings, and metadata."""

from codesearch.caching.cache import Cache, CacheEntry
from codesearch.caching.ast_cache import ASTCache
from codesearch.caching.embedding_cache import EmbeddingCache
from codesearch.caching.invalidation import CacheInvalidator
from codesearch.caching.cleanup import CacheCleanup

__all__ = [
    "Cache",
    "CacheEntry",
    "ASTCache",
    "EmbeddingCache",
    "CacheInvalidator",
    "CacheCleanup",
]
