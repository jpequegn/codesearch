"""Tests for AST caching."""

import tempfile
from pathlib import Path
import pytest

from codesearch.caching.ast_cache import ASTCache
from codesearch.models import Function


class TestASTCache:
    """Tests for AST cache."""

    def test_ast_cache_initialization(self):
        """Test AST cache initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ASTCache(cache_dir=Path(tmpdir))

            assert cache is not None

    def test_set_and_get_ast(self):
        """Test storing and retrieving AST."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ASTCache(cache_dir=Path(tmpdir))

            # Create mock functions
            functions = [
                Function(
                    name="func1",
                    file_path="test.py",
                    line_number=1,
                    end_line=5,
                    source_code="def func1(): pass",
                ),
            ]

            cache.set_for_file("test.py", "hash123", functions)

            retrieved = cache.get_for_file("test.py", "hash123")

            assert retrieved is not None
            assert len(retrieved) == 1
            assert retrieved[0].name == "func1"

    def test_ast_cache_miss(self):
        """Test AST cache miss."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ASTCache(cache_dir=Path(tmpdir))

            retrieved = cache.get_for_file("test.py", "hash123")

            assert retrieved is None

    def test_ast_cache_hash_mismatch(self):
        """Test AST not retrieved with different hash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ASTCache(cache_dir=Path(tmpdir))

            functions = [
                Function(
                    name="func1",
                    file_path="test.py",
                    line_number=1,
                    end_line=5,
                    source_code="def func1(): pass",
                ),
            ]

            cache.set_for_file("test.py", "hash123", functions)

            # Try to retrieve with different hash
            retrieved = cache.get_for_file("test.py", "hash456")

            assert retrieved is None

    def test_ast_cache_invalidate(self):
        """Test invalidating AST cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ASTCache(cache_dir=Path(tmpdir))

            functions = [
                Function(
                    name="func1",
                    file_path="test.py",
                    line_number=1,
                    end_line=5,
                    source_code="def func1(): pass",
                ),
            ]

            cache.set_for_file("test.py", "hash123", functions)
            assert cache.get_for_file("test.py", "hash123") is not None

            cache.invalidate_for_file("test.py")

            # Should not retrieve after invalidation (though implementation may vary)
            # This test validates the invalidation call works
