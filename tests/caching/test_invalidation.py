"""Tests for cache invalidation."""

import tempfile
from pathlib import Path
import pytest

from codesearch.caching.invalidation import CacheInvalidator


class TestCacheInvalidator:
    """Tests for cache invalidation."""

    def test_get_file_hash(self):
        """Test computing file hash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.py"
            test_file.write_text("def hello(): pass")

            invalidator = CacheInvalidator()
            hash1 = invalidator.get_file_hash(str(test_file))

            assert hash1 is not None
            assert len(hash1) == 64  # SHA256 hex is 64 characters
            assert hash1.isalnum()

    def test_track_file(self):
        """Test tracking file hash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.py"
            test_file.write_text("def hello(): pass")

            invalidator = CacheInvalidator()
            hash1 = invalidator.track_file(str(test_file))

            assert hash1 is not None
            assert len(hash1) == 64

    def test_has_changed_untracked(self):
        """Test file is considered changed if never tracked."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.py"
            test_file.write_text("def hello(): pass")

            invalidator = CacheInvalidator()
            # Never tracked before
            assert invalidator.has_changed(str(test_file))

    def test_has_changed_no_modification(self):
        """Test file not changed when contents are same."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.py"
            test_file.write_text("def hello(): pass")

            invalidator = CacheInvalidator()
            invalidator.track_file(str(test_file))

            # File hasn't changed
            assert not invalidator.has_changed(str(test_file))

    def test_has_changed_after_modification(self):
        """Test file change detection after modification."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.py"
            test_file.write_text("def hello(): pass")

            invalidator = CacheInvalidator()
            invalidator.track_file(str(test_file))

            # Modify file
            test_file.write_text("def hello(): return 42")

            assert invalidator.has_changed(str(test_file))

    def test_get_changed_files(self):
        """Test getting list of changed files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = Path(tmpdir) / "file1.py"
            file2 = Path(tmpdir) / "file2.py"
            file3 = Path(tmpdir) / "file3.py"

            file1.write_text("code1")
            file2.write_text("code2")
            file3.write_text("code3")

            invalidator = CacheInvalidator()
            invalidator.track_file(str(file1))
            invalidator.track_file(str(file2))
            invalidator.track_file(str(file3))

            # Modify only file2
            file2.write_text("modified")

            changed = invalidator.get_changed_files(
                [str(file1), str(file2), str(file3)]
            )

            assert len(changed) == 1
            assert str(file2) in changed

    def test_invalidate_file(self):
        """Test invalidating file tracking."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.py"
            test_file.write_text("def hello(): pass")

            invalidator = CacheInvalidator()
            invalidator.track_file(str(test_file))

            # Invalidate tracking
            assert invalidator.invalidate_file(str(test_file))

            # File is now considered changed since not tracked
            assert invalidator.has_changed(str(test_file))

    def test_invalidate_untracked_file(self):
        """Test invalidating file that was never tracked."""
        invalidator = CacheInvalidator()
        assert not invalidator.invalidate_file("/nonexistent/file.py")

    def test_clear_tracking(self):
        """Test clearing all tracked files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = Path(tmpdir) / "file1.py"
            file2 = Path(tmpdir) / "file2.py"

            file1.write_text("code1")
            file2.write_text("code2")

            invalidator = CacheInvalidator()
            invalidator.track_file(str(file1))
            invalidator.track_file(str(file2))

            invalidator.clear_tracking()

            # Files are now considered changed since tracking cleared
            assert invalidator.has_changed(str(file1))
            assert invalidator.has_changed(str(file2))

    def test_get_stats(self):
        """Test getting invalidator statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = Path(tmpdir) / "file1.py"
            file2 = Path(tmpdir) / "file2.py"

            file1.write_text("code1")
            file2.write_text("code2")

            invalidator = CacheInvalidator()
            invalidator.track_file(str(file1))
            invalidator.track_file(str(file2))

            stats = invalidator.get_stats()

            assert "tracked_files" in stats
            assert stats["tracked_files"] == 2

    def test_same_content_different_files(self):
        """Test that same content produces same hash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = Path(tmpdir) / "file1.py"
            file2 = Path(tmpdir) / "file2.py"

            content = "def hello(): pass"
            file1.write_text(content)
            file2.write_text(content)

            invalidator = CacheInvalidator()
            hash1 = invalidator.get_file_hash(str(file1))
            hash2 = invalidator.get_file_hash(str(file2))

            assert hash1 == hash2
