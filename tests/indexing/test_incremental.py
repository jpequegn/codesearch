"""Tests for incremental indexing module."""

import json
import tempfile
from pathlib import Path
import pytest

from codesearch.indexing.incremental import (
    FileManifest,
    ChangeDetector,
    IncrementalIndexer,
)


class TestFileManifest:
    """Tests for file manifest tracking."""

    def test_manifest_initialization(self):
        """Test manifest initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"
            manifest = FileManifest(manifest_path)

            assert manifest.files == {}
            assert manifest.manifest_path == manifest_path

    def test_add_file(self):
        """Test adding file to manifest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"
            manifest = FileManifest(manifest_path)

            manifest.add_file("test.py", "abc123", ["entity1", "entity2"])

            assert "test.py" in manifest.files
            assert manifest.files["test.py"]["hash"] == "abc123"
            assert manifest.files["test.py"]["entities"] == ["entity1", "entity2"]

    def test_save_and_load(self):
        """Test saving and loading manifest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"

            # Create and save manifest
            manifest1 = FileManifest(manifest_path)
            manifest1.add_file("file1.py", "hash1", ["entity1"])
            manifest1.add_file("file2.py", "hash2", ["entity2", "entity3"])
            manifest1.save()

            assert manifest_path.exists()

            # Load manifest
            manifest2 = FileManifest(manifest_path)

            assert len(manifest2.files) == 2
            assert manifest2.get_file_hash("file1.py") == "hash1"
            assert manifest2.get_file_hash("file2.py") == "hash2"
            assert manifest2.get_file_entities("file1.py") == ["entity1"]

    def test_remove_file(self):
        """Test removing file from manifest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"
            manifest = FileManifest(manifest_path)

            manifest.add_file("test.py", "hash123", ["entity1", "entity2"])
            entities = manifest.remove_file("test.py")

            assert entities == ["entity1", "entity2"]
            assert not manifest.has_file("test.py")

    def test_has_file(self):
        """Test checking if file is in manifest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"
            manifest = FileManifest(manifest_path)

            manifest.add_file("tracked.py", "hash1")

            assert manifest.has_file("tracked.py")
            assert not manifest.has_file("untracked.py")

    def test_manifest_persistence(self):
        """Test manifest persists across instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"

            # Create manifest and add files
            m1 = FileManifest(manifest_path)
            m1.add_file("file1.py", "hash1", ["e1"])
            m1.add_file("file2.py", "hash2", ["e2", "e3"])
            m1.save()

            # Load in new instance
            m2 = FileManifest(manifest_path)

            assert len(m2.files) == 2
            assert m2.get_file_hash("file1.py") == "hash1"


class TestChangeDetector:
    """Tests for change detection."""

    def test_compute_file_hash(self):
        """Test computing file hash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.txt"
            file_path.write_text("test content")

            hash1 = ChangeDetector.compute_file_hash(str(file_path))
            hash2 = ChangeDetector.compute_file_hash(str(file_path))

            # Same content should produce same hash
            assert hash1 == hash2
            assert len(hash1) == 64  # SHA256 hex is 64 chars

    def test_compute_file_hash_different_content(self):
        """Test that different content produces different hashes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = Path(tmpdir) / "test1.txt"
            file2 = Path(tmpdir) / "test2.txt"

            file1.write_text("content1")
            file2.write_text("content2")

            hash1 = ChangeDetector.compute_file_hash(str(file1))
            hash2 = ChangeDetector.compute_file_hash(str(file2))

            assert hash1 != hash2

    def test_detect_added_files(self):
        """Test detecting added files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"
            manifest = FileManifest(manifest_path)
            manifest.add_file("old.py", "hash1")

            current_files = ["old.py", "new.py"]

            added, modified, deleted = ChangeDetector.detect_changes(
                current_files, manifest
            )

            assert added == ["new.py"]
            assert modified == []
            assert deleted == []

    def test_detect_deleted_files(self):
        """Test detecting deleted files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"
            manifest = FileManifest(manifest_path)
            manifest.add_file("deleted.py", "hash1")
            manifest.add_file("kept.py", "hash2")

            current_files = ["kept.py"]

            added, modified, deleted = ChangeDetector.detect_changes(
                current_files, manifest
            )

            assert added == []
            assert modified == []
            assert deleted == ["deleted.py"]

    def test_detect_modified_files(self):
        """Test detecting modified files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.py"
            file_path.write_text("original content")

            # Initial hash
            initial_hash = ChangeDetector.compute_file_hash(str(file_path))
            manifest_path = Path(tmpdir) / "manifest.json"
            manifest = FileManifest(manifest_path)
            manifest.add_file(str(file_path), initial_hash)

            # Modify file
            file_path.write_text("modified content")

            current_files = [str(file_path)]

            added, modified, deleted = ChangeDetector.detect_changes(
                current_files, manifest
            )

            assert added == []
            assert str(file_path) in modified
            assert deleted == []

    def test_detect_all_change_types(self):
        """Test detecting all types of changes simultaneously."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Set up manifest with some files
            manifest_path = Path(tmpdir) / "manifest.json"
            manifest = FileManifest(manifest_path)
            manifest.add_file("deleted.py", "hash1")
            manifest.add_file("kept.py", "hash3")

            # Current files - simulating adds and deletions
            current_files = ["kept.py", "added.py"]

            added, modified, deleted = ChangeDetector.detect_changes(
                current_files, manifest
            )

            # Should detect added.py as added
            assert len(added) > 0
            assert "added.py" in added
            # Should detect deleted.py as deleted
            assert len(deleted) > 0
            assert "deleted.py" in deleted


class TestIncrementalIndexer:
    """Tests for incremental indexer."""

    def test_indexer_initialization(self):
        """Test indexer initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"
            manifest = FileManifest(manifest_path)
            indexer = IncrementalIndexer(manifest)

            assert indexer.manifest == manifest
            assert indexer.get_indexed_count() == 0

    def test_prepare_update(self):
        """Test preparing an incremental update."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"
            manifest = FileManifest(manifest_path)
            manifest.add_file("old.py", "hash1")

            indexer = IncrementalIndexer(manifest)
            current_files = ["old.py", "new.py"]

            changes = indexer.prepare_update(current_files)

            assert "added" in changes
            assert "modified" in changes
            assert "deleted" in changes
            assert "new.py" in changes["added"]

    def test_update_file(self):
        """Test updating file in manifest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"
            indexer = IncrementalIndexer(FileManifest(manifest_path))

            indexer.update_file("test.py", ["entity1", "entity2"])

            assert indexer.manifest.has_file("test.py")
            assert indexer.manifest.get_file_entities("test.py") == ["entity1", "entity2"]

    def test_remove_file(self):
        """Test removing file from indexer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"
            manifest = FileManifest(manifest_path)
            manifest.add_file("test.py", "hash1", ["entity1", "entity2"])

            indexer = IncrementalIndexer(manifest)
            entities = indexer.remove_file("test.py")

            assert entities == ["entity1", "entity2"]
            assert not indexer.manifest.has_file("test.py")

    def test_finalize(self):
        """Test finalizing update."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"
            indexer = IncrementalIndexer(FileManifest(manifest_path))

            indexer.update_file("test.py", ["entity1"])
            indexer.finalize()

            # Load in new instance
            manifest2 = FileManifest(manifest_path)
            assert manifest2.has_file("test.py")

    def test_get_indexed_count(self):
        """Test getting indexed file count."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"
            indexer = IncrementalIndexer(FileManifest(manifest_path))

            assert indexer.get_indexed_count() == 0

            indexer.update_file("file1.py", ["e1"])
            indexer.update_file("file2.py", ["e2"])

            assert indexer.get_indexed_count() == 2

    def test_clear_manifest(self):
        """Test clearing manifest for full rebuild."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"
            manifest = FileManifest(manifest_path)
            manifest.add_file("file1.py", "hash1")
            manifest.add_file("file2.py", "hash2")

            indexer = IncrementalIndexer(manifest)
            assert indexer.get_indexed_count() == 2

            indexer.clear_manifest()

            assert indexer.get_indexed_count() == 0
            assert indexer.manifest.files == {}

    def test_full_incremental_workflow(self):
        """Test complete incremental indexing workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"

            # First index: 3 files
            indexer1 = IncrementalIndexer(FileManifest(manifest_path))
            indexer1.update_file("file1.py", ["f1_e1", "f1_e2"])
            indexer1.update_file("file2.py", ["f2_e1"])
            indexer1.update_file("file3.py", ["f3_e1"])
            indexer1.finalize()

            assert indexer1.get_indexed_count() == 3

            # Second update: add, modify, delete
            indexer2 = IncrementalIndexer(FileManifest(manifest_path))
            changes = indexer2.prepare_update(["file1.py", "file2.py", "file4.py"])

            # file3.py is deleted
            deleted_entities = indexer2.remove_file("file3.py")
            assert deleted_entities == ["f3_e1"]

            # file1.py is updated
            indexer2.update_file("file1.py", ["f1_e1", "f1_e2", "f1_e3"])

            # file4.py is added
            indexer2.update_file("file4.py", ["f4_e1"])

            indexer2.finalize()

            # Verify final state
            indexer3 = IncrementalIndexer(FileManifest(manifest_path))
            assert indexer3.get_indexed_count() == 3  # file1, file2, file4
            assert not indexer3.manifest.has_file("file3.py")
