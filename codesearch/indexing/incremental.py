"""Incremental indexing support for efficient index updates."""

import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class FileManifest:
    """Tracks indexed files and their metadata for incremental indexing."""

    def __init__(self, manifest_path: Optional[Path] = None):
        """Initialize file manifest.

        Args:
            manifest_path: Path to store/load manifest file
        """
        self.manifest_path = manifest_path or Path.home() / ".codesearch" / "manifest.json"
        self.files: Dict[str, dict] = {}
        self.load()

    def load(self) -> None:
        """Load manifest from disk."""
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, "r") as f:
                    data = json.load(f)
                    self.files = data.get("files", {})
                logger.debug(f"Loaded manifest with {len(self.files)} files")
            except Exception as e:
                logger.warning(f"Failed to load manifest: {e}")
                self.files = {}

    def save(self) -> None:
        """Save manifest to disk."""
        try:
            self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.manifest_path, "w") as f:
                json.dump({"files": self.files}, f, indent=2)
            logger.debug(f"Saved manifest with {len(self.files)} files")
        except Exception as e:
            logger.error(f"Failed to save manifest: {e}")

    def add_file(
        self, file_path: str, file_hash: str, entities: Optional[List[str]] = None
    ) -> None:
        """Add or update file in manifest.

        Args:
            file_path: Path to file
            file_hash: Hash of file contents
            entities: List of entity IDs extracted from file
        """
        self.files[file_path] = {
            "hash": file_hash,
            "timestamp": datetime.now().isoformat(),
            "entities": entities or [],
        }

    def remove_file(self, file_path: str) -> List[str]:
        """Remove file from manifest and return its entities.

        Args:
            file_path: Path to file

        Returns:
            List of entity IDs that were in this file
        """
        if file_path in self.files:
            entities = self.files[file_path].get("entities", [])
            del self.files[file_path]
            return entities
        return []

    def get_file_hash(self, file_path: str) -> Optional[str]:
        """Get hash of file from manifest.

        Args:
            file_path: Path to file

        Returns:
            File hash or None if not found
        """
        if file_path in self.files:
            return self.files[file_path].get("hash")
        return None

    def get_file_entities(self, file_path: str) -> List[str]:
        """Get entities from a file.

        Args:
            file_path: Path to file

        Returns:
            List of entity IDs from this file
        """
        if file_path in self.files:
            return self.files[file_path].get("entities", [])
        return []

    def has_file(self, file_path: str) -> bool:
        """Check if file is in manifest.

        Args:
            file_path: Path to file

        Returns:
            True if file is in manifest
        """
        return file_path in self.files


class ChangeDetector:
    """Detects changes in repository files."""

    @staticmethod
    def compute_file_hash(file_path: str) -> str:
        """Compute hash of file contents.

        Args:
            file_path: Path to file

        Returns:
            SHA256 hash of file contents
        """
        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
        except Exception as e:
            logger.warning(f"Failed to hash file {file_path}: {e}")
            return ""
        return sha256_hash.hexdigest()

    @staticmethod
    def detect_changes(
        current_files: List[str], manifest: FileManifest
    ) -> Tuple[List[str], List[str], List[str]]:
        """Detect changes in files.

        Args:
            current_files: List of current file paths
            manifest: File manifest

        Returns:
            Tuple of (added_files, modified_files, deleted_files)
        """
        current_set = set(current_files)
        manifest_set = set(manifest.files.keys())

        # Added files (in current but not in manifest)
        added = list(current_set - manifest_set)

        # Deleted files (in manifest but not in current)
        deleted = list(manifest_set - current_set)

        # Check for modified files
        modified = []
        for file_path in current_set & manifest_set:
            current_hash = ChangeDetector.compute_file_hash(file_path)
            stored_hash = manifest.get_file_hash(file_path)
            if current_hash and stored_hash and current_hash != stored_hash:
                modified.append(file_path)

        return added, modified, deleted


class IncrementalIndexer:
    """Manages incremental index updates."""

    def __init__(self, manifest: Optional[FileManifest] = None):
        """Initialize incremental indexer.

        Args:
            manifest: File manifest to use
        """
        self.manifest = manifest or FileManifest()
        self.detector = ChangeDetector()

    def prepare_update(
        self, current_files: List[str]
    ) -> Dict[str, List[str]]:
        """Prepare an incremental index update.

        Args:
            current_files: List of current file paths

        Returns:
            Dict with 'added', 'modified', 'deleted' file lists
        """
        added, modified, deleted = self.detector.detect_changes(
            current_files, self.manifest
        )

        logger.info(f"Incremental update: +{len(added)} ~{len(modified)} -{len(deleted)}")

        return {"added": added, "modified": modified, "deleted": deleted}

    def update_file(
        self, file_path: str, entities: Optional[List[str]] = None
    ) -> None:
        """Update file in manifest after indexing.

        Args:
            file_path: Path to file
            entities: List of entity IDs extracted from file
        """
        file_hash = self.detector.compute_file_hash(file_path)
        self.manifest.add_file(file_path, file_hash, entities or [])

    def remove_file(self, file_path: str) -> List[str]:
        """Remove file from manifest.

        Args:
            file_path: Path to file

        Returns:
            List of entity IDs to remove from index
        """
        return self.manifest.remove_file(file_path)

    def finalize(self) -> None:
        """Save manifest after update."""
        self.manifest.save()

    def get_indexed_count(self) -> int:
        """Get total number of indexed files.

        Returns:
            Number of files in manifest
        """
        return len(self.manifest.files)

    def clear_manifest(self) -> None:
        """Clear the manifest (for full rebuild)."""
        self.manifest.files = {}
        self.manifest.save()
        logger.info("Cleared manifest for full rebuild")
