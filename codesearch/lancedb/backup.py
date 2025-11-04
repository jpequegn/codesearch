"""Backup and restore functionality for LanceDB databases."""

import lancedb
from pathlib import Path
from typing import Optional, Dict
import logging
import shutil
from datetime import datetime, timezone
import json


logger = logging.getLogger(__name__)


class DatabaseBackupManager:
    """Manages database backups and restore operations.

    Features:
    - Full database backups to timestamped directories
    - Restore from backup with validation
    - Backup metadata tracking
    - Incremental backup support (future)
    """

    def __init__(self, db_path: Optional[Path] = None, backup_dir: Optional[Path] = None):
        """Initialize backup manager.

        Args:
            db_path: Path to LanceDB directory
            backup_dir: Directory for backups (default: db_path/backups/)
        """
        self.db_path = db_path or Path(".lancedb")
        self.backup_dir = backup_dir or self.db_path / "backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Backup manager initialized for {self.db_path}")

    def backup(self, backup_name: Optional[str] = None) -> Dict:
        """Create a full database backup.

        Args:
            backup_name: Custom backup name (default: timestamp-based)

        Returns:
            Dictionary with backup metadata including path, size, timestamp
        """
        # Generate backup name if not provided
        if backup_name is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            backup_name = f"backup_{timestamp}"

        backup_path = self.backup_dir / backup_name

        try:
            # Check if database exists
            if not self.db_path.exists():
                raise FileNotFoundError(f"Database path not found: {self.db_path}")

            # Copy entire database directory, excluding backups subdirectory
            def ignore_patterns(directory, contents):
                # Don't copy the backups directory itself to avoid recursion
                return {"backups"} if "backups" in contents else set()

            shutil.copytree(self.db_path, backup_path, dirs_exist_ok=False, ignore=ignore_patterns)

            # Calculate backup size
            backup_size = self._get_directory_size(backup_path)

            # Create backup metadata
            metadata = {
                "name": backup_name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "path": str(backup_path),
                "size_bytes": backup_size,
                "type": "full",
                "status": "completed",
            }

            # Save backup metadata
            self._save_backup_metadata(backup_name, metadata)

            logger.info(
                f"Created backup '{backup_name}' at {backup_path} "
                f"({self._format_size(backup_size)})"
            )

            return metadata

        except Exception as e:
            logger.error(f"Backup failed: {e}")
            # Clean up failed backup
            if backup_path.exists():
                shutil.rmtree(backup_path)
            raise

    def restore(self, backup_name: str, restore_path: Optional[Path] = None) -> Dict:
        """Restore database from a backup.

        Args:
            backup_name: Name of backup to restore
            restore_path: Where to restore to (default: overwrites current db_path)

        Returns:
            Dictionary with restore metadata and status
        """
        backup_path = self.backup_dir / backup_name

        if not backup_path.exists():
            raise FileNotFoundError(f"Backup not found: {backup_path}")

        restore_path = restore_path or self.db_path

        try:
            # Create temporary directory for the current database before overwriting
            temp_path = self.db_path.parent / f"{self.db_path.name}_restore_backup"

            # Move current database aside if restoring to same location
            if restore_path == self.db_path and self.db_path.exists():
                if temp_path.exists():
                    shutil.rmtree(temp_path)
                shutil.move(str(self.db_path), str(temp_path))

            # Restore from backup
            shutil.copytree(backup_path, restore_path, dirs_exist_ok=True)

            # Clean up temp backup
            if temp_path.exists():
                shutil.rmtree(temp_path)

            metadata = {
                "backup_name": backup_name,
                "restored_to": str(restore_path),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": "completed",
            }

            logger.info(f"Restored database from backup '{backup_name}'")

            return metadata

        except Exception as e:
            logger.error(f"Restore failed: {e}")
            # Attempt to restore from temp if available
            if temp_path.exists():
                try:
                    if self.db_path.exists():
                        shutil.rmtree(self.db_path)
                    shutil.move(str(temp_path), str(self.db_path))
                    logger.info("Rolled back to pre-restore database")
                except Exception as rollback_e:
                    logger.error(f"Rollback failed: {rollback_e}")
            raise

    def list_backups(self) -> list:
        """List all available backups.

        Returns:
            List of backup names with metadata
        """
        backups = []

        try:
            for backup_path in sorted(self.backup_dir.iterdir()):
                if backup_path.is_dir():
                    metadata_file = backup_path / ".backup_metadata.json"
                    if metadata_file.exists():
                        try:
                            with open(metadata_file, "r") as f:
                                metadata = json.load(f)
                                backups.append(metadata)
                        except Exception as e:
                            logger.warning(f"Could not read metadata for {backup_path.name}: {e}")

            return backups

        except Exception as e:
            logger.error(f"Error listing backups: {e}")
            return []

    def get_backup_info(self, backup_name: str) -> Dict:
        """Get metadata for a specific backup.

        Args:
            backup_name: Name of backup

        Returns:
            Backup metadata dictionary
        """
        metadata_file = self.backup_dir / backup_name / ".backup_metadata.json"

        try:
            if metadata_file.exists():
                with open(metadata_file, "r") as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error reading backup info: {e}")

        return {}

    def delete_backup(self, backup_name: str, confirm: bool = False) -> bool:
        """Delete a backup.

        Args:
            backup_name: Name of backup to delete
            confirm: Must be True to prevent accidental deletion

        Returns:
            True if deletion successful
        """
        if not confirm:
            logger.warning(f"Delete backup '{backup_name}' requested but not confirmed")
            return False

        backup_path = self.backup_dir / backup_name

        try:
            if backup_path.exists():
                shutil.rmtree(backup_path)
                logger.info(f"Deleted backup '{backup_name}'")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete backup: {e}")
            return False

    def cleanup_old_backups(self, keep_count: int = 5) -> Dict:
        """Clean up old backups, keeping only the most recent ones.

        Args:
            keep_count: Number of backups to keep (default: 5)

        Returns:
            Dictionary with cleanup statistics
        """
        try:
            backups = self.list_backups()
            backups_sorted = sorted(backups, key=lambda x: x.get("timestamp", ""), reverse=True)

            deleted_count = 0
            freed_space = 0

            for backup in backups_sorted[keep_count:]:
                backup_name = backup.get("name")
                backup_size = backup.get("size_bytes", 0)

                if self.delete_backup(backup_name, confirm=True):
                    deleted_count += 1
                    freed_space += backup_size

            logger.info(
                f"Cleaned up {deleted_count} backups, freed {self._format_size(freed_space)}"
            )

            return {
                "deleted_count": deleted_count,
                "freed_space_bytes": freed_space,
                "freed_space": self._format_size(freed_space),
                "remaining_backups": keep_count,
            }

        except Exception as e:
            logger.error(f"Error cleaning up backups: {e}")
            return {"error": str(e)}

    def _save_backup_metadata(self, backup_name: str, metadata: Dict):
        """Save backup metadata to file.

        Args:
            backup_name: Name of backup
            metadata: Metadata dictionary to save
        """
        metadata_file = self.backup_dir / backup_name / ".backup_metadata.json"

        try:
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save backup metadata: {e}")

    @staticmethod
    def _get_directory_size(path: Path) -> int:
        """Calculate total size of a directory in bytes.

        Args:
            path: Directory path

        Returns:
            Total size in bytes
        """
        total = 0
        try:
            for item in path.rglob("*"):
                if item.is_file():
                    total += item.stat().st_size
        except Exception as e:
            logger.error(f"Error calculating directory size: {e}")

        return total

    @staticmethod
    def _format_size(bytes_size: int) -> str:
        """Format bytes to human-readable size.

        Args:
            bytes_size: Size in bytes

        Returns:
            Formatted string (e.g., "1.5 MB")
        """
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if bytes_size < 1024:
                return f"{bytes_size:.1f} {unit}"
            bytes_size /= 1024
        return f"{bytes_size:.1f} PB"
