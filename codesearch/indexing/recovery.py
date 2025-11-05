"""Recovery infrastructure for interrupted indexing operations."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages checkpoints for recovery from interrupted indexing."""

    def __init__(self, checkpoint_dir: Optional[Path] = None):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoint files
        """
        self.checkpoint_dir = checkpoint_dir or (
            Path.home() / ".codesearch" / "checkpoints"
        )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def create_checkpoint(
        self,
        session_id: str,
        processed_files: List[str],
        failed_files: List[str],
        metadata: Optional[Dict] = None,
    ) -> str:
        """
        Create a checkpoint for recovery.

        Args:
            session_id: Session identifier
            processed_files: List of successfully processed files
            failed_files: List of files that failed
            metadata: Additional metadata to store

        Returns:
            Path to checkpoint file
        """
        checkpoint_data = {
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat(),
            "processed_files": processed_files,
            "failed_files": failed_files,
            "metadata": metadata or {},
        }

        checkpoint_path = self.checkpoint_dir / f"{session_id}.checkpoint.json"

        try:
            with open(checkpoint_path, "w") as f:
                json.dump(checkpoint_data, f, indent=2)
            logger.debug(f"Created checkpoint: {checkpoint_path}")
            return str(checkpoint_path)
        except Exception as e:
            logger.error(f"Failed to create checkpoint: {e}")
            raise

    def load_checkpoint(self, session_id: str) -> Optional[Dict]:
        """
        Load a checkpoint to resume indexing.

        Args:
            session_id: Session identifier

        Returns:
            Checkpoint data or None if not found
        """
        checkpoint_path = self.checkpoint_dir / f"{session_id}.checkpoint.json"

        if not checkpoint_path.exists():
            logger.debug(f"No checkpoint found for session: {session_id}")
            return None

        try:
            with open(checkpoint_path, "r") as f:
                data = json.load(f)
            logger.debug(f"Loaded checkpoint: {checkpoint_path}")
            return data
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None

    def list_checkpoints(self) -> List[str]:
        """
        List all available checkpoints.

        Returns:
            List of session IDs with checkpoints
        """
        checkpoints = []
        for checkpoint_file in self.checkpoint_dir.glob("*.checkpoint.json"):
            session_id = checkpoint_file.stem.replace(".checkpoint", "")
            checkpoints.append(session_id)
        return checkpoints

    def delete_checkpoint(self, session_id: str) -> bool:
        """
        Delete a checkpoint after successful indexing.

        Args:
            session_id: Session identifier

        Returns:
            True if deleted, False if not found
        """
        checkpoint_path = self.checkpoint_dir / f"{session_id}.checkpoint.json"

        if not checkpoint_path.exists():
            return False

        try:
            checkpoint_path.unlink()
            logger.debug(f"Deleted checkpoint: {checkpoint_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete checkpoint: {e}")
            return False

    def get_checkpoint_age(self, session_id: str) -> Optional[float]:
        """
        Get age of checkpoint in seconds.

        Args:
            session_id: Session identifier

        Returns:
            Age in seconds or None if not found
        """
        checkpoint_path = self.checkpoint_dir / f"{session_id}.checkpoint.json"

        if not checkpoint_path.exists():
            return None

        try:
            stat = checkpoint_path.stat()
            age = (datetime.utcnow() - datetime.fromtimestamp(stat.st_mtime)).total_seconds()
            return age
        except Exception as e:
            logger.error(f"Failed to get checkpoint age: {e}")
            return None


class RecoveryState:
    """Represents the state to recover from."""

    def __init__(self, checkpoint_data: Dict):
        """
        Initialize recovery state from checkpoint.

        Args:
            checkpoint_data: Checkpoint data dictionary
        """
        self.session_id = checkpoint_data["session_id"]
        self.checkpoint_timestamp = checkpoint_data["timestamp"]
        self.processed_files = set(checkpoint_data["processed_files"])
        self.failed_files = set(checkpoint_data["failed_files"])
        self.metadata = checkpoint_data.get("metadata", {})

    def get_files_to_reprocess(self, all_files: List[str]) -> List[str]:
        """
        Get list of files that need to be processed in recovery.

        Args:
            all_files: List of all files to index

        Returns:
            List of files not yet processed
        """
        processed = self.processed_files | self.failed_files
        return [f for f in all_files if f not in processed]

    def get_recovery_summary(self) -> Dict:
        """Get summary of recovery state."""
        return {
            "session_id": self.session_id,
            "checkpoint_timestamp": self.checkpoint_timestamp,
            "previously_processed": len(self.processed_files),
            "previously_failed": len(self.failed_files),
            "files_to_reprocess": len(self.get_files_to_reprocess([])),
        }


class InterruptionDetector:
    """Detects and handles interruptions in indexing."""

    @staticmethod
    def is_interrupted(session_id: str, checkpoint_manager: CheckpointManager) -> bool:
        """
        Check if a session was interrupted.

        Args:
            session_id: Session identifier
            checkpoint_manager: CheckpointManager instance

        Returns:
            True if session has a checkpoint (was interrupted)
        """
        return checkpoint_manager.load_checkpoint(session_id) is not None

    @staticmethod
    def recover_from_interruption(
        session_id: str, checkpoint_manager: CheckpointManager
    ) -> Optional[RecoveryState]:
        """
        Recover from an interrupted session.

        Args:
            session_id: Session identifier
            checkpoint_manager: CheckpointManager instance

        Returns:
            RecoveryState or None if no checkpoint found
        """
        checkpoint_data = checkpoint_manager.load_checkpoint(session_id)

        if checkpoint_data is None:
            return None

        return RecoveryState(checkpoint_data)

    @staticmethod
    def cleanup_old_checkpoints(
        checkpoint_manager: CheckpointManager, max_age_hours: int = 24
    ) -> int:
        """
        Clean up old checkpoints.

        Args:
            checkpoint_manager: CheckpointManager instance
            max_age_hours: Maximum age in hours to keep

        Returns:
            Number of checkpoints deleted
        """
        deleted = 0
        max_age_seconds = max_age_hours * 3600

        for session_id in checkpoint_manager.list_checkpoints():
            age = checkpoint_manager.get_checkpoint_age(session_id)

            if age is not None and age > max_age_seconds:
                if checkpoint_manager.delete_checkpoint(session_id):
                    deleted += 1
                    logger.debug(f"Deleted old checkpoint for {session_id}")

        return deleted
