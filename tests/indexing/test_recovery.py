"""Tests for recovery infrastructure."""

import json
import tempfile
import time
from pathlib import Path

import pytest

from codesearch.indexing.recovery import (
    CheckpointManager,
    RecoveryState,
    InterruptionDetector,
)


class TestCheckpointManager:
    """Tests for CheckpointManager."""

    def test_create_checkpoint(self):
        """Test creating a checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(checkpoint_dir=Path(tmpdir))

            checkpoint_path = manager.create_checkpoint(
                session_id="test-session-1",
                processed_files=["file1.py", "file2.py"],
                failed_files=["file3.py"],
                metadata={"repo": "test"},
            )

            assert Path(checkpoint_path).exists()
            assert "test-session-1" in checkpoint_path

    def test_load_checkpoint(self):
        """Test loading a checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(checkpoint_dir=Path(tmpdir))

            processed = ["file1.py", "file2.py"]
            failed = ["file3.py"]

            manager.create_checkpoint(
                session_id="test-session-1",
                processed_files=processed,
                failed_files=failed,
                metadata={"repo": "test"},
            )

            checkpoint = manager.load_checkpoint("test-session-1")

            assert checkpoint is not None
            assert checkpoint["session_id"] == "test-session-1"
            assert checkpoint["processed_files"] == processed
            assert checkpoint["failed_files"] == failed
            assert checkpoint["metadata"]["repo"] == "test"

    def test_load_nonexistent_checkpoint(self):
        """Test loading nonexistent checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(checkpoint_dir=Path(tmpdir))

            checkpoint = manager.load_checkpoint("nonexistent")

            assert checkpoint is None

    def test_delete_checkpoint(self):
        """Test deleting a checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(checkpoint_dir=Path(tmpdir))

            manager.create_checkpoint(
                session_id="test-session-1",
                processed_files=["file1.py"],
                failed_files=[],
            )

            assert manager.load_checkpoint("test-session-1") is not None

            result = manager.delete_checkpoint("test-session-1")

            assert result is True
            assert manager.load_checkpoint("test-session-1") is None

    def test_delete_nonexistent_checkpoint(self):
        """Test deleting nonexistent checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(checkpoint_dir=Path(tmpdir))

            result = manager.delete_checkpoint("nonexistent")

            assert result is False

    def test_list_checkpoints(self):
        """Test listing checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(checkpoint_dir=Path(tmpdir))

            manager.create_checkpoint("session1", ["file1.py"], [])
            manager.create_checkpoint("session2", ["file2.py"], [])

            checkpoints = manager.list_checkpoints()

            assert len(checkpoints) == 2
            assert "session1" in checkpoints
            assert "session2" in checkpoints

    def test_get_checkpoint_age(self):
        """Test getting checkpoint age."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(checkpoint_dir=Path(tmpdir))

            manager.create_checkpoint("session1", ["file1.py"], [])

            # Small delay to ensure age > 0
            time.sleep(0.1)

            age = manager.get_checkpoint_age("session1")

            assert age is not None
            assert age > 0

    def test_get_nonexistent_checkpoint_age(self):
        """Test getting age of nonexistent checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(checkpoint_dir=Path(tmpdir))

            age = manager.get_checkpoint_age("nonexistent")

            assert age is None


class TestRecoveryState:
    """Tests for RecoveryState."""

    def test_create_recovery_state(self):
        """Test creating recovery state from checkpoint."""
        checkpoint_data = {
            "session_id": "test-session-1",
            "timestamp": "2025-11-05T10:00:00",
            "processed_files": ["file1.py", "file2.py"],
            "failed_files": ["file3.py"],
            "metadata": {"repo": "test"},
        }

        state = RecoveryState(checkpoint_data)

        assert state.session_id == "test-session-1"
        assert "file1.py" in state.processed_files
        assert "file3.py" in state.failed_files
        assert state.metadata["repo"] == "test"

    def test_get_files_to_reprocess(self):
        """Test getting files to reprocess."""
        checkpoint_data = {
            "session_id": "test-session-1",
            "timestamp": "2025-11-05T10:00:00",
            "processed_files": ["file1.py", "file2.py"],
            "failed_files": ["file3.py"],
            "metadata": {},
        }

        state = RecoveryState(checkpoint_data)
        all_files = ["file1.py", "file2.py", "file3.py", "file4.py", "file5.py"]

        to_reprocess = state.get_files_to_reprocess(all_files)

        assert len(to_reprocess) == 2
        assert "file4.py" in to_reprocess
        assert "file5.py" in to_reprocess

    def test_get_recovery_summary(self):
        """Test getting recovery summary."""
        checkpoint_data = {
            "session_id": "test-session-1",
            "timestamp": "2025-11-05T10:00:00",
            "processed_files": ["file1.py", "file2.py"],
            "failed_files": ["file3.py"],
            "metadata": {},
        }

        state = RecoveryState(checkpoint_data)
        summary = state.get_recovery_summary()

        assert summary["session_id"] == "test-session-1"
        assert summary["previously_processed"] == 2
        assert summary["previously_failed"] == 1


class TestInterruptionDetector:
    """Tests for InterruptionDetector."""

    def test_detect_interrupted_session(self):
        """Test detecting interrupted session."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(checkpoint_dir=Path(tmpdir))
            manager.create_checkpoint("session1", ["file1.py"], [])

            is_interrupted = InterruptionDetector.is_interrupted("session1", manager)

            assert is_interrupted is True

    def test_detect_non_interrupted_session(self):
        """Test detecting non-interrupted session."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(checkpoint_dir=Path(tmpdir))

            is_interrupted = InterruptionDetector.is_interrupted("nonexistent", manager)

            assert is_interrupted is False

    def test_recover_from_interruption(self):
        """Test recovering from interruption."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(checkpoint_dir=Path(tmpdir))
            manager.create_checkpoint(
                "session1",
                ["file1.py", "file2.py"],
                ["file3.py"],
            )

            recovery_state = InterruptionDetector.recover_from_interruption(
                "session1", manager
            )

            assert recovery_state is not None
            assert recovery_state.session_id == "session1"
            assert len(recovery_state.processed_files) == 2

    def test_recover_from_non_existent_interruption(self):
        """Test recovering from non-existent interruption."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(checkpoint_dir=Path(tmpdir))

            recovery_state = InterruptionDetector.recover_from_interruption(
                "nonexistent", manager
            )

            assert recovery_state is None

    def test_cleanup_old_checkpoints(self):
        """Test cleaning up old checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)
            manager = CheckpointManager(checkpoint_dir=checkpoint_dir)

            # Create multiple checkpoints
            manager.create_checkpoint("session1", ["file1.py"], [])
            time.sleep(0.1)
            manager.create_checkpoint("session2", ["file2.py"], [])

            # Cleanup with very short age (should delete old ones)
            deleted = InterruptionDetector.cleanup_old_checkpoints(
                manager, max_age_hours=0  # 0 hours = delete everything
            )

            # At least one checkpoint should be old enough to delete
            assert deleted >= 1
