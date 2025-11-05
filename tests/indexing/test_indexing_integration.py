"""Integration tests for the indexing pipeline."""

import tempfile
from pathlib import Path
import uuid

import pytest

from codesearch.indexing.scanner import RepositoryScannerImpl
from codesearch.indexing.errors import IndexingSession, ErrorRecoveryStrategy
from codesearch.indexing.recovery import CheckpointManager
from codesearch.parsers.python_parser import PythonParser


class TestIndexingScannerIntegration:
    """Integration tests for repository scanning."""

    def test_scan_simple_repository(self):
        """Test scanning a simple repository with Python files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)

            # Create sample Python files
            (repo_path / "module1.py").write_text("""
def function1():
    pass

class Class1:
    pass
""")
            (repo_path / "module2.py").write_text("""
def function2():
    pass
""")

            scanner = RepositoryScannerImpl()
            files = scanner.scan_repository(str(repo_path), "test_repo")

            assert len(files) == 2
            assert any(f.file_path.endswith("module1.py") for f in files)
            assert any(f.file_path.endswith("module2.py") for f in files)

    def test_scan_ignores_ignored_patterns(self):
        """Test that scanner respects ignore patterns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)

            # Create files including ignored ones
            (repo_path / "main.py").write_text("def main(): pass")
            (repo_path / "test.pyc").write_text("compiled")
            (repo_path / "__pycache__").mkdir()
            (repo_path / "__pycache__" / "test.pyc").write_text("compiled")

            scanner = RepositoryScannerImpl()
            files = scanner.scan_repository(str(repo_path), "test_repo")

            # Should find Python files but not compiled bytecode
            python_files = [f for f in files if f.file_path.endswith(".py")]
            assert len(python_files) >= 1

    def test_scan_nested_directories(self):
        """Test scanning nested directory structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)

            # Create nested structure
            (repo_path / "src").mkdir()
            (repo_path / "src" / "main.py").write_text("def main(): pass")
            (repo_path / "src" / "utils").mkdir()
            (repo_path / "src" / "utils" / "helpers.py").write_text("def helper(): pass")
            (repo_path / "tests").mkdir()
            (repo_path / "tests" / "test_main.py").write_text("def test(): pass")

            scanner = RepositoryScannerImpl()
            files = scanner.scan_repository(str(repo_path), "test_repo")

            assert len(files) >= 3
            assert any("src" in f.file_path and "main.py" in f.file_path for f in files)
            assert any("helpers.py" in f.file_path for f in files)

    def test_get_statistics(self):
        """Test getting statistics from scan."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)

            (repo_path / "file1.py").write_text("# code")
            (repo_path / "file2.py").write_text("# code")
            (repo_path / "file3.py").write_text("# code")

            scanner = RepositoryScannerImpl()
            files = scanner.scan_repository(str(repo_path), "test_repo")
            stats = scanner.get_statistics()

            assert stats["total_files"] == 3
            assert stats["by_repository"]["test_repo"] == 3
            assert "python" in stats["by_language"]

    def test_scan_empty_repository(self):
        """Test scanning empty repository."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scanner = RepositoryScannerImpl()
            files = scanner.scan_repository(str(tmpdir), "empty_repo")

            assert len(files) == 0

    def test_scan_nonexistent_repository(self):
        """Test scanning nonexistent repository."""
        scanner = RepositoryScannerImpl()

        with pytest.raises(ValueError):
            scanner.scan_repository("/nonexistent/path", "bad_repo")

    def test_scan_multiple_repositories(self):
        """Test scanning multiple repositories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            repo1 = tmpdir / "repo1"
            repo2 = tmpdir / "repo2"

            repo1.mkdir()
            repo2.mkdir()

            (repo1 / "file1.py").write_text("def f1(): pass")
            (repo2 / "file2.py").write_text("def f2(): pass")

            scanner = RepositoryScannerImpl()
            results = scanner.scan_multiple_repositories({
                "repo1": str(repo1),
                "repo2": str(repo2),
            })

            assert len(results) == 2
            assert len(results["repo1"]) == 1
            assert len(results["repo2"]) == 1


class TestIndexingSessionIntegration:
    """Integration tests for indexing sessions."""

    def test_full_indexing_session_with_errors(self):
        """Test complete indexing session with error tracking."""
        session = IndexingSession(session_id=str(uuid.uuid4()))

        # Simulate indexing
        for i in range(5):
            session.mark_file_completed(f"file{i}.py")

        for i in range(2):
            session.mark_file_failed(f"bad{i}.py")

        session.finish()

        summary = session.get_error_summary()

        assert summary["completed_files"] == 5
        assert summary["failed_files"] == 2
        assert summary["success_rate"] == 5/7
        assert summary["duration_seconds"] >= 0

    def test_recovery_after_interruption(self):
        """Test recovery from interrupted session."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_manager = CheckpointManager(checkpoint_dir=Path(tmpdir))

            # Create checkpoint
            checkpoint_path = checkpoint_manager.create_checkpoint(
                session_id="interrupted-session",
                processed_files=["file1.py", "file2.py"],
                failed_files=["file3.py"],
                metadata={"repo": "test"},
            )

            assert Path(checkpoint_path).exists()

            # Load and verify
            checkpoint = checkpoint_manager.load_checkpoint("interrupted-session")
            assert checkpoint is not None
            assert len(checkpoint["processed_files"]) == 2
            assert len(checkpoint["failed_files"]) == 1

            # Cleanup
            result = checkpoint_manager.delete_checkpoint("interrupted-session")
            assert result is True
            assert checkpoint_manager.load_checkpoint("interrupted-session") is None


class TestErrorRecoveryStrategyIntegration:
    """Integration tests for error recovery."""

    def test_recovery_strategy_decision_flow(self):
        """Test complete recovery strategy decision flow."""
        from codesearch.indexing.errors import (
            IndexingError,
            ErrorCategory,
            ErrorSeverity,
        )

        strategy = ErrorRecoveryStrategy()
        session = IndexingSession(session_id=str(uuid.uuid4()))

        # Test with non-recoverable error
        error = IndexingError(
            file_path="test.py",
            category=ErrorCategory.PARSING,
            message="Syntax error",
            severity=ErrorSeverity.WARNING,
            recoverable=False,
        )

        session.add_error(error)

        # Non-recoverable errors shouldn't retry
        assert not strategy.should_retry(error, attempt=1)

        # Non-critical errors should continue
        assert strategy.should_continue(error, session)

        # Test with critical errors
        session2 = IndexingSession(session_id=str(uuid.uuid4()))
        for i in range(6):
            crit_error = IndexingError(
                file_path=f"crit{i}.py",
                category=ErrorCategory.DATABASE,
                message="Critical error",
                severity=ErrorSeverity.CRITICAL,
            )
            session2.add_error(crit_error)

        # After many critical errors, new critical errors should not continue
        new_error = IndexingError(
            file_path="new_crit.py",
            category=ErrorCategory.DATABASE,
            message="Another critical error",
            severity=ErrorSeverity.CRITICAL,
        )
        assert not strategy.should_continue(new_error, session2)

    def test_parser_integration_with_error_handling(self):
        """Test parser integration with error handling."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)

            # Create valid and invalid Python files
            (repo_path / "valid.py").write_text("""
def valid_function():
    return "valid"
""")
            (repo_path / "invalid.py").write_text("def broken(:\n")

            parser = PythonParser()
            session = IndexingSession(session_id=str(uuid.uuid4()))

            # Parse valid file
            try:
                entities = parser.parse_file(str(repo_path / "valid.py"))
                if entities:
                    session.mark_file_completed(str(repo_path / "valid.py"))
            except Exception as e:
                session.mark_file_failed(str(repo_path / "valid.py"))

            # Parse invalid file
            try:
                entities = parser.parse_file(str(repo_path / "invalid.py"))
                if entities:
                    session.mark_file_completed(str(repo_path / "invalid.py"))
            except SyntaxError as e:
                session.mark_file_failed(str(repo_path / "invalid.py"))

            session.finish()
            summary = session.get_error_summary()

            assert summary["completed_files"] == 1
            assert summary["failed_files"] == 1
            assert summary["success_rate"] == 0.5
