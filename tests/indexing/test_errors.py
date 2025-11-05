"""Tests for error handling infrastructure."""

import pytest
from datetime import datetime

from codesearch.indexing.errors import (
    ErrorSeverity,
    ErrorCategory,
    IndexingError,
    IndexingSession,
    ErrorRecoveryStrategy,
    IndexIntegrityValidator,
)


class TestIndexingError:
    """Tests for IndexingError class."""

    def test_error_creation(self):
        """Test creating an indexing error."""
        error = IndexingError(
            file_path="test.py",
            category=ErrorCategory.PARSING,
            message="Syntax error",
            severity=ErrorSeverity.WARNING,
        )

        assert error.file_path == "test.py"
        assert error.category == ErrorCategory.PARSING
        assert error.message == "Syntax error"
        assert error.severity == ErrorSeverity.WARNING
        assert error.recoverable is True
        assert error.timestamp is not None

    def test_error_string_representation(self):
        """Test string representation of error."""
        error = IndexingError(
            file_path="test.py",
            category=ErrorCategory.PARSING,
            message="Syntax error",
            severity=ErrorSeverity.ERROR,
        )

        error_str = str(error)
        assert "ERROR" in error_str
        assert "parsing" in error_str
        assert "test.py" in error_str

    def test_error_to_dict(self):
        """Test converting error to dictionary."""
        error = IndexingError(
            file_path="test.py",
            category=ErrorCategory.FILE_READ,
            message="File not found",
            severity=ErrorSeverity.WARNING,
            recoverable=True,
        )

        error_dict = error.to_dict()
        assert error_dict["file_path"] == "test.py"
        assert error_dict["category"] == "file_read"
        assert error_dict["message"] == "File not found"
        assert error_dict["severity"] == "warning"
        assert error_dict["recoverable"] is True

    def test_error_with_exception(self):
        """Test error with exception details."""
        exc = FileNotFoundError("test.py not found")
        error = IndexingError(
            file_path="test.py",
            category=ErrorCategory.FILE_READ,
            message="Cannot read file",
            exception=exc,
        )

        assert error.exception is exc


class TestIndexingSession:
    """Tests for IndexingSession class."""

    def test_session_creation(self):
        """Test creating an indexing session."""
        session = IndexingSession(session_id="test-session-1")

        assert session.session_id == "test-session-1"
        assert session.start_time is not None
        assert session.end_time is None
        assert len(session.errors) == 0
        assert len(session.completed_files) == 0
        assert session.interrupted is False

    def test_add_error_to_session(self):
        """Test adding error to session."""
        session = IndexingSession(session_id="test-session-1")
        error = IndexingError(
            file_path="test.py",
            category=ErrorCategory.PARSING,
            message="Syntax error",
        )

        session.add_error(error)

        assert len(session.errors) == 1
        assert session.errors[0] == error

    def test_mark_file_completed(self):
        """Test marking file as completed."""
        session = IndexingSession(session_id="test-session-1")

        session.mark_file_completed("file1.py")
        session.mark_file_completed("file2.py")

        assert len(session.completed_files) == 2
        assert "file1.py" in session.completed_files

    def test_mark_file_skipped(self):
        """Test marking file as skipped."""
        session = IndexingSession(session_id="test-session-1")

        session.mark_file_skipped("file1.py")

        assert len(session.skipped_files) == 1
        assert "file1.py" in session.skipped_files

    def test_mark_file_failed(self):
        """Test marking file as failed."""
        session = IndexingSession(session_id="test-session-1")

        session.mark_file_failed("file1.py")

        assert len(session.failed_files) == 1
        assert "file1.py" in session.failed_files

    def test_get_error_summary(self):
        """Test getting error summary."""
        session = IndexingSession(session_id="test-session-1")

        session.mark_file_completed("file1.py")
        session.mark_file_completed("file2.py")
        session.mark_file_failed("file3.py")

        error1 = IndexingError(
            file_path="file3.py",
            category=ErrorCategory.PARSING,
            message="Syntax error",
            severity=ErrorSeverity.WARNING,
        )
        error2 = IndexingError(
            file_path="file3.py",
            category=ErrorCategory.FILE_READ,
            message="Cannot read file",
            severity=ErrorSeverity.ERROR,
        )

        session.add_error(error1)
        session.add_error(error2)
        session.finish()

        summary = session.get_error_summary()

        assert summary["total_errors"] == 2
        assert summary["completed_files"] == 2
        assert summary["failed_files"] == 1
        assert summary["success_rate"] == 2/3
        assert "parsing" in summary["errors_by_category"]
        assert "warning" in summary["errors_by_severity"]

    def test_session_finish(self):
        """Test finishing session."""
        session = IndexingSession(session_id="test-session-1")
        assert session.end_time is None

        session.finish()

        assert session.end_time is not None
        assert session.end_time >= session.start_time


class TestErrorRecoveryStrategy:
    """Tests for ErrorRecoveryStrategy."""

    def test_should_retry_recoverable_error(self):
        """Test retry decision for recoverable error."""
        strategy = ErrorRecoveryStrategy()
        error = IndexingError(
            file_path="test.py",
            category=ErrorCategory.DATABASE,
            message="Connection timeout",
            recoverable=True,
        )

        # First attempt should retry
        assert strategy.should_retry(error, attempt=1) is True
        # Second attempt should retry
        assert strategy.should_retry(error, attempt=2) is True
        # Third attempt should not retry (max 3)
        assert strategy.should_retry(error, attempt=3) is False

    def test_should_not_retry_non_recoverable_error(self):
        """Test no retry for non-recoverable error."""
        strategy = ErrorRecoveryStrategy()
        error = IndexingError(
            file_path="test.py",
            category=ErrorCategory.PARSING,
            message="Syntax error",
            recoverable=False,
        )

        assert strategy.should_retry(error, attempt=1) is False

    def test_should_continue_after_non_critical_error(self):
        """Test continuation after non-critical error."""
        strategy = ErrorRecoveryStrategy()
        session = IndexingSession(session_id="test-session-1")
        error = IndexingError(
            file_path="test.py",
            category=ErrorCategory.PARSING,
            message="Syntax error",
            severity=ErrorSeverity.WARNING,
        )

        assert strategy.should_continue(error, session) is True

    def test_should_continue_after_critical_error(self):
        """Test continuation after critical error."""
        strategy = ErrorRecoveryStrategy()
        session = IndexingSession(session_id="test-session-1")

        # Add critical errors to session
        for i in range(3):
            error = IndexingError(
                file_path=f"file{i}.py",
                category=ErrorCategory.DATABASE,
                message="Critical error",
                severity=ErrorSeverity.CRITICAL,
            )
            session.add_error(error)

        # Should continue with few critical errors
        new_error = IndexingError(
            file_path="file4.py",
            category=ErrorCategory.DATABASE,
            message="Critical error",
            severity=ErrorSeverity.CRITICAL,
        )

        assert strategy.should_continue(new_error, session) is True

        # Add more errors
        for i in range(5, 8):
            error = IndexingError(
                file_path=f"file{i}.py",
                category=ErrorCategory.DATABASE,
                message="Critical error",
                severity=ErrorSeverity.CRITICAL,
            )
            session.add_error(error)

        # Should not continue with many critical errors
        assert strategy.should_continue(new_error, session) is False

    def test_get_retry_delay(self):
        """Test exponential backoff retry delay."""
        strategy = ErrorRecoveryStrategy()

        # Exponential backoff: 0.1, 0.2, 0.4
        assert strategy.get_retry_delay(1) == 0.1
        assert strategy.get_retry_delay(2) == 0.2
        assert strategy.get_retry_delay(3) == 0.4


class TestIndexIntegrityValidator:
    """Tests for IndexIntegrityValidator."""

    def test_validate_valid_manifest(self):
        """Test validating a valid manifest."""
        validator = IndexIntegrityValidator()
        manifest = {
            "files": {
                "file1.py": {
                    "hash": "abc123",
                    "timestamp": "2025-11-05T10:00:00",
                    "entities": ["func1"],
                },
            }
        }

        errors = validator.validate_manifest(manifest)

        assert len(errors) == 0

    def test_validate_invalid_manifest_type(self):
        """Test validating invalid manifest type."""
        validator = IndexIntegrityValidator()

        errors = validator.validate_manifest("not a dict")

        assert len(errors) == 1
        assert errors[0].category == ErrorCategory.MANIFEST
        assert errors[0].severity == ErrorSeverity.CRITICAL

    def test_validate_manifest_missing_hash(self):
        """Test validating manifest with missing hash."""
        validator = IndexIntegrityValidator()
        manifest = {
            "files": {
                "file1.py": {
                    "timestamp": "2025-11-05T10:00:00",
                },
            }
        }

        errors = validator.validate_manifest(manifest)

        assert len(errors) >= 1
        assert any(e.message == "File entry missing 'hash' field" for e in errors)

    def test_check_file_consistency_match(self):
        """Test checking file hash consistency (match)."""
        validator = IndexIntegrityValidator()

        error = validator.check_file_consistency(
            "file1.py",
            "abc123",
            "abc123",
        )

        assert error is None

    def test_check_file_consistency_mismatch(self):
        """Test checking file hash consistency (mismatch)."""
        validator = IndexIntegrityValidator()

        error = validator.check_file_consistency(
            "file1.py",
            "abc123",
            "def456",
        )

        assert error is not None
        assert error.category == ErrorCategory.INTEGRITY
        assert "hash mismatch" in error.message.lower()
