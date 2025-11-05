"""Error handling and recovery infrastructure for indexing operations."""

import logging
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Severity levels for indexing errors."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Categories of errors that can occur during indexing."""

    FILE_READ = "file_read"  # Error reading file from disk
    PARSING = "parsing"  # Error parsing file content
    VALIDATION = "validation"  # Error validating extracted data
    EXTRACTION = "extraction"  # Error extracting entities
    DATABASE = "database"  # Error storing in database
    MANIFEST = "manifest"  # Error managing manifest
    INTERRUPTION = "interruption"  # Indexing was interrupted
    INTEGRITY = "integrity"  # Index integrity check failed


@dataclass
class IndexingError:
    """Represents a single indexing error."""

    file_path: str
    category: ErrorCategory
    message: str
    severity: ErrorSeverity = ErrorSeverity.ERROR
    timestamp: datetime = field(default_factory=datetime.utcnow)
    exception: Optional[Exception] = None
    recoverable: bool = True
    context: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        """Return string representation of error."""
        return f"[{self.severity.value.upper()}] {self.category.value}: {self.message} (file: {self.file_path})"

    def to_dict(self) -> dict:
        """Convert error to dictionary for logging/reporting."""
        return {
            "file_path": self.file_path,
            "category": self.category.value,
            "message": self.message,
            "severity": self.severity.value,
            "timestamp": self.timestamp.isoformat(),
            "recoverable": self.recoverable,
            "context": self.context,
        }


@dataclass
class IndexingSession:
    """Tracks a complete indexing session including errors and recovery state."""

    session_id: str
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    errors: List[IndexingError] = field(default_factory=list)
    completed_files: List[str] = field(default_factory=list)
    skipped_files: List[str] = field(default_factory=list)
    failed_files: List[str] = field(default_factory=list)
    interrupted: bool = False
    checkpoint_path: Optional[str] = None

    def add_error(self, error: IndexingError) -> None:
        """Add an error to the session."""
        self.errors.append(error)
        logger.log(
            level=getattr(logging, error.severity.value.upper()),
            msg=str(error)
        )

    def mark_file_completed(self, file_path: str) -> None:
        """Mark a file as successfully processed."""
        self.completed_files.append(file_path)

    def mark_file_skipped(self, file_path: str) -> None:
        """Mark a file as skipped."""
        self.skipped_files.append(file_path)

    def mark_file_failed(self, file_path: str) -> None:
        """Mark a file as failed."""
        self.failed_files.append(file_path)

    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of errors in this session."""
        errors_by_category = {}
        errors_by_severity = {}

        for error in self.errors:
            cat = error.category.value
            sev = error.severity.value

            errors_by_category[cat] = errors_by_category.get(cat, 0) + 1
            errors_by_severity[sev] = errors_by_severity.get(sev, 0) + 1

        duration = (self.end_time or datetime.utcnow()) - self.start_time

        return {
            "total_errors": len(self.errors),
            "errors_by_category": errors_by_category,
            "errors_by_severity": errors_by_severity,
            "completed_files": len(self.completed_files),
            "skipped_files": len(self.skipped_files),
            "failed_files": len(self.failed_files),
            "duration_seconds": duration.total_seconds(),
            "success_rate": self._calculate_success_rate(),
            "interrupted": self.interrupted,
        }

    def _calculate_success_rate(self) -> float:
        """Calculate success rate for this session."""
        total = len(self.completed_files) + len(self.skipped_files) + len(self.failed_files)
        if total == 0:
            return 0.0
        return len(self.completed_files) / total

    def finish(self) -> None:
        """Mark session as finished."""
        self.end_time = datetime.utcnow()


class ErrorRecoveryStrategy:
    """Strategy for recovering from errors during indexing."""

    def should_retry(self, error: IndexingError, attempt: int = 1) -> bool:
        """Determine if an error should be retried."""
        if not error.recoverable:
            return False

        # Retry transient errors up to 3 times
        if error.category in [ErrorCategory.DATABASE, ErrorCategory.FILE_READ]:
            return attempt < 3

        # Don't retry parsing/validation errors
        return False

    def should_continue(self, error: IndexingError, session: IndexingSession) -> bool:
        """Determine if indexing should continue after an error."""
        # Continue for non-critical errors
        if error.severity != ErrorSeverity.CRITICAL:
            return True

        # For critical errors, only continue if we have few errors so far
        critical_count = sum(
            1 for e in session.errors
            if e.severity == ErrorSeverity.CRITICAL
        )
        return critical_count < 5

    def get_retry_delay(self, attempt: int) -> float:
        """Get delay (in seconds) before retrying."""
        # Exponential backoff: 0.1, 0.2, 0.4 seconds
        return 0.1 * (2 ** (attempt - 1))


class IndexIntegrityValidator:
    """Validates the integrity of the index after indexing."""

    def validate_manifest(self, manifest: Dict[str, dict]) -> List[IndexingError]:
        """Validate manifest file integrity."""
        errors = []

        if not isinstance(manifest, dict):
            errors.append(IndexingError(
                file_path="manifest.json",
                category=ErrorCategory.MANIFEST,
                message="Manifest is not a valid dictionary",
                severity=ErrorSeverity.CRITICAL,
                recoverable=False,
            ))
            return errors

        # Check that all files in manifest have required fields
        for file_path, file_data in manifest.get("files", {}).items():
            if not isinstance(file_data, dict):
                errors.append(IndexingError(
                    file_path=file_path,
                    category=ErrorCategory.MANIFEST,
                    message="File entry in manifest is not a dictionary",
                    severity=ErrorSeverity.WARNING,
                    recoverable=True,
                ))
                continue

            if "hash" not in file_data:
                errors.append(IndexingError(
                    file_path=file_path,
                    category=ErrorCategory.MANIFEST,
                    message="File entry missing 'hash' field",
                    severity=ErrorSeverity.WARNING,
                    recoverable=True,
                ))

            if "timestamp" not in file_data:
                errors.append(IndexingError(
                    file_path=file_path,
                    category=ErrorCategory.MANIFEST,
                    message="File entry missing 'timestamp' field",
                    severity=ErrorSeverity.INFO,
                    recoverable=True,
                ))

        return errors

    def check_file_consistency(
        self,
        file_path: str,
        stored_hash: str,
        current_hash: str,
    ) -> Optional[IndexingError]:
        """Check if a file's hash matches what was stored."""
        if stored_hash != current_hash:
            return IndexingError(
                file_path=file_path,
                category=ErrorCategory.INTEGRITY,
                message=f"File hash mismatch: stored={stored_hash}, current={current_hash}",
                severity=ErrorSeverity.WARNING,
                recoverable=True,
            )
        return None
