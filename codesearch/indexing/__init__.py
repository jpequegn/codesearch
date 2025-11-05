"""Indexing module for code scanning and processing."""

from codesearch.indexing.scanner import RepositoryScannerImpl
from codesearch.indexing.errors import (
    ErrorSeverity,
    ErrorCategory,
    IndexingError,
    IndexingSession,
    ErrorRecoveryStrategy,
    IndexIntegrityValidator,
)
from codesearch.indexing.recovery import (
    CheckpointManager,
    RecoveryState,
    InterruptionDetector,
)
from codesearch.indexing.error_handler import IndexingErrorHandler

__all__ = [
    "RepositoryScannerImpl",
    "ErrorSeverity",
    "ErrorCategory",
    "IndexingError",
    "IndexingSession",
    "ErrorRecoveryStrategy",
    "IndexIntegrityValidator",
    "CheckpointManager",
    "RecoveryState",
    "InterruptionDetector",
    "IndexingErrorHandler",
]
