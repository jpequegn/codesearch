"""Error handling for indexing operations."""

import logging
from typing import Callable, Optional, TypeVar, Dict, Any

from codesearch.indexing.errors import (
    IndexingError,
    ErrorCategory,
    ErrorSeverity,
    IndexingSession,
    ErrorRecoveryStrategy,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


class IndexingErrorHandler:
    """Handles errors during indexing with recovery and reporting."""

    def __init__(self, session: IndexingSession, recovery_strategy: Optional[ErrorRecoveryStrategy] = None):
        """
        Initialize error handler.

        Args:
            session: The indexing session to track errors in
            recovery_strategy: Strategy for error recovery
        """
        self.session = session
        self.recovery_strategy = recovery_strategy or ErrorRecoveryStrategy()
        self.retry_counts: Dict[str, int] = {}

    def handle_file_read_error(
        self,
        file_path: str,
        exception: Exception,
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Handle a file read error.

        Args:
            file_path: Path to the file
            exception: The exception that was raised
            context: Additional context about the error

        Returns:
            True if should continue indexing, False if should stop
        """
        error = IndexingError(
            file_path=file_path,
            category=ErrorCategory.FILE_READ,
            message=f"Failed to read file: {str(exception)}",
            severity=ErrorSeverity.WARNING,
            exception=exception,
            recoverable=True,
            context=context or {},
        )

        self.session.add_error(error)
        self.session.mark_file_failed(file_path)

        # Can usually continue after file read errors
        return self.recovery_strategy.should_continue(error, self.session)

    def handle_parsing_error(
        self,
        file_path: str,
        exception: Exception,
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Handle a parsing error.

        Args:
            file_path: Path to the file
            exception: The exception that was raised
            context: Additional context about the error

        Returns:
            True if should continue indexing, False if should stop
        """
        error = IndexingError(
            file_path=file_path,
            category=ErrorCategory.PARSING,
            message=f"Failed to parse file: {str(exception)}",
            severity=ErrorSeverity.WARNING,
            exception=exception,
            recoverable=True,
            context=context or {},
        )

        self.session.add_error(error)
        self.session.mark_file_failed(file_path)

        # Continue after parsing errors (file is malformed but can skip it)
        return self.recovery_strategy.should_continue(error, self.session)

    def handle_validation_error(
        self,
        file_path: str,
        exception: Exception,
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Handle a validation error.

        Args:
            file_path: Path to the file
            exception: The exception that was raised
            context: Additional context about the error

        Returns:
            True if should continue indexing, False if should stop
        """
        error = IndexingError(
            file_path=file_path,
            category=ErrorCategory.VALIDATION,
            message=f"Validation failed: {str(exception)}",
            severity=ErrorSeverity.INFO,
            exception=exception,
            recoverable=True,
            context=context or {},
        )

        self.session.add_error(error)
        self.session.mark_file_failed(file_path)

        # Continue after validation errors
        return self.recovery_strategy.should_continue(error, self.session)

    def handle_database_error(
        self,
        file_path: str,
        exception: Exception,
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Handle a database error.

        Args:
            file_path: Path to the file
            exception: The exception that was raised
            context: Additional context about the error

        Returns:
            True if should continue indexing, False if should stop
        """
        error = IndexingError(
            file_path=file_path,
            category=ErrorCategory.DATABASE,
            message=f"Database operation failed: {str(exception)}",
            severity=ErrorSeverity.ERROR,
            exception=exception,
            recoverable=True,
            context=context or {},
        )

        self.session.add_error(error)
        self.session.mark_file_failed(file_path)

        # Database errors are more serious
        return self.recovery_strategy.should_continue(error, self.session)

    def should_retry(self, file_path: str) -> bool:
        """
        Check if a file should be retried.

        Args:
            file_path: Path to the file

        Returns:
            True if should retry, False otherwise
        """
        attempt = self.retry_counts.get(file_path, 1)
        self.retry_counts[file_path] = attempt + 1

        # Get the error for this file
        for error in self.session.errors:
            if error.file_path == file_path:
                return self.recovery_strategy.should_retry(error, attempt)

        return False

    def safe_execute(
        self,
        func: Callable[..., T],
        file_path: str,
        *args,
        **kwargs
    ) -> Optional[T]:
        """
        Safely execute a function with error handling.

        Args:
            func: Function to execute
            file_path: Path being processed (for context)
            *args: Arguments to pass to function
            **kwargs: Keyword arguments to pass to function

        Returns:
            Result from function or None if error occurred
        """
        try:
            return func(*args, **kwargs)
        except (IOError, OSError) as e:
            self.handle_file_read_error(file_path, e)
            return None
        except SyntaxError as e:
            self.handle_parsing_error(file_path, e)
            return None
        except ValueError as e:
            self.handle_validation_error(file_path, e)
            return None
        except Exception as e:
            # For unknown errors, log as error
            logger.error(f"Unexpected error processing {file_path}: {e}", exc_info=True)
            self.session.add_error(IndexingError(
                file_path=file_path,
                category=ErrorCategory.EXTRACTION,
                message=f"Unexpected error: {str(e)}",
                severity=ErrorSeverity.ERROR,
                exception=e,
                recoverable=False,
            ))
            self.session.mark_file_failed(file_path)
            return None

    def report_session_errors(self) -> Dict[str, Any]:
        """
        Generate a report of all errors in the session.

        Returns:
            Dictionary with error summary and details
        """
        summary = self.session.get_error_summary()

        # Add error details
        summary["errors"] = [
            {
                "file_path": error.file_path,
                "category": error.category.value,
                "message": error.message,
                "severity": error.severity.value,
                "timestamp": error.timestamp.isoformat(),
                "recoverable": error.recoverable,
            }
            for error in self.session.errors
        ]

        return summary
