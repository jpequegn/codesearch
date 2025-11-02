from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class IngestionError:
    """Track individual entity ingestion failures with detailed error information."""

    entity_id: str
    error_type: str  # "validation", "insertion", "relationship", "metadata"
    message: str     # Detailed error message
    recoverable: bool  # Can be retried?
    timestamp: datetime

    def __str__(self) -> str:
        """Human-readable error representation."""
        recoverable_str = "recoverable" if self.recoverable else "unrecoverable"
        return (f"IngestionError({self.entity_id}, {self.error_type}, "
                f"{recoverable_str}): {self.message}")


@dataclass
class IngestionResult:
    """Comprehensive result tracking for batch ingestion operations."""

    # Entity counters
    inserted_count: int = 0
    skipped_count: int = 0
    updated_count: int = 0
    failed_count: int = 0

    # Relationship ingestion metrics
    relationships_inserted: int = 0
    relationships_failed: int = 0

    # Metadata ingestion metrics
    metadata_inserted: int = 0
    metadata_failed: int = 0

    # Error tracking
    errors: List[IngestionError] = field(default_factory=list)

    # Audit information
    batch_id: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    duration_ms: float = 0.0

    def summary(self) -> str:
        """Human-readable summary of ingestion results."""
        return (f"Inserted {self.inserted_count}, skipped {self.skipped_count}, "
                f"updated {self.updated_count}, failed {self.failed_count}")

    def is_successful(self) -> bool:
        """True if at least some records were inserted or updated."""
        return self.inserted_count + self.updated_count > 0

    def add_error(self, error: IngestionError) -> None:
        """Add an error to the result and update failed counter."""
        self.errors.append(error)
        self.failed_count += 1
