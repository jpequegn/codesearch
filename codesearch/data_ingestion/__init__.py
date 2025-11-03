"""Data ingestion pipeline for batch entity insertion into LanceDB."""

from codesearch.data_ingestion.pipeline import DataIngestionPipeline
from codesearch.data_ingestion.models import IngestionResult, IngestionError
from codesearch.data_ingestion.deduplication import DeduplicationCache
from codesearch.data_ingestion.validation import IngestionValidator
from codesearch.data_ingestion.audit import AuditTrail, AuditRecord

__all__ = [
    "DataIngestionPipeline",
    "IngestionResult",
    "IngestionError",
    "DeduplicationCache",
    "IngestionValidator",
    "AuditTrail",
    "AuditRecord",
]
