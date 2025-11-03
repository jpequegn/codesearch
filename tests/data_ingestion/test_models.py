from datetime import datetime
from codesearch.data_ingestion.models import IngestionError, IngestionResult


def test_ingestion_error_creation():
    """Test IngestionError dataclass creation and fields."""
    error = IngestionError(
        entity_id="repo:file.py:Parser:parse",
        error_type="validation",
        message="code_vector has wrong dimensions: expected 768, got 512",
        recoverable=True,
        timestamp=datetime.utcnow()
    )

    assert error.entity_id == "repo:file.py:Parser:parse"
    assert error.error_type == "validation"
    assert error.message == "code_vector has wrong dimensions: expected 768, got 512"
    assert error.recoverable is True
    assert isinstance(error.timestamp, datetime)


def test_ingestion_error_string_representation():
    """Test IngestionError string representation for logging."""
    error = IngestionError(
        entity_id="repo:file.py:Utils:format",
        error_type="insertion",
        message="Database constraint violation",
        recoverable=False,
        timestamp=datetime.utcnow()
    )

    error_str = str(error)
    assert "repo:file.py:Utils:format" in error_str
    assert "insertion" in error_str


def test_ingestion_result_creation():
    """Test IngestionResult initialization with default values."""
    result = IngestionResult(batch_id="batch-123")

    assert result.inserted_count == 0
    assert result.skipped_count == 0
    assert result.updated_count == 0
    assert result.failed_count == 0
    assert result.batch_id == "batch-123"
    assert result.errors == []


def test_ingestion_result_summary():
    """Test summary message generation."""
    result = IngestionResult(
        inserted_count=100,
        skipped_count=5,
        updated_count=2,
        failed_count=1
    )

    summary = result.summary()
    assert "Inserted 100" in summary
    assert "skipped 5" in summary
    assert "updated 2" in summary
    assert "failed 1" in summary


def test_ingestion_result_is_successful():
    """Test success determination logic."""
    result_empty = IngestionResult()
    assert result_empty.is_successful() is False

    result_inserted = IngestionResult(inserted_count=5)
    assert result_inserted.is_successful() is True

    result_updated = IngestionResult(updated_count=3)
    assert result_updated.is_successful() is True

    result_only_failed = IngestionResult(failed_count=10)
    assert result_only_failed.is_successful() is False


def test_ingestion_result_add_error():
    """Test error tracking and counter updates."""
    result = IngestionResult()

    error = IngestionError(
        entity_id="test-entity",
        error_type="validation",
        message="Test error",
        recoverable=True,
        timestamp=datetime.utcnow()
    )

    result.add_error(error)

    assert len(result.errors) == 1
    assert result.failed_count == 1
    assert result.errors[0] == error
