import pytest
from unittest.mock import Mock, MagicMock
from datetime import datetime
from codesearch.data_ingestion.pipeline import DataIngestionPipeline
from codesearch.data_ingestion.models import IngestionResult
from codesearch.models import CodeEntity


@pytest.fixture
def mock_client():
    """Create mock LanceDB client."""
    client = Mock()
    client.get_table = Mock(return_value=Mock())
    return client


@pytest.fixture
def valid_entity():
    """Create a valid CodeEntity."""
    return CodeEntity(
        entity_id="repo:file.py:Parser:parse",
        name="parse",
        code_text="def parse(self): return self.data",
        code_vector=[0.1] * 768,
        language="python",
        entity_type="function",
        repository="repo",
        file_path="file.py",
        start_line=10,
        end_line=12,
        visibility="public",
        source_hash="abc123"
    )


def test_pipeline_initialization(mock_client):
    """Test that DataIngestionPipeline initializes properly."""
    pipeline = DataIngestionPipeline(mock_client, batch_size=1000)

    assert pipeline.client == mock_client
    assert pipeline.batch_size == 1000
    assert pipeline.dedup_cache is not None
    assert pipeline.validator is not None


def test_pipeline_default_batch_size(mock_client):
    """Test that pipeline uses default batch size of 1000."""
    pipeline = DataIngestionPipeline(mock_client)
    assert pipeline.batch_size == 1000


def test_ingest_batch_returns_result(mock_client, valid_entity):
    """Test that ingest_batch returns IngestionResult."""
    pipeline = DataIngestionPipeline(mock_client)

    result = pipeline.ingest_batch([valid_entity])

    assert isinstance(result, IngestionResult)
    assert result.batch_id != ""
    assert isinstance(result.timestamp, datetime)


def test_ingest_batch_single_new_entity(mock_client, valid_entity):
    """Test ingesting a single new entity."""
    pipeline = DataIngestionPipeline(mock_client)

    result = pipeline.ingest_batch([valid_entity])

    assert result.inserted_count == 1
    assert result.skipped_count == 0
    assert result.failed_count == 0


def test_ingest_batch_skips_duplicates(mock_client, valid_entity):
    """Test that duplicate entities are skipped."""
    pipeline = DataIngestionPipeline(mock_client)

    # Add entity to cache to make it a duplicate
    pipeline.dedup_cache.add(valid_entity)

    result = pipeline.ingest_batch([valid_entity])

    assert result.inserted_count == 0
    assert result.skipped_count == 1
    assert result.failed_count == 0


def test_ingest_batch_validation_failure(mock_client):
    """Test that validation failures are tracked."""
    pipeline = DataIngestionPipeline(mock_client)

    invalid_entity = CodeEntity(
        entity_id="",  # Empty entity_id will fail validation
        name="parse",
        code_text="def parse(self): pass",
        code_vector=[0.1] * 768,
        language="python",
        entity_type="function",
        repository="repo",
        file_path="file.py",
        start_line=10,
        end_line=12,
        visibility="public",
        source_hash="abc123"
    )

    result = pipeline.ingest_batch([invalid_entity])

    assert result.inserted_count == 0
    assert result.failed_count == 1
    assert len(result.errors) > 0


def test_ingest_batch_mixed_entities(mock_client, valid_entity):
    """Test batch with mix of new, duplicate, and invalid entities."""
    pipeline = DataIngestionPipeline(mock_client)

    # Add one entity to cache to make it a duplicate
    pipeline.dedup_cache.add(valid_entity)

    invalid_entity = CodeEntity(
        entity_id="",
        name="test",
        code_text="test",
        code_vector=[0.1] * 768,
        language="python",
        entity_type="function",
        repository="repo",
        file_path="test.py",
        start_line=1,
        end_line=2,
        visibility="public",
        source_hash="different123"
    )

    new_entity = CodeEntity(
        entity_id="new:test",
        name="new_func",
        code_text="def new_func(): pass",
        code_vector=[0.2] * 768,
        language="python",
        entity_type="function",
        repository="repo",
        file_path="new.py",
        start_line=1,
        end_line=2,
        visibility="public",
        source_hash="new456"
    )

    result = pipeline.ingest_batch([valid_entity, invalid_entity, new_entity])

    # Should have 1 inserted (new_entity), 1 skipped (duplicate valid_entity), 1 failed (invalid)
    assert result.inserted_count == 1
    assert result.skipped_count == 1
    assert result.failed_count == 1


def test_ingest_batch_result_summary(mock_client, valid_entity):
    """Test IngestionResult summary generation."""
    pipeline = DataIngestionPipeline(mock_client)

    result = pipeline.ingest_batch([valid_entity])

    summary = result.summary()
    assert "Inserted" in summary
    assert str(result.inserted_count) in summary
