import pytest
from unittest.mock import Mock
from codesearch.data_ingestion.pipeline import DataIngestionPipeline
from codesearch.models import SearchMetadata


@pytest.fixture
def mock_client():
    """Create mock LanceDB client."""
    client = Mock()
    client.get_table = Mock(return_value=Mock())
    return client


def test_ingest_metadata_returns_result(mock_client):
    """Test that ingest_metadata returns IngestionResult."""
    pipeline = DataIngestionPipeline(mock_client)

    entity_id = "repo:file.py:Parser:parse"
    metadata = [
        SearchMetadata(
            metadata_id=entity_id,
            entity_id=entity_id,
            repository="repo",
            file_path="file.py"
        )
    ]

    result = pipeline.ingest_metadata(metadata)

    assert result is not None
    assert result.metadata_inserted == 1
    assert result.metadata_failed == 0


def test_ingest_metadata_validation_failure(mock_client):
    """Test that invalid metadata is tracked."""
    pipeline = DataIngestionPipeline(mock_client)

    entity_id = "repo:file.py:Parser:parse"
    # metadata_id doesn't match entity_id
    metadata = [
        SearchMetadata(
            metadata_id="wrong_id",
            entity_id=entity_id,
            repository="repo",
            file_path="file.py"
        )
    ]

    result = pipeline.ingest_metadata(metadata)

    assert result.metadata_inserted == 0
    assert result.metadata_failed == 1
    assert len(result.errors) > 0


def test_ingest_metadata_mixed_valid_invalid(mock_client):
    """Test batch with valid and invalid metadata."""
    pipeline = DataIngestionPipeline(mock_client)

    valid_id = "repo:file.py:Parser:parse"
    metadata = [
        SearchMetadata(
            metadata_id=valid_id,
            entity_id=valid_id,
            repository="repo",
            file_path="file.py"
        ),
        SearchMetadata(
            metadata_id="wrong_id",  # Invalid
            entity_id="repo:file.py:Utils:format",
            repository="repo",
            file_path="file.py"
        )
    ]

    result = pipeline.ingest_metadata(metadata)

    assert result.metadata_inserted == 1
    assert result.metadata_failed == 1
