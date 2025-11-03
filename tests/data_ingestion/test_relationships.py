import pytest
from unittest.mock import Mock
from codesearch.data_ingestion.pipeline import DataIngestionPipeline
from codesearch.models import CodeRelationship


@pytest.fixture
def mock_client():
    """Create mock LanceDB client."""
    client = Mock()
    client.get_table = Mock(return_value=Mock())
    return client


def test_ingest_relationships_returns_result(mock_client):
    """Test that ingest_relationships returns IngestionResult."""
    pipeline = DataIngestionPipeline(mock_client)

    relationships = [
        CodeRelationship(
            caller_id="repo:file.py:Parser:parse",
            callee_id="repo:file.py:Utils:format",
            relationship_type="calls"
        )
    ]

    result = pipeline.ingest_relationships(relationships)

    assert result is not None
    assert result.relationships_inserted == 1
    assert result.relationships_failed == 0


def test_ingest_relationships_validation_failure(mock_client):
    """Test that invalid relationships are tracked."""
    pipeline = DataIngestionPipeline(mock_client)

    # Self-referential relationship (invalid)
    relationships = [
        CodeRelationship(
            caller_id="repo:file.py:Parser:parse",
            callee_id="repo:file.py:Parser:parse",  # Same as caller
            relationship_type="calls"
        )
    ]

    result = pipeline.ingest_relationships(relationships)

    assert result.relationships_inserted == 0
    assert result.relationships_failed == 1
    assert len(result.errors) > 0


def test_ingest_relationships_mixed_valid_invalid(mock_client):
    """Test batch with valid and invalid relationships."""
    pipeline = DataIngestionPipeline(mock_client)

    relationships = [
        CodeRelationship(
            caller_id="repo:file.py:Parser:parse",
            callee_id="repo:file.py:Utils:format",
            relationship_type="calls"
        ),
        CodeRelationship(
            caller_id="repo:file.py:Bad:func",
            callee_id="repo:file.py:Bad:func",  # Invalid self-ref
            relationship_type="calls"
        )
    ]

    result = pipeline.ingest_relationships(relationships)

    assert result.relationships_inserted == 1
    assert result.relationships_failed == 1
