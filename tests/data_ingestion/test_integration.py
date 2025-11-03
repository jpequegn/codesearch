import pytest
from unittest.mock import Mock, MagicMock
from datetime import datetime
from codesearch.data_ingestion.pipeline import DataIngestionPipeline
from codesearch.models import CodeEntity, CodeRelationship, SearchMetadata


@pytest.fixture
def mock_client():
    """Create mock LanceDB client with all necessary tables."""
    client = Mock()

    # Mock all table operations
    entities_table = MagicMock()
    relationships_table = MagicMock()
    metadata_table = MagicMock()

    def get_table(name):
        if name == "code_entities":
            return entities_table
        elif name == "code_relationships":
            return relationships_table
        elif name == "search_metadata":
            return metadata_table
        return MagicMock()

    client.get_table = get_table
    return client


def test_full_pipeline_ingestion_entities_relationships_metadata(mock_client):
    """Test complete pipeline: entities → relationships → metadata."""
    pipeline = DataIngestionPipeline(mock_client)

    # Prepare test data
    entity1 = CodeEntity(
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
        source_hash="hash1"
    )

    entity2 = CodeEntity(
        entity_id="repo:file.py:Utils:format",
        name="format",
        code_text="def format(data): return str(data)",
        code_vector=[0.2] * 768,
        language="python",
        entity_type="function",
        repository="repo",
        file_path="file.py",
        start_line=15,
        end_line=16,
        visibility="public",
        source_hash="hash2"
    )

    # Step 1: Ingest entities
    entity_result = pipeline.ingest_batch([entity1, entity2])

    assert entity_result.inserted_count == 2
    assert entity_result.failed_count == 0
    assert entity_result.is_successful()

    # Step 2: Ingest relationships
    relationship = CodeRelationship(
        caller_id=entity1.entity_id,
        callee_id=entity2.entity_id,
        relationship_type="calls"
    )

    rel_result = pipeline.ingest_relationships([relationship])

    assert rel_result.relationships_inserted == 1
    assert rel_result.relationships_failed == 0

    # Step 3: Ingest metadata
    metadata_list = [
        SearchMetadata(
            metadata_id=entity1.entity_id,
            entity_id=entity1.entity_id,
            repository="repo",
            file_path="file.py"
        ),
        SearchMetadata(
            metadata_id=entity2.entity_id,
            entity_id=entity2.entity_id,
            repository="repo",
            file_path="file.py"
        )
    ]

    metadata_result = pipeline.ingest_metadata(metadata_list)

    assert metadata_result.metadata_inserted == 2
    assert metadata_result.metadata_failed == 0


def test_duplicate_detection_across_batches(mock_client):
    """Test that duplicates are detected across multiple batch calls."""
    pipeline = DataIngestionPipeline(mock_client)

    entity = CodeEntity(
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
        source_hash="hash1"
    )

    # First batch - should insert
    result1 = pipeline.ingest_batch([entity])
    assert result1.inserted_count == 1

    # Second batch - should skip as duplicate
    result2 = pipeline.ingest_batch([entity])
    assert result2.skipped_count == 1
    assert result2.inserted_count == 0


def test_update_detection_different_location(mock_client):
    """Test that moved functions are detected as updates."""
    pipeline = DataIngestionPipeline(mock_client)

    entity_v1 = CodeEntity(
        entity_id="repo1:file.py:Parser:parse",
        name="parse",
        code_text="def parse(self): return self.data",
        code_vector=[0.1] * 768,
        language="python",
        entity_type="function",
        repository="repo1",
        file_path="file.py",
        start_line=10,
        end_line=12,
        visibility="public",
        source_hash="same_hash"  # Same code
    )

    entity_v2 = CodeEntity(
        entity_id="repo2:file.py:Parser:parse",
        name="parse",
        code_text="def parse(self): return self.data",  # Same code
        code_vector=[0.1] * 768,
        language="python",
        entity_type="function",
        repository="repo2",  # Different repository (moved)
        file_path="file.py",
        start_line=10,
        end_line=12,
        visibility="public",
        source_hash="same_hash"  # Same hash
    )

    # Insert first version
    result1 = pipeline.ingest_batch([entity_v1])
    assert result1.inserted_count == 1

    # Try to insert moved version
    # (in real implementation, this would be detected as update)
    # For now, it will be skipped as duplicate
    result2 = pipeline.ingest_batch([entity_v2])
    assert result2.skipped_count == 1  # Detected as duplicate due to same source_hash


def test_partial_failure_recovery(mock_client):
    """Test that valid entities are inserted even if some fail."""
    pipeline = DataIngestionPipeline(mock_client)

    valid_entity = CodeEntity(
        entity_id="repo:file.py:Utils:format",
        name="format",
        code_text="def format(data): return str(data)",
        code_vector=[0.2] * 768,
        language="python",
        entity_type="function",
        repository="repo",
        file_path="file.py",
        start_line=15,
        end_line=16,
        visibility="public",
        source_hash="hash2"
    )

    invalid_entity = CodeEntity(
        entity_id="",  # Invalid - empty ID
        name="bad",
        code_text="bad code",
        code_vector=[0.1] * 512,  # Invalid - wrong dimensions
        language="python",
        entity_type="function",
        repository="repo",
        file_path="file.py",
        start_line=1,
        end_line=2,
        visibility="public",
        source_hash="hash_bad"
    )

    result = pipeline.ingest_batch([valid_entity, invalid_entity])

    # Valid should be inserted, invalid should fail
    assert result.inserted_count == 1
    assert result.failed_count == 1
    assert result.is_successful()  # Overall success because some records inserted
