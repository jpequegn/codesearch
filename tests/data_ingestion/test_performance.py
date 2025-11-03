import pytest
import time
from unittest.mock import Mock
from codesearch.data_ingestion.pipeline import DataIngestionPipeline
from codesearch.models import CodeEntity


@pytest.fixture
def mock_client():
    """Create mock LanceDB client."""
    client = Mock()
    client.get_table = Mock(return_value=Mock())
    return client


def create_test_entities(count: int) -> list:
    """Create test entities for performance testing."""
    entities = []
    for i in range(count):
        entity = CodeEntity(
            entity_id=f"repo:file{i}.py:Module{i}:func{i}",
            name=f"func{i}",
            code_text=f"def func{i}(): return {i}",
            code_vector=[float(i % 768) / 768.0] * 768,
            language="python",
            entity_type="function",
            repository=f"repo{i % 5}",
            file_path=f"file{i}.py",
            start_line=i * 10,
            end_line=i * 10 + 5,
            visibility="public",
            source_hash=f"hash{i}"
        )
        entities.append(entity)
    return entities


def test_throughput_small_batch(mock_client):
    """Test throughput with 100-entity batch."""
    pipeline = DataIngestionPipeline(mock_client)
    entities = create_test_entities(100)

    result = pipeline.ingest_batch(entities)

    assert result.inserted_count == 100
    assert result.duration_ms > 0

    # Calculate throughput: entities per second
    throughput = (result.inserted_count / result.duration_ms) * 1000

    # Should achieve >1000 entities/sec
    assert throughput > 1000, f"Throughput {throughput:.0f} entities/sec is too low"


def test_throughput_large_batch(mock_client):
    """Test throughput with 1000-entity batch."""
    pipeline = DataIngestionPipeline(mock_client, batch_size=1000)
    entities = create_test_entities(1000)

    result = pipeline.ingest_batch(entities)

    assert result.inserted_count == 1000

    throughput = (result.inserted_count / result.duration_ms) * 1000
    assert throughput > 1000, f"Throughput {throughput:.0f} entities/sec is too low"


def test_dedup_cache_performance(mock_client):
    """Test deduplication cache lookup performance."""
    pipeline = DataIngestionPipeline(mock_client)
    entities = create_test_entities(1000)

    # Add all entities to cache
    start = time.time()
    for entity in entities:
        pipeline.dedup_cache.add(entity)
    add_time = (time.time() - start) * 1000000  # Convert to microseconds

    # Lookup performance should be O(1)
    start = time.time()
    for entity in entities:
        pipeline.dedup_cache.is_duplicate(entity)
    lookup_time = (time.time() - start) * 1000000  # Convert to microseconds

    avg_lookup_us = lookup_time / len(entities)

    # Average lookup should be <1ms
    assert avg_lookup_us < 1000, f"Average lookup {avg_lookup_us:.2f}µs is too slow"


def test_validation_performance(mock_client):
    """Test validation throughput."""
    pipeline = DataIngestionPipeline(mock_client)
    entities = create_test_entities(100)

    start = time.time()
    for entity in entities:
        pipeline.validator.validate_entity(entity)
    duration = (time.time() - start) * 1000

    throughput = (len(entities) / duration) * 1000

    # Should validate >1000 entities/sec
    assert throughput > 1000, f"Validation throughput {throughput:.0f} entities/sec is too low"
