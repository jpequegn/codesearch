import pytest
from unittest.mock import Mock, AsyncMock
from codesearch.data_ingestion.deduplication import DeduplicationCache
from codesearch.models import CodeEntity


@pytest.fixture
def mock_client():
    """Create mock LanceDB client."""
    return Mock()


@pytest.fixture
def mock_code_entity():
    """Create a sample CodeEntity for testing."""
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
        source_hash="abc123def456"
    )


def test_deduplication_cache_initialization(mock_client):
    """Test that DeduplicationCache initializes empty hash set."""
    cache = DeduplicationCache(mock_client)

    assert isinstance(cache.hashes, set)
    assert len(cache.hashes) == 0
    assert cache.client == mock_client


def test_is_duplicate_returns_false_for_new_hash(mock_client, mock_code_entity):
    """Test that new source_hash is not marked as duplicate."""
    cache = DeduplicationCache(mock_client)

    assert cache.is_duplicate(mock_code_entity) is False


def test_is_duplicate_returns_true_after_add(mock_client, mock_code_entity):
    """Test that hash is marked as duplicate after adding."""
    cache = DeduplicationCache(mock_client)

    cache.add(mock_code_entity)
    assert cache.is_duplicate(mock_code_entity) is True


def test_add_updates_hashes_set(mock_client, mock_code_entity):
    """Test that add() properly updates internal hashes set."""
    cache = DeduplicationCache(mock_client)
    assert len(cache.hashes) == 0

    cache.add(mock_code_entity)
    assert len(cache.hashes) == 1
    assert mock_code_entity.source_hash in cache.hashes


def test_detect_update_same_code_different_location(mock_client):
    """Test detecting update when code unchanged but location changed."""
    existing = CodeEntity(
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
        source_hash="abc123"
    )

    updated = CodeEntity(
        entity_id="repo2:file.py:Parser:parse",
        name="parse",
        code_text="def parse(self): return self.data",  # Same code
        code_vector=[0.1] * 768,
        language="python",
        entity_type="function",
        repository="repo2",  # Different repository
        file_path="file.py",
        start_line=10,
        end_line=12,
        visibility="public",
        source_hash="abc123"  # Same hash
    )

    cache = DeduplicationCache(mock_client)
    assert cache.detect_update(updated, existing) is True


def test_detect_update_different_code_same_location(mock_client):
    """Test that different code is NOT marked as update."""
    existing = CodeEntity(
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

    modified = CodeEntity(
        entity_id="repo:file.py:Parser:parse",
        name="parse",
        code_text="def parse(self): return None",  # Different code
        code_vector=[0.2] * 768,
        language="python",
        entity_type="function",
        repository="repo",
        file_path="file.py",
        start_line=10,
        end_line=13,
        visibility="public",
        source_hash="different456"  # Different hash
    )

    cache = DeduplicationCache(mock_client)
    assert cache.detect_update(modified, existing) is False
