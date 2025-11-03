import pytest
from unittest.mock import Mock, MagicMock
from codesearch.query.engine import QueryEngine
from codesearch.query.models import SearchResult
from codesearch.query.filters import LanguageFilter


@pytest.fixture
def mock_client():
    """Create mock LanceDB client."""
    client = Mock()

    # Mock code_entities table
    entities_table = MagicMock()
    client.get_table = Mock(return_value=entities_table)

    return client


def test_query_engine_initialization(mock_client):
    """Test QueryEngine initializes with client."""
    engine = QueryEngine(mock_client)

    assert engine.client == mock_client
    assert engine.code_entities_table is not None


def test_search_vector_returns_results(mock_client):
    """Test search_vector returns SearchResult list."""
    # Setup mock to return results
    mock_table = mock_client.get_table.return_value
    mock_search = MagicMock()
    mock_search.limit.return_value.to_list.return_value = [
        {
            'entity_id': 'repo:file.py:Parser:parse',
            'name': 'parse',
            'code_text': 'def parse(self): pass',
            '_distance': 0.1,  # LanceDB distance (lower is better)
            'language': 'python',
            'file_path': 'file.py',
            'repository': 'repo',
            'entity_type': 'function',
            'start_line': 10,
            'end_line': 12
        }
    ]
    mock_table.search.return_value = mock_search

    engine = QueryEngine(mock_client)
    results = engine.search_vector([0.1] * 768, limit=10)

    assert len(results) > 0
    assert isinstance(results[0], SearchResult)


def test_search_vector_with_limit(mock_client):
    """Test search_vector respects limit parameter."""
    mock_table = mock_client.get_table.return_value
    mock_search = MagicMock()
    mock_search.limit.return_value.to_list.return_value = []
    mock_table.search.return_value = mock_search

    engine = QueryEngine(mock_client)
    results = engine.search_vector([0.1] * 768, limit=20)

    # Verify limit was passed (may be multiplied for filtering)
    mock_search.limit.assert_called()


def test_search_vector_with_filters(mock_client):
    """Test search_vector applies metadata filters."""
    # Setup mock to return mixed results
    mock_table = mock_client.get_table.return_value
    mock_search = MagicMock()
    mock_search.limit.return_value.to_list.return_value = [
        {
            'entity_id': 'repo:file.py:Parser:parse',
            'name': 'parse',
            'code_text': 'def parse(self): pass',
            '_distance': 0.1,
            'language': 'python',
            'file_path': 'file.py',
            'repository': 'repo',
            'entity_type': 'function',
            'start_line': 10,
            'end_line': 12
        },
        {
            'entity_id': 'repo:file.js:Parser:parse',
            'name': 'parse',
            'code_text': 'function parse() {}',
            '_distance': 0.12,
            'language': 'javascript',
            'file_path': 'file.js',
            'repository': 'repo',
            'entity_type': 'function',
            'start_line': 5,
            'end_line': 7
        }
    ]
    mock_table.search.return_value = mock_search

    engine = QueryEngine(mock_client)
    results = engine.search_vector(
        [0.1] * 768,
        filters=[LanguageFilter(['python'])],
        limit=10
    )

    # Should filter to only python results
    assert len(results) == 1
    assert results[0].language == 'python'
