from unittest.mock import MagicMock, Mock, patch

import pytest

from codesearch.query.engine import QueryEngine
from codesearch.query.exceptions import QueryError
from codesearch.query.filters import LanguageFilter
from codesearch.query.models import SearchResult


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


# Tests for search_text (Issue #54)


def test_search_text_embeds_and_searches(mock_client):
    """Test search_text embeds query and delegates to search_vector."""
    # Setup mock table to return results
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
        }
    ]
    mock_table.search.return_value = mock_search

    # Mock the embedder
    mock_embedder = Mock()
    mock_embedder.embed_code.return_value = [0.1] * 768

    engine = QueryEngine(mock_client, embedder=mock_embedder)
    results = engine.search_text("parse function", limit=10)

    # Verify embedder was called with query text
    mock_embedder.embed_code.assert_called_once_with("parse function")

    # Verify search was performed
    mock_table.search.assert_called_once()
    assert len(results) == 1
    assert results[0].name == 'parse'


def test_search_text_with_filters(mock_client):
    """Test search_text passes filters to search_vector."""
    # Setup mock table
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
        }
    ]
    mock_table.search.return_value = mock_search

    mock_embedder = Mock()
    mock_embedder.embed_code.return_value = [0.1] * 768

    engine = QueryEngine(mock_client, embedder=mock_embedder)
    results = engine.search_text(
        "parse function",
        filters=[LanguageFilter(['python'])],
        limit=5
    )

    assert len(results) == 1
    assert results[0].language == 'python'


def test_search_text_lazy_loads_embedder(mock_client):
    """Test that embedder is lazy-loaded if not provided."""
    # Setup mock table
    mock_table = mock_client.get_table.return_value
    mock_search = MagicMock()
    mock_search.limit.return_value.to_list.return_value = []
    mock_table.search.return_value = mock_search

    engine = QueryEngine(mock_client)

    # Embedder should not be loaded yet
    assert engine._embedder is None

    # Patch EmbeddingGenerator to avoid loading actual model
    with patch('codesearch.query.engine.EmbeddingGenerator') as mock_gen_class:
        mock_gen = Mock()
        mock_gen.embed_code.return_value = [0.1] * 768
        mock_gen_class.return_value = mock_gen

        # Access embedder property should trigger lazy loading
        _ = engine.embedder

        # Should have created embedder
        mock_gen_class.assert_called_once()


def test_search_text_handles_embedding_error(mock_client):
    """Test search_text wraps embedding errors in QueryError."""
    mock_embedder = Mock()
    mock_embedder.embed_code.side_effect = RuntimeError("Model failed to load")

    engine = QueryEngine(mock_client, embedder=mock_embedder)

    with pytest.raises(QueryError) as exc_info:
        engine.search_text("test query")

    assert "Text search failed" in str(exc_info.value)


def test_query_engine_accepts_embedder_in_constructor(mock_client):
    """Test QueryEngine can accept pre-configured embedder."""
    mock_embedder = Mock()

    engine = QueryEngine(mock_client, embedder=mock_embedder)

    assert engine._embedder is mock_embedder
    assert engine.embedder is mock_embedder
