import pytest
from unittest.mock import Mock, MagicMock
from codesearch.query.engine import QueryEngine
from codesearch.query.filters import LanguageFilter, FilePathFilter
from codesearch.query.models import SearchResult


@pytest.fixture
def mock_client_with_results():
    """Create mock client that returns realistic search results."""
    client = Mock()
    mock_table = MagicMock()

    # Sample search results (what LanceDB would return)
    search_results = [
        {
            'entity_id': 'repo:src/parser.py:Parser:parse',
            'name': 'parse',
            'code_text': 'def parse(self, text): return self._parse(text)',
            '_distance': 0.05,
            'language': 'python',
            'file_path': 'src/parser.py',
            'repository': 'repo',
            'entity_type': 'function',
            'start_line': 10,
            'end_line': 12
        },
        {
            'entity_id': 'repo:src/utils.py:parse_json',
            'name': 'parse_json',
            'code_text': 'def parse_json(s): return json.loads(s)',
            '_distance': 0.08,
            'language': 'python',
            'file_path': 'src/utils.py',
            'repository': 'repo',
            'entity_type': 'function',
            'start_line': 20,
            'end_line': 21
        },
        {
            'entity_id': 'repo:lib/parser.js:parse',
            'name': 'parse',
            'code_text': 'function parse(text) { return parseText(text); }',
            '_distance': 0.12,
            'language': 'javascript',
            'file_path': 'lib/parser.js',
            'repository': 'repo',
            'entity_type': 'function',
            'start_line': 5,
            'end_line': 7
        }
    ]

    mock_search = MagicMock()
    mock_search.limit.return_value.to_list.return_value = search_results
    mock_table.search.return_value = mock_search
    client.get_table.return_value = mock_table

    return client


def test_full_search_with_filters(mock_client_with_results):
    """Test complete search flow with filtering."""
    engine = QueryEngine(mock_client_with_results)

    results = engine.search_vector(
        [0.1] * 768,
        filters=[LanguageFilter(['python'])],
        limit=10
    )

    # Should return 2 python results (parser and utils)
    assert len(results) == 2
    assert all(r.language == 'python' for r in results)
    assert results[0].similarity_score > results[1].similarity_score


def test_search_with_file_path_filter(mock_client_with_results):
    """Test filtering by file path."""
    engine = QueryEngine(mock_client_with_results)

    results = engine.search_vector(
        [0.1] * 768,
        filters=[FilePathFilter(['src/'])],
        limit=10
    )

    # Should return 2 results from src/
    assert len(results) == 2
    assert all('src/' in r.file_path for r in results)


def test_search_with_multiple_filters(mock_client_with_results):
    """Test filtering with multiple filter types (AND logic)."""
    engine = QueryEngine(mock_client_with_results)

    results = engine.search_vector(
        [0.1] * 768,
        filters=[
            LanguageFilter(['python', 'javascript']),
            FilePathFilter(['src/'])
        ],
        limit=10
    )

    # Should return only Python results from src/ (python OR javascript) AND (src/)
    assert len(results) == 2
    assert all(r.language == 'python' for r in results)
    assert all('src/' in r.file_path for r in results)


def test_pagination(mock_client_with_results):
    """Test pagination with offset and limit."""
    engine = QueryEngine(mock_client_with_results)

    # First page
    page1 = engine.search_vector([0.1] * 768, limit=1, offset=0)
    assert len(page1) == 1
    assert page1[0].name == 'parse'

    # Second page
    page2 = engine.search_vector([0.1] * 768, limit=1, offset=1)
    assert len(page2) == 1
    assert page2[0].name == 'parse_json'


def test_search_result_sorting_by_similarity(mock_client_with_results):
    """Test results are sorted by similarity (best first)."""
    engine = QueryEngine(mock_client_with_results)

    results = engine.search_vector([0.1] * 768, limit=10)

    # Results should be sorted by similarity descending
    for i in range(len(results) - 1):
        assert results[i].similarity_score >= results[i + 1].similarity_score
