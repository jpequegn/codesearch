import time
import pytest
from unittest.mock import Mock, MagicMock
from codesearch.query.engine import QueryEngine


def create_large_result_set(size: int) -> list:
    """Create a large mock result set for performance testing."""
    results = []
    for i in range(size):
        results.append({
            'entity_id': f'repo:file{i}.py:Func:func{i}',
            'name': f'func{i}',
            'code_text': f'def func{i}(): pass',
            '_distance': 0.1 + (i * 0.001),  # Vary distances
            'language': 'python' if i % 2 == 0 else 'javascript',
            'file_path': f'src/file{i}.py',
            'repository': 'repo',
            'entity_type': 'function',
            'start_line': i * 10,
            'end_line': i * 10 + 2
        })
    return results


@pytest.fixture
def mock_client_perf(request):
    """Create mock client for performance testing."""
    result_count = getattr(request, 'param', 100)

    client = Mock()
    mock_table = MagicMock()

    results = create_large_result_set(result_count)

    mock_search = MagicMock()
    mock_search.limit.return_value.to_list.return_value = results
    mock_table.search.return_value = mock_search
    client.get_table.return_value = mock_table

    return client


def test_search_latency_small_results(mock_client_perf):
    """Test search latency with 100 results."""
    client = mock_client_perf
    engine = QueryEngine(client)

    start = time.time()
    results = engine.search_vector([0.1] * 768, limit=10)
    duration_ms = (time.time() - start) * 1000

    assert len(results) == 10
    # Should complete well under 500ms (target: 100-500ms)
    assert duration_ms < 500, f"Search took {duration_ms:.2f}ms, expected <500ms"


@pytest.mark.parametrize('mock_client_perf', [1000], indirect=True)
def test_search_latency_large_results(mock_client_perf):
    """Test search latency with 1000 results."""
    client = mock_client_perf
    engine = QueryEngine(client)

    start = time.time()
    results = engine.search_vector([0.1] * 768, limit=10)
    duration_ms = (time.time() - start) * 1000

    assert len(results) == 10
    # Even with large result set, should be under 500ms
    assert duration_ms < 500, f"Search took {duration_ms:.2f}ms, expected <500ms"


def test_filter_performance(mock_client_perf):
    """Test filter application performance."""
    from codesearch.query.filters import LanguageFilter

    client = mock_client_perf
    engine = QueryEngine(client)

    start = time.time()
    results = engine.search_vector(
        [0.1] * 768,
        filters=[LanguageFilter(['python'])],
        limit=10
    )
    duration_ms = (time.time() - start) * 1000

    # Should still be under 500ms with filtering
    assert duration_ms < 500, f"Filtered search took {duration_ms:.2f}ms, expected <500ms"
    # Should filter to roughly half (alternating python/javascript)
    assert 4 <= len(results) <= 10


def test_pagination_performance(mock_client_perf):
    """Test pagination overhead."""
    client = mock_client_perf
    engine = QueryEngine(client)

    # Get multiple pages
    times = []
    for offset in [0, 10, 20, 30]:
        start = time.time()
        results = engine.search_vector([0.1] * 768, limit=10, offset=offset)
        duration_ms = (time.time() - start) * 1000
        times.append(duration_ms)

    # All pages should have similar latency (pagination is efficient)
    assert all(t < 500 for t in times), f"Some pages exceeded 500ms: {times}"
