import pytest
from codesearch.query.filters import (
    LanguageFilter, FilePathFilter, EntityTypeFilter, RepositoryFilter
)
from codesearch.query.models import SearchResult


@pytest.fixture
def sample_result():
    """Create a sample SearchResult for filter testing."""
    return SearchResult(
        entity_id="repo:src/parser.py:Parser:parse",
        name="parse",
        code_text="def parse(self): pass",
        similarity_score=0.9,
        language="python",
        file_path="src/parser.py",
        repository="repo",
        entity_type="function",
        start_line=10,
        end_line=12
    )


def test_language_filter_single(sample_result):
    """Test LanguageFilter with single language."""
    filter = LanguageFilter(['python'])
    assert filter.matches(sample_result) is True


def test_language_filter_multiple(sample_result):
    """Test LanguageFilter with multiple languages (OR logic)."""
    filter = LanguageFilter(['python', 'javascript'])
    assert filter.matches(sample_result) is True

    filter_no_match = LanguageFilter(['javascript', 'go'])
    assert filter_no_match.matches(sample_result) is False


def test_file_path_filter(sample_result):
    """Test FilePathFilter with substring matching."""
    filter = FilePathFilter(['src/'])
    assert filter.matches(sample_result) is True

    filter_no_match = FilePathFilter(['lib/', 'tests/'])
    assert filter_no_match.matches(sample_result) is False


def test_entity_type_filter(sample_result):
    """Test EntityTypeFilter."""
    filter = EntityTypeFilter(['function', 'method'])
    assert filter.matches(sample_result) is True

    filter_no_match = EntityTypeFilter(['class', 'module'])
    assert filter_no_match.matches(sample_result) is False


def test_repository_filter(sample_result):
    """Test RepositoryFilter."""
    filter = RepositoryFilter(['repo'])
    assert filter.matches(sample_result) is True

    filter_no_match = RepositoryFilter(['other-repo'])
    assert filter_no_match.matches(sample_result) is False
