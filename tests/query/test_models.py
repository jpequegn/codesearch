from dataclasses import dataclass
from codesearch.query.models import SearchResult


def test_search_result_creation():
    """Test SearchResult dataclass creation and fields."""
    result = SearchResult(
        entity_id="repo:file.py:Parser:parse",
        name="parse",
        code_text="def parse(self): return self.data",
        similarity_score=0.95,
        language="python",
        file_path="file.py",
        repository="repo",
        entity_type="function",
        start_line=10,
        end_line=12
    )

    assert result.entity_id == "repo:file.py:Parser:parse"
    assert result.name == "parse"
    assert result.similarity_score == 0.95
    assert result.language == "python"
    assert result.entity_type == "function"


def test_search_result_string_representation():
    """Test SearchResult string representation for CLI output."""
    result = SearchResult(
        entity_id="repo:file.py:Utils:format",
        name="format",
        code_text="def format(data): return str(data)",
        similarity_score=0.87,
        language="javascript",
        file_path="utils.js",
        repository="repo",
        entity_type="function",
        start_line=5,
        end_line=6
    )

    result_str = str(result)
    assert "format" in result_str
    assert "javascript" in result_str
    assert "0.87" in result_str
