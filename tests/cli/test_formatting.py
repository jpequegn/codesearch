"""Tests for output formatting."""

import json
import pytest
from codesearch.query.models import SearchResult
from codesearch.cli.formatting import (
    format_results_json,
    format_results_table,
    format_results,
    display_call_graph,
    highlight_code,
    display_code_snippet,
    ProgressTracker,
)


@pytest.fixture
def sample_results():
    """Sample search results for testing."""
    return [
        SearchResult(
            entity_id="repo:file.py:parse",
            name="parse",
            code_text="def parse(): pass",
            similarity_score=0.95,
            language="python",
            file_path="file.py",
            repository="repo",
            entity_type="function",
            start_line=1,
            end_line=2,
        ),
        SearchResult(
            entity_id="repo:util.py:process",
            name="process",
            code_text="def process(): pass",
            similarity_score=0.87,
            language="python",
            file_path="util.py",
            repository="repo",
            entity_type="function",
            start_line=10,
            end_line=15,
        ),
    ]


def test_format_results_json(sample_results):
    """Test JSON formatting."""
    json_output = format_results_json(sample_results)
    data = json.loads(json_output)
    assert len(data) == 2
    assert data[0]["name"] == "parse"
    assert data[0]["similarity_score"] == 0.95
    assert data[1]["name"] == "process"


def test_format_results_table(sample_results):
    """Test table formatting with Rich table."""
    table_output = format_results_table(sample_results)
    assert "parse" in table_output
    assert "python" in table_output
    assert "95" in table_output  # 0.95 = 95%
    # Rich tables use different box drawing characters
    assert len(table_output) > 0


def test_format_results_table_empty():
    """Test table formatting with empty results."""
    table_output = format_results_table([])
    assert "No results found" in table_output


def test_format_results_dispatch(sample_results):
    """Test format_results dispatcher function."""
    # Test JSON format
    json_output = format_results(sample_results, format="json")
    data = json.loads(json_output)
    assert len(data) == 2

    # Test table format
    table_output = format_results(sample_results, format="table")
    assert "parse" in table_output
    assert len(table_output) > 0


def test_highlight_code():
    """Test syntax highlighting for code."""
    code = "def hello():\n    print('world')"
    highlighted = highlight_code(code, language="python")
    # Should contain the original code (may have ANSI codes)
    assert "def" in highlighted
    assert "hello" in highlighted
    assert "print" in highlighted
    # Should be non-empty
    assert len(highlighted) > 0


def test_highlight_code_invalid_language():
    """Test syntax highlighting with invalid language."""
    code = "def hello(): pass"
    highlighted = highlight_code(code, language="unknown_language_xyz")
    # Should fall back to plain code
    assert "def hello" in highlighted


def test_display_code_snippet():
    """Test code snippet display with line numbers."""
    code = "def foo():\n    return 42"
    snippet = display_code_snippet(code, language="python", start_line=10)
    # Should contain code and line numbers
    assert "def foo" in snippet or "foo" in snippet
    assert len(snippet) > 0


def test_display_code_snippet_multiline():
    """Test code snippet with multiple lines."""
    code = "def add(a, b):\n    return a + b\n\ndef multiply(x, y):\n    return x * y"
    snippet = display_code_snippet(code, language="python", start_line=5)
    assert "add" in snippet or "multiply" in snippet
    assert len(snippet) > 0


def test_display_call_graph():
    """Test call graph display with tree visualization."""
    relationships = [
        {"caller": "main", "callee": "parse"},
        {"caller": "parse", "callee": "process"},
    ]
    output = display_call_graph(relationships)
    # Should contain graph elements
    assert "main" in output
    assert "parse" in output
    assert "process" in output
    assert len(output) > 0


def test_display_call_graph_with_source_target():
    """Test call graph with source_id and target_id keys."""
    relationships = [
        {"source_id": "module1:func1", "target_id": "module2:func2", "relationship_type": "calls"},
        {"source_id": "module2:func2", "target_id": "module3:func3", "relationship_type": "calls"},
    ]
    output = display_call_graph(relationships)
    assert "func1" in output
    assert "func2" in output
    assert "func3" in output
    assert len(output) > 0


def test_display_call_graph_empty():
    """Test call graph display with empty relationships."""
    output = display_call_graph([])
    assert "No relationships found" in output


def test_progress_tracker_context_manager():
    """Test progress tracker as context manager."""
    with ProgressTracker(description="Testing", total=10) as tracker:
        assert tracker.task_id is not None
        assert tracker.progress is not None
        tracker.update(advance=5)
        tracker.update(advance=3)
        tracker.update(advance=2)


def test_progress_tracker_update_with_description():
    """Test progress tracker update with new description."""
    with ProgressTracker(description="Initial", total=5) as tracker:
        tracker.update(advance=2, description="Updated")
        tracker.update(advance=3)
