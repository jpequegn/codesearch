"""Tests for output formatting."""

import json
import pytest
from codesearch.query.models import SearchResult
from codesearch.cli.formatting import (
    format_results_json,
    format_results_table,
    format_results,
    display_call_graph,
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
    """Test table formatting."""
    table_output = format_results_table(sample_results)
    assert "parse" in table_output
    assert "python" in table_output
    assert "0.95" in table_output
    assert "┌" in table_output  # Check for box drawing characters


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
    assert "┌" in table_output


def test_display_call_graph():
    """Test call graph display."""
    relationships = [
        {"caller": "main", "callee": "parse"},
        {"caller": "parse", "callee": "process"},
    ]
    output = display_call_graph(relationships)
    assert "Call Graph:" in output
    assert "main → parse" in output
    assert "parse → process" in output


def test_display_call_graph_empty():
    """Test call graph display with empty relationships."""
    output = display_call_graph([])
    assert "No relationships found" in output
