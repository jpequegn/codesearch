"""Tests for CLI commands."""

from unittest.mock import Mock, patch, MagicMock
import pytest
from typer.testing import CliRunner
from codesearch.cli.main import app
from codesearch.query.models import SearchResult

runner = CliRunner()


@pytest.fixture
def mock_search_results():
    """Mock search results for testing."""
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


def test_pattern_command_with_results(mock_search_results):
    """Test pattern search command with valid results."""
    with patch("codesearch.cli.commands.QueryEngine") as mock_engine_class:
        mock_engine = Mock()
        mock_engine.search_text.return_value = mock_search_results
        mock_engine_class.return_value = mock_engine

        with patch("codesearch.cli.commands.get_db_path", return_value="/tmp/test.db"):
            with patch("codesearch.cli.commands.validate_db_exists", return_value=True):
                with patch("codesearch.cli.commands.lancedb.connect"):
                    result = runner.invoke(app, ["pattern", "parse function"])
                    assert result.exit_code == 0
                    assert "parse" in result.stdout


def test_pattern_command_no_database():
    """Test pattern command when database doesn't exist."""
    with patch("codesearch.cli.commands.get_db_path", return_value="/tmp/test.db"):
        with patch("codesearch.cli.commands.validate_db_exists", return_value=False):
            result = runner.invoke(app, ["pattern", "parse function"])
            assert result.exit_code == 2
            assert "Database not found" in result.stdout


def test_pattern_command_no_results():
    """Test pattern command with no results."""
    with patch("codesearch.cli.commands.QueryEngine") as mock_engine_class:
        mock_engine = Mock()
        mock_engine.search_text.return_value = []
        mock_engine_class.return_value = mock_engine

        with patch("codesearch.cli.commands.get_db_path", return_value="/tmp/test.db"):
            with patch("codesearch.cli.commands.validate_db_exists", return_value=True):
                with patch("codesearch.cli.commands.lancedb.connect"):
                    result = runner.invoke(app, ["pattern", "nonexistent"])
                    assert result.exit_code == 0
                    assert "No results found" in result.stdout


def test_pattern_command_json_output(mock_search_results):
    """Test pattern command with JSON output."""
    with patch("codesearch.cli.commands.QueryEngine") as mock_engine_class:
        mock_engine = Mock()
        mock_engine.search_text.return_value = mock_search_results
        mock_engine_class.return_value = mock_engine

        with patch("codesearch.cli.commands.get_db_path", return_value="/tmp/test.db"):
            with patch("codesearch.cli.commands.validate_db_exists", return_value=True):
                with patch("codesearch.cli.commands.lancedb.connect"):
                    result = runner.invoke(app, ["pattern", "parse", "--output", "json"])
                    assert result.exit_code == 0
                    assert '"name": "parse"' in result.stdout or '"name":"parse"' in result.stdout


def test_pattern_command_with_limit(mock_search_results):
    """Test pattern command with custom limit."""
    with patch("codesearch.cli.commands.QueryEngine") as mock_engine_class:
        mock_engine = Mock()
        mock_engine.search_text.return_value = mock_search_results[:1]
        mock_engine_class.return_value = mock_engine

        with patch("codesearch.cli.commands.get_db_path", return_value="/tmp/test.db"):
            with patch("codesearch.cli.commands.validate_db_exists", return_value=True):
                with patch("codesearch.cli.commands.lancedb.connect"):
                    result = runner.invoke(app, ["pattern", "parse", "--limit", "1"])
                    assert result.exit_code == 0
                    mock_engine.search_text.assert_called_once()


def test_find_similar_command(mock_search_results):
    """Test find-similar command."""
    with patch("codesearch.cli.commands.QueryEngine") as mock_engine_class:
        mock_engine = Mock()
        mock_engine.search_vector.return_value = mock_search_results[1:]  # Exclude original
        mock_engine_class.return_value = mock_engine

        with patch("codesearch.cli.commands.get_db_path", return_value="/tmp/test.db"):
            with patch("codesearch.cli.commands.validate_db_exists", return_value=True):
                with patch("codesearch.cli.commands.lancedb.connect"):
                    result = runner.invoke(app, ["find-similar", "parse"])
                    assert result.exit_code == 0


def test_find_similar_no_database():
    """Test find-similar when database doesn't exist."""
    with patch("codesearch.cli.commands.get_db_path", return_value="/tmp/test.db"):
        with patch("codesearch.cli.commands.validate_db_exists", return_value=False):
            result = runner.invoke(app, ["find-similar", "parse"])
            assert result.exit_code == 2
            assert "Database not found" in result.stdout


def test_dependencies_command():
    """Test dependencies command."""
    with patch("codesearch.cli.commands.lancedb.connect") as mock_connect:
        with patch("codesearch.cli.commands.get_db_path", return_value="/tmp/test.db"):
            with patch("codesearch.cli.commands.validate_db_exists", return_value=True):
                result = runner.invoke(app, ["dependencies", "parse"])
                assert result.exit_code == 0
                assert "Dependencies for 'parse'" in result.stdout


def test_dependencies_no_database():
    """Test dependencies when database doesn't exist."""
    with patch("codesearch.cli.commands.get_db_path", return_value="/tmp/test.db"):
        with patch("codesearch.cli.commands.validate_db_exists", return_value=False):
            result = runner.invoke(app, ["dependencies", "parse"])
            assert result.exit_code == 2


def test_index_command():
    """Test index command."""
    with patch("codesearch.cli.commands.lancedb.connect"):
        with patch("codesearch.cli.commands.get_db_path", return_value="/tmp/test.db"):
            with patch("os.makedirs"):
                result = runner.invoke(app, ["index", "/tmp/test_repo"])
                assert result.exit_code == 0
                assert "Indexing" in result.stdout


def test_refactor_dupes_command():
    """Test refactor-dupes command."""
    with patch("codesearch.cli.commands.lancedb.connect"):
        with patch("codesearch.cli.commands.get_db_path", return_value="/tmp/test.db"):
            with patch("codesearch.cli.commands.validate_db_exists", return_value=True):
                result = runner.invoke(app, ["refactor-dupes"])
                assert result.exit_code == 0
                assert "Finding duplicates" in result.stdout


def test_refactor_dupes_no_database():
    """Test refactor-dupes when database doesn't exist."""
    with patch("codesearch.cli.commands.get_db_path", return_value="/tmp/test.db"):
        with patch("codesearch.cli.commands.validate_db_exists", return_value=False):
            result = runner.invoke(app, ["refactor-dupes"])
            assert result.exit_code == 2


def test_refactor_dupes_custom_threshold():
    """Test refactor-dupes with custom threshold."""
    with patch("codesearch.cli.commands.lancedb.connect"):
        with patch("codesearch.cli.commands.get_db_path", return_value="/tmp/test.db"):
            with patch("codesearch.cli.commands.validate_db_exists", return_value=True):
                result = runner.invoke(app, ["refactor-dupes", "--threshold", "0.90"])
                assert result.exit_code == 0
                assert "0.9" in result.stdout
