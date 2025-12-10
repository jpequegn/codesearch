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


def test_index_command_with_scanner(tmp_path):
    """Test index command with scanner integration."""
    from codesearch.models import FileMetadata
    from datetime import datetime

    # Create a real temp directory
    test_repo = tmp_path / "test_repo"
    test_repo.mkdir()

    mock_files = [
        FileMetadata(
            file_path=str(test_repo / "test.py"),
            language="python",
            size_bytes=100,
            modified_time=datetime.now(),
            is_ignored=False,
        ),
        FileMetadata(
            file_path=str(test_repo / "utils.py"),
            language="python",
            size_bytes=200,
            modified_time=datetime.now(),
            is_ignored=False,
        ),
    ]

    with patch("codesearch.cli.commands.lancedb.connect"):
        with patch("codesearch.cli.commands.get_db_path", return_value=str(tmp_path / "test.db")):
            with patch("codesearch.cli.commands.RepositoryScannerImpl") as mock_scanner_class:
                mock_scanner = Mock()
                mock_scanner.scan_repository.return_value = mock_files
                mock_scanner.get_statistics.return_value = {
                    "total_files": 2,
                    "by_language": {"python": 2},
                    "total_size_bytes": 300,
                }
                mock_scanner.config = Mock()
                mock_scanner_class.return_value = mock_scanner

                with patch("codesearch.cli.commands.Console"):
                    result = runner.invoke(app, ["index", str(test_repo), "--force"])
                    assert result.exit_code == 0
                    assert "Found 2 files to index" in result.stdout
                    assert "python: 2 files" in result.stdout
                    mock_scanner.scan_repository.assert_called_once()


def test_index_command_no_files_found(tmp_path):
    """Test index command when no files are found."""
    # Create a real temp directory
    empty_repo = tmp_path / "empty_repo"
    empty_repo.mkdir()

    with patch("codesearch.cli.commands.lancedb.connect"):
        with patch("codesearch.cli.commands.get_db_path", return_value=str(tmp_path / "test.db")):
            with patch("codesearch.cli.commands.RepositoryScannerImpl") as mock_scanner_class:
                mock_scanner = Mock()
                mock_scanner.scan_repository.return_value = []
                mock_scanner.get_statistics.return_value = {
                    "total_files": 0,
                    "by_language": {},
                    "total_size_bytes": 0,
                }
                mock_scanner.config = Mock()
                mock_scanner_class.return_value = mock_scanner

                with patch("codesearch.cli.commands.Console"):
                    result = runner.invoke(app, ["index", str(empty_repo), "--force"])
                    assert result.exit_code == 0
                    assert "No files found" in result.stdout


def test_index_command_with_language_filter(tmp_path):
    """Test index command with language filter."""
    from codesearch.models import FileMetadata
    from datetime import datetime

    # Create a real temp directory
    test_repo = tmp_path / "test_repo"
    test_repo.mkdir()

    mock_files = [
        FileMetadata(
            file_path=str(test_repo / "test.py"),
            language="python",
            size_bytes=100,
            modified_time=datetime.now(),
            is_ignored=False,
        ),
    ]

    with patch("codesearch.cli.commands.lancedb.connect"):
        with patch("codesearch.cli.commands.get_db_path", return_value=str(tmp_path / "test.db")):
            with patch("codesearch.cli.commands.RepositoryScannerImpl") as mock_scanner_class:
                mock_scanner = Mock()
                mock_scanner.scan_repository.return_value = mock_files
                mock_scanner.get_statistics.return_value = {
                    "total_files": 1,
                    "by_language": {"python": 1},
                    "total_size_bytes": 100,
                }
                mock_scanner.config = Mock()
                mock_scanner_class.return_value = mock_scanner

                with patch("codesearch.cli.commands.Console"):
                    result = runner.invoke(
                        app, ["index", str(test_repo), "--force", "--language", "python"]
                    )
                    assert result.exit_code == 0
                    # Verify language filter was applied
                    assert mock_scanner.config.supported_languages == {"python"}


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
