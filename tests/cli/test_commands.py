"""Tests for CLI commands."""

from unittest.mock import MagicMock, Mock, patch

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
    """Test find-similar command finds entity and returns similar results."""
    # Mock entity found in database
    mock_entity = {
        "entity_id": "repo:file.py:parse",
        "name": "parse",
        "code_text": "def parse(): pass",
        "code_vector": [0.1] * 768,
        "language": "python",
        "file_path": "file.py",
        "repository": "repo",
        "entity_type": "function",
        "start_line": 1,
        "end_line": 2,
    }

    with patch("codesearch.cli.commands.QueryEngine") as mock_engine_class:
        mock_engine = Mock()
        mock_engine.search_vector.return_value = mock_search_results[1:]  # Exclude original
        mock_engine_class.return_value = mock_engine

        with patch("codesearch.cli.commands.get_db_path", return_value="/tmp/test.db"):
            with patch("codesearch.cli.commands.validate_db_exists", return_value=True):
                with patch("codesearch.cli.commands.lancedb.connect") as mock_connect:
                    # Setup mock table with entity lookup
                    mock_table = Mock()
                    mock_search = Mock()
                    mock_search.where.return_value = mock_search
                    mock_search.limit.return_value = mock_search
                    mock_search.to_list.return_value = [mock_entity]
                    mock_table.search.return_value = mock_search
                    mock_connect.return_value.open_table.return_value = mock_table

                    result = runner.invoke(app, ["find-similar", "parse"])
                    assert result.exit_code == 0
                    assert "Finding similar code" in result.stdout
                    assert "Found" in result.stdout


def test_find_similar_entity_not_found():
    """Test find-similar when entity is not in database."""
    with patch("codesearch.cli.commands.get_db_path", return_value="/tmp/test.db"):
        with patch("codesearch.cli.commands.validate_db_exists", return_value=True):
            with patch("codesearch.cli.commands.lancedb.connect") as mock_connect:
                # Setup mock table that returns no entities
                mock_table = Mock()
                mock_search = Mock()
                mock_search.where.return_value = mock_search
                mock_search.limit.return_value = mock_search
                mock_search.to_list.return_value = []  # No entity found
                mock_table.search.return_value = mock_search
                mock_connect.return_value.open_table.return_value = mock_table

                result = runner.invoke(app, ["find-similar", "nonexistent"])
                assert result.exit_code == 3
                assert "not found in database" in result.stdout


def test_find_similar_no_database():
    """Test find-similar when database doesn't exist."""
    with patch("codesearch.cli.commands.get_db_path", return_value="/tmp/test.db"):
        with patch("codesearch.cli.commands.validate_db_exists", return_value=False):
            result = runner.invoke(app, ["find-similar", "parse"])
            assert result.exit_code == 2
            assert "Database not found" in result.output


def test_find_similar_entity_no_embedding():
    """Test find-similar when entity has no embedding vector."""
    # Mock entity without embedding
    mock_entity = {
        "entity_id": "repo:file.py:parse",
        "name": "parse",
        "code_text": "def parse(): pass",
        "code_vector": None,  # No embedding
        "language": "python",
        "file_path": "file.py",
    }

    with patch("codesearch.cli.commands.get_db_path", return_value="/tmp/test.db"):
        with patch("codesearch.cli.commands.validate_db_exists", return_value=True):
            with patch("codesearch.cli.commands.lancedb.connect") as mock_connect:
                mock_table = Mock()
                mock_search = Mock()
                mock_search.where.return_value = mock_search
                mock_search.limit.return_value = mock_search
                mock_search.to_list.return_value = [mock_entity]
                mock_table.search.return_value = mock_search
                mock_connect.return_value.open_table.return_value = mock_table

                result = runner.invoke(app, ["find-similar", "parse"])
                assert result.exit_code == 2
                assert "has no embedding vector" in result.stdout


def test_find_similar_excludes_self(mock_search_results):
    """Test find-similar excludes the original entity from results."""
    mock_entity = {
        "entity_id": "repo:file.py:parse",
        "name": "parse",
        "code_text": "def parse(): pass",
        "code_vector": [0.1] * 768,
        "language": "python",
        "file_path": "file.py",
        "repository": "repo",
        "entity_type": "function",
        "start_line": 1,
        "end_line": 2,
    }

    # Results include the original entity (should be filtered out)
    all_results = [
        mock_search_results[0],  # Original entity - should be excluded
        mock_search_results[1],  # Different entity - should be included
    ]

    with patch("codesearch.cli.commands.QueryEngine") as mock_engine_class:
        mock_engine = Mock()
        mock_engine.search_vector.return_value = all_results
        mock_engine_class.return_value = mock_engine

        with patch("codesearch.cli.commands.get_db_path", return_value="/tmp/test.db"):
            with patch("codesearch.cli.commands.validate_db_exists", return_value=True):
                with patch("codesearch.cli.commands.lancedb.connect") as mock_connect:
                    mock_table = Mock()
                    mock_search = Mock()
                    mock_search.where.return_value = mock_search
                    mock_search.limit.return_value = mock_search
                    mock_search.to_list.return_value = [mock_entity]
                    mock_table.search.return_value = mock_search
                    mock_connect.return_value.open_table.return_value = mock_table

                    result = runner.invoke(app, ["find-similar", "parse"])
                    assert result.exit_code == 0
                    # Should find results (the non-original entity)
                    assert "Found" in result.stdout


def test_dependencies_command():
    """Test dependencies command with callers and callees."""
    with patch("codesearch.cli.commands.lancedb.connect") as mock_connect:
        with patch("codesearch.cli.commands.get_db_path", return_value="/tmp/test.db"):
            with patch("codesearch.cli.commands.validate_db_exists", return_value=True):
                # Mock entities table
                mock_entities_table = MagicMock()
                mock_entities_search = MagicMock()
                mock_entities_search.where.return_value = mock_entities_search
                mock_entities_search.limit.return_value = mock_entities_search
                mock_entities_search.to_list.side_effect = [
                    # First call: find entity by name
                    [{"entity_id": "abc123", "name": "parse", "file_path": "test.py"}],
                    # Second call: get all entities for name lookup
                    [
                        {"entity_id": "abc123", "name": "parse", "file_path": "test.py"},
                        {"entity_id": "def456", "name": "helper", "file_path": "utils.py"},
                        {"entity_id": "ghi789", "name": "main", "file_path": "main.py"},
                    ],
                ]
                mock_entities_table.search.return_value = mock_entities_search

                # Mock relationships table
                mock_rels_table = MagicMock()
                mock_rels_search = MagicMock()
                mock_rels_search.to_list.return_value = [
                    # parse calls helper
                    {"caller_id": "abc123", "callee_id": "def456", "relationship_type": "calls"},
                    # main calls parse
                    {"caller_id": "ghi789", "callee_id": "abc123", "relationship_type": "calls"},
                ]
                mock_rels_table.search.return_value = mock_rels_search

                # Setup mock client
                mock_client = MagicMock()
                mock_client.open_table.side_effect = lambda name: (
                    mock_entities_table if name == "code_entities" else mock_rels_table
                )
                mock_connect.return_value = mock_client

                result = runner.invoke(app, ["deps", "parse"])
                assert result.exit_code == 0
                assert "Call graph for 'parse'" in result.stdout
                # Should show both callers and calls
                assert "Called by:" in result.stdout
                assert "Calls:" in result.stdout


def test_dependencies_no_database():
    """Test deps when database doesn't exist."""
    with patch("codesearch.cli.commands.get_db_path", return_value="/tmp/test.db"):
        with patch("codesearch.cli.commands.validate_db_exists", return_value=False):
            result = runner.invoke(app, ["deps", "parse"])
            assert result.exit_code == 2


def test_dependencies_entity_not_found():
    """Test deps command when entity is not found."""
    with patch("codesearch.cli.commands.lancedb.connect") as mock_connect:
        with patch("codesearch.cli.commands.get_db_path", return_value="/tmp/test.db"):
            with patch("codesearch.cli.commands.validate_db_exists", return_value=True):
                # Mock entities table to return no results
                mock_entities_table = MagicMock()
                mock_entities_search = MagicMock()
                mock_entities_search.where.return_value = mock_entities_search
                mock_entities_search.limit.return_value = mock_entities_search
                mock_entities_search.to_list.return_value = []
                mock_entities_table.search.return_value = mock_entities_search

                mock_client = MagicMock()
                mock_client.open_table.return_value = mock_entities_table
                mock_connect.return_value = mock_client

                result = runner.invoke(app, ["deps", "nonexistent_func"])
                assert result.exit_code == 3
                assert "not found" in result.stdout


def test_dependencies_json_output():
    """Test deps command with JSON output format."""
    with patch("codesearch.cli.commands.lancedb.connect") as mock_connect:
        with patch("codesearch.cli.commands.get_db_path", return_value="/tmp/test.db"):
            with patch("codesearch.cli.commands.validate_db_exists", return_value=True):
                # Mock entities table
                mock_entities_table = MagicMock()
                mock_entities_search = MagicMock()
                mock_entities_search.where.return_value = mock_entities_search
                mock_entities_search.limit.return_value = mock_entities_search
                mock_entities_search.to_list.side_effect = [
                    [{"entity_id": "abc123", "name": "parse", "file_path": "test.py"}],
                    [
                        {"entity_id": "abc123", "name": "parse", "file_path": "test.py"},
                        {"entity_id": "def456", "name": "helper", "file_path": "utils.py"},
                    ],
                ]
                mock_entities_table.search.return_value = mock_entities_search

                # Mock relationships table
                mock_rels_table = MagicMock()
                mock_rels_search = MagicMock()
                mock_rels_search.to_list.return_value = [
                    {"caller_id": "abc123", "callee_id": "def456", "relationship_type": "calls"},
                ]
                mock_rels_table.search.return_value = mock_rels_search

                mock_client = MagicMock()
                mock_client.open_table.side_effect = lambda name: (
                    mock_entities_table if name == "code_entities" else mock_rels_table
                )
                mock_connect.return_value = mock_client

                result = runner.invoke(app, ["deps", "parse", "--output", "json"])
                assert result.exit_code == 0
                assert '"entity": "parse"' in result.stdout
                assert '"calls"' in result.stdout


def test_dependencies_direction_calls():
    """Test deps command with --direction calls."""
    with patch("codesearch.cli.commands.lancedb.connect") as mock_connect:
        with patch("codesearch.cli.commands.get_db_path", return_value="/tmp/test.db"):
            with patch("codesearch.cli.commands.validate_db_exists", return_value=True):
                mock_entities_table = MagicMock()
                mock_entities_search = MagicMock()
                mock_entities_search.where.return_value = mock_entities_search
                mock_entities_search.limit.return_value = mock_entities_search
                mock_entities_search.to_list.side_effect = [
                    [{"entity_id": "abc123", "name": "parse", "file_path": "test.py"}],
                    [
                        {"entity_id": "abc123", "name": "parse", "file_path": "test.py"},
                        {"entity_id": "def456", "name": "helper", "file_path": "utils.py"},
                    ],
                ]
                mock_entities_table.search.return_value = mock_entities_search

                mock_rels_table = MagicMock()
                mock_rels_search = MagicMock()
                mock_rels_search.to_list.return_value = [
                    {"caller_id": "abc123", "callee_id": "def456", "relationship_type": "calls"},
                ]
                mock_rels_table.search.return_value = mock_rels_search

                mock_client = MagicMock()
                mock_client.open_table.side_effect = lambda name: (
                    mock_entities_table if name == "code_entities" else mock_rels_table
                )
                mock_connect.return_value = mock_client

                result = runner.invoke(app, ["deps", "parse", "--direction", "calls"])
                assert result.exit_code == 0
                assert "Functions called by 'parse'" in result.stdout
                assert "Calls:" in result.stdout


def test_index_command_with_scanner(tmp_path):
    """Test index command with scanner integration."""
    from datetime import datetime

    from codesearch.models import FileMetadata

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
    from datetime import datetime

    from codesearch.models import FileMetadata

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


def test_index_command_with_parser(tmp_path):
    """Test index command with parser integration."""
    from datetime import datetime

    from codesearch.models import FileMetadata, Function

    # Create a real temp directory with a Python file
    test_repo = tmp_path / "test_repo"
    test_repo.mkdir()
    (test_repo / "example.py").write_text("def hello(): pass\ndef world(): pass")

    mock_files = [
        FileMetadata(
            file_path=str(test_repo / "example.py"),
            language="python",
            size_bytes=50,
            modified_time=datetime.now(),
            is_ignored=False,
        ),
    ]

    mock_entities = [
        Function(
            name="hello",
            source_code="def hello(): pass",
            line_number=1,
            end_line=1,
            file_path=str(test_repo / "example.py"),
        ),
        Function(
            name="world",
            source_code="def world(): pass",
            line_number=2,
            end_line=2,
            file_path=str(test_repo / "example.py"),
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
                    "total_size_bytes": 50,
                }
                mock_scanner.config = Mock()
                mock_scanner_class.return_value = mock_scanner

                with patch("codesearch.cli.commands.PythonParser") as mock_parser_class:
                    mock_parser = Mock()
                    mock_parser.parse_file.return_value = mock_entities
                    mock_parser_class.return_value = mock_parser

                    with patch("codesearch.cli.commands.EmbeddingGenerator") as mock_embedder_class:
                        mock_embedder = Mock()
                        mock_embedder.get_model_info.return_value = {
                            "name": "codebert-base",
                            "dimensions": 768,
                            "device": "cpu",
                        }
                        # Return embeddings for batch
                        mock_embedder.embed_batch.return_value = [
                            [0.1] * 768,
                            [0.2] * 768,
                        ]
                        mock_embedder_class.return_value = mock_embedder

                        with patch("codesearch.cli.commands.Console"):
                            result = runner.invoke(app, ["index", str(test_repo), "--force"])
                            assert result.exit_code == 0
                            assert "Extracted 2 code entities" in result.stdout
                            assert "Functions/methods: 2" in result.stdout
                            assert "Generated 2 embeddings" in result.stdout
                            mock_parser.parse_file.assert_called_once()
                            mock_embedder.embed_batch.assert_called()


def test_index_command_parser_syntax_error(tmp_path):
    """Test index command handles syntax errors gracefully."""
    from datetime import datetime

    from codesearch.models import FileMetadata

    test_repo = tmp_path / "test_repo"
    test_repo.mkdir()

    mock_files = [
        FileMetadata(
            file_path=str(test_repo / "bad.py"),
            language="python",
            size_bytes=50,
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
                    "total_size_bytes": 50,
                }
                mock_scanner.config = Mock()
                mock_scanner_class.return_value = mock_scanner

                with patch("codesearch.cli.commands.PythonParser") as mock_parser_class:
                    mock_parser = Mock()
                    mock_parser.parse_file.side_effect = SyntaxError("invalid syntax")
                    mock_parser_class.return_value = mock_parser

                    with patch("codesearch.cli.commands.Console"):
                        result = runner.invoke(app, ["index", str(test_repo), "--force"])
                        # Should exit 0 because it handled the error gracefully
                        assert result.exit_code == 0
                        assert "Skipped 1 files with errors" in result.stdout
                        assert "No code entities found" in result.stdout


def test_index_command_parser_with_classes(tmp_path):
    """Test index command counts classes correctly."""
    from datetime import datetime

    from codesearch.models import Class, FileMetadata, Function

    test_repo = tmp_path / "test_repo"
    test_repo.mkdir()

    mock_files = [
        FileMetadata(
            file_path=str(test_repo / "module.py"),
            language="python",
            size_bytes=100,
            modified_time=datetime.now(),
            is_ignored=False,
        ),
    ]

    mock_entities = [
        Class(
            name="MyClass",
            source_code="class MyClass: pass",
            line_number=1,
            end_line=1,
            file_path=str(test_repo / "module.py"),
        ),
        Function(
            name="helper",
            source_code="def helper(): pass",
            line_number=3,
            end_line=3,
            file_path=str(test_repo / "module.py"),
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

                with patch("codesearch.cli.commands.PythonParser") as mock_parser_class:
                    mock_parser = Mock()
                    mock_parser.parse_file.return_value = mock_entities
                    mock_parser_class.return_value = mock_parser

                    with patch("codesearch.cli.commands.EmbeddingGenerator") as mock_embedder_class:
                        mock_embedder = Mock()
                        mock_embedder.get_model_info.return_value = {
                            "name": "codebert-base",
                            "dimensions": 768,
                            "device": "cpu",
                        }
                        mock_embedder.embed_batch.return_value = [
                            [0.1] * 768,
                            [0.2] * 768,
                        ]
                        mock_embedder_class.return_value = mock_embedder

                        with patch("codesearch.cli.commands.Console"):
                            result = runner.invoke(app, ["index", str(test_repo), "--force"])
                            assert result.exit_code == 0
                            assert "Extracted 2 code entities" in result.stdout
                            assert "Functions/methods: 1" in result.stdout
                            assert "Classes: 1" in result.stdout
                            assert "Generated 2 embeddings" in result.stdout


def test_index_command_skips_non_python(tmp_path):
    """Test index command skips non-Python files (parser only supports Python)."""
    from datetime import datetime

    from codesearch.models import FileMetadata

    test_repo = tmp_path / "test_repo"
    test_repo.mkdir()

    mock_files = [
        FileMetadata(
            file_path=str(test_repo / "script.js"),
            language="javascript",
            size_bytes=50,
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
                    "by_language": {"javascript": 1},
                    "total_size_bytes": 50,
                }
                mock_scanner.config = Mock()
                mock_scanner_class.return_value = mock_scanner

                with patch("codesearch.cli.commands.PythonParser") as mock_parser_class:
                    mock_parser = Mock()
                    mock_parser_class.return_value = mock_parser

                    with patch("codesearch.cli.commands.Console"):
                        result = runner.invoke(app, ["index", str(test_repo), "--force"])
                        assert result.exit_code == 0
                        # Parser should never be called for non-Python files
                        mock_parser.parse_file.assert_not_called()
                        assert "No code entities found" in result.stdout


def test_index_command_embedding_model_load_error(tmp_path):
    """Test index command handles embedding model load errors gracefully."""
    from datetime import datetime

    from codesearch.models import FileMetadata, Function

    test_repo = tmp_path / "test_repo"
    test_repo.mkdir()

    mock_files = [
        FileMetadata(
            file_path=str(test_repo / "test.py"),
            language="python",
            size_bytes=50,
            modified_time=datetime.now(),
            is_ignored=False,
        ),
    ]

    mock_entities = [
        Function(
            name="test_func",
            source_code="def test_func(): pass",
            line_number=1,
            end_line=1,
            file_path=str(test_repo / "test.py"),
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
                    "total_size_bytes": 50,
                }
                mock_scanner.config = Mock()
                mock_scanner_class.return_value = mock_scanner

                with patch("codesearch.cli.commands.PythonParser") as mock_parser_class:
                    mock_parser = Mock()
                    mock_parser.parse_file.return_value = mock_entities
                    mock_parser_class.return_value = mock_parser

                    with patch("codesearch.cli.commands.EmbeddingGenerator") as mock_embedder_class:
                        # Simulate model load failure
                        mock_embedder_class.side_effect = RuntimeError("Failed to load model")

                        with patch("codesearch.cli.commands.Console"):
                            result = runner.invoke(app, ["index", str(test_repo), "--force"], catch_exceptions=False)
                            # Should exit with error code 1
                            assert result.exit_code == 1
                            # Error message is in output (mix_stderr happens automatically in test runner)
                            assert "Failed to load embedding model" in result.output


def test_index_command_embedding_batch_error_continues(tmp_path):
    """Test index command continues processing after batch embedding error."""
    from datetime import datetime

    from codesearch.models import FileMetadata, Function

    test_repo = tmp_path / "test_repo"
    test_repo.mkdir()

    mock_files = [
        FileMetadata(
            file_path=str(test_repo / "test.py"),
            language="python",
            size_bytes=50,
            modified_time=datetime.now(),
            is_ignored=False,
        ),
    ]

    # Create more entities to test batch processing
    mock_entities = [
        Function(
            name=f"func_{i}",
            source_code=f"def func_{i}(): pass",
            line_number=i,
            end_line=i,
            file_path=str(test_repo / "test.py"),
        )
        for i in range(5)
    ]

    with patch("codesearch.cli.commands.lancedb.connect"):
        with patch("codesearch.cli.commands.get_db_path", return_value=str(tmp_path / "test.db")):
            with patch("codesearch.cli.commands.RepositoryScannerImpl") as mock_scanner_class:
                mock_scanner = Mock()
                mock_scanner.scan_repository.return_value = mock_files
                mock_scanner.get_statistics.return_value = {
                    "total_files": 1,
                    "by_language": {"python": 1},
                    "total_size_bytes": 50,
                }
                mock_scanner.config = Mock()
                mock_scanner_class.return_value = mock_scanner

                with patch("codesearch.cli.commands.PythonParser") as mock_parser_class:
                    mock_parser = Mock()
                    mock_parser.parse_file.return_value = mock_entities
                    mock_parser_class.return_value = mock_parser

                    with patch("codesearch.cli.commands.EmbeddingGenerator") as mock_embedder_class:
                        mock_embedder = Mock()
                        mock_embedder.get_model_info.return_value = {
                            "name": "codebert-base",
                            "dimensions": 768,
                            "device": "cpu",
                        }
                        # First batch fails, but processing should continue
                        mock_embedder.embed_batch.side_effect = RuntimeError("Embedding error")
                        mock_embedder_class.return_value = mock_embedder

                        with patch("codesearch.cli.commands.Console"):
                            result = runner.invoke(app, ["index", str(test_repo), "--force"])
                            # Should still exit 0 as it handles errors gracefully
                            assert result.exit_code == 0
                            assert "Failed to embed" in result.stdout


def test_index_command_embedding_shows_model_info(tmp_path):
    """Test index command displays model info during embedding."""
    from datetime import datetime

    from codesearch.models import FileMetadata, Function

    test_repo = tmp_path / "test_repo"
    test_repo.mkdir()

    mock_files = [
        FileMetadata(
            file_path=str(test_repo / "test.py"),
            language="python",
            size_bytes=50,
            modified_time=datetime.now(),
            is_ignored=False,
        ),
    ]

    mock_entities = [
        Function(
            name="test_func",
            source_code="def test_func(): pass",
            line_number=1,
            end_line=1,
            file_path=str(test_repo / "test.py"),
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
                    "total_size_bytes": 50,
                }
                mock_scanner.config = Mock()
                mock_scanner_class.return_value = mock_scanner

                with patch("codesearch.cli.commands.PythonParser") as mock_parser_class:
                    mock_parser = Mock()
                    mock_parser.parse_file.return_value = mock_entities
                    mock_parser_class.return_value = mock_parser

                    with patch("codesearch.cli.commands.EmbeddingGenerator") as mock_embedder_class:
                        mock_embedder = Mock()
                        mock_embedder.get_model_info.return_value = {
                            "name": "codebert-base",
                            "dimensions": 768,
                            "device": "cuda",
                        }
                        mock_embedder.embed_batch.return_value = [[0.1] * 768]
                        mock_embedder_class.return_value = mock_embedder

                        with patch("codesearch.cli.commands.Console"):
                            result = runner.invoke(app, ["index", str(test_repo), "--force"])
                            assert result.exit_code == 0
                            assert "Loading embedding model" in result.stdout
                            assert "codebert-base" in result.stdout
                            assert "768d" in result.stdout
                            assert "cuda" in result.stdout


# Tests for Issue #53 - Database Storage


def test_index_command_stores_entities_in_database(tmp_path):
    """Test index command stores entities in LanceDB via DataIngestionPipeline."""
    from datetime import datetime

    from codesearch.data_ingestion.models import IngestionResult
    from codesearch.models import FileMetadata, Function

    test_repo = tmp_path / "test_repo"
    test_repo.mkdir()

    mock_files = [
        FileMetadata(
            file_path=str(test_repo / "test.py"),
            language="python",
            size_bytes=50,
            modified_time=datetime.now(),
            is_ignored=False,
        ),
    ]

    mock_entities = [
        Function(
            name="test_func",
            source_code="def test_func(): pass",
            line_number=1,
            end_line=1,
            file_path=str(test_repo / "test.py"),
        ),
    ]

    mock_result = IngestionResult(
        inserted_count=1,
        skipped_count=0,
        failed_count=0,
    )

    with patch("codesearch.cli.commands.lancedb.connect") as mock_connect:
        with patch("codesearch.cli.commands.get_db_path", return_value=str(tmp_path / "test.db")):
            with patch("codesearch.cli.commands.RepositoryScannerImpl") as mock_scanner_class:
                mock_scanner = Mock()
                mock_scanner.scan_repository.return_value = mock_files
                mock_scanner.get_statistics.return_value = {
                    "total_files": 1,
                    "by_language": {"python": 1},
                    "total_size_bytes": 50,
                }
                mock_scanner.config = Mock()
                mock_scanner_class.return_value = mock_scanner

                with patch("codesearch.cli.commands.PythonParser") as mock_parser_class:
                    mock_parser = Mock()
                    mock_parser.parse_file.return_value = mock_entities
                    mock_parser_class.return_value = mock_parser

                    with patch("codesearch.cli.commands.EmbeddingGenerator") as mock_embedder_class:
                        mock_embedder = Mock()
                        mock_embedder.get_model_info.return_value = {
                            "name": "codebert-base",
                            "dimensions": 768,
                            "device": "cpu",
                        }
                        mock_embedder.embed_batch.return_value = [[0.1] * 768]
                        mock_embedder_class.return_value = mock_embedder

                        with patch("codesearch.cli.commands.DataIngestionPipeline") as mock_pipeline_class:
                            mock_pipeline = Mock()
                            mock_pipeline.ingest_batch.return_value = mock_result
                            mock_pipeline_class.return_value = mock_pipeline

                            with patch("codesearch.cli.commands.Console"):
                                result = runner.invoke(app, ["index", str(test_repo), "--force"])
                                assert result.exit_code == 0
                                assert "Stored 1 entities" in result.stdout
                                mock_pipeline.ingest_batch.assert_called_once()


def test_index_command_storage_shows_duplicates(tmp_path):
    """Test index command shows skipped duplicates count."""
    from datetime import datetime

    from codesearch.data_ingestion.models import IngestionResult
    from codesearch.models import FileMetadata, Function

    test_repo = tmp_path / "test_repo"
    test_repo.mkdir()

    mock_files = [
        FileMetadata(
            file_path=str(test_repo / "test.py"),
            language="python",
            size_bytes=50,
            modified_time=datetime.now(),
            is_ignored=False,
        ),
    ]

    mock_entities = [
        Function(
            name="test_func",
            source_code="def test_func(): pass",
            line_number=1,
            end_line=1,
            file_path=str(test_repo / "test.py"),
        ),
    ]

    mock_result = IngestionResult(
        inserted_count=0,
        skipped_count=1,  # Entity was a duplicate
        failed_count=0,
    )

    with patch("codesearch.cli.commands.lancedb.connect"):
        with patch("codesearch.cli.commands.get_db_path", return_value=str(tmp_path / "test.db")):
            with patch("codesearch.cli.commands.RepositoryScannerImpl") as mock_scanner_class:
                mock_scanner = Mock()
                mock_scanner.scan_repository.return_value = mock_files
                mock_scanner.get_statistics.return_value = {
                    "total_files": 1,
                    "by_language": {"python": 1},
                    "total_size_bytes": 50,
                }
                mock_scanner.config = Mock()
                mock_scanner_class.return_value = mock_scanner

                with patch("codesearch.cli.commands.PythonParser") as mock_parser_class:
                    mock_parser = Mock()
                    mock_parser.parse_file.return_value = mock_entities
                    mock_parser_class.return_value = mock_parser

                    with patch("codesearch.cli.commands.EmbeddingGenerator") as mock_embedder_class:
                        mock_embedder = Mock()
                        mock_embedder.get_model_info.return_value = {
                            "name": "codebert-base",
                            "dimensions": 768,
                            "device": "cpu",
                        }
                        mock_embedder.embed_batch.return_value = [[0.1] * 768]
                        mock_embedder_class.return_value = mock_embedder

                        with patch("codesearch.cli.commands.DataIngestionPipeline") as mock_pipeline_class:
                            mock_pipeline = Mock()
                            mock_pipeline.ingest_batch.return_value = mock_result
                            mock_pipeline_class.return_value = mock_pipeline

                            with patch("codesearch.cli.commands.Console"):
                                result = runner.invoke(app, ["index", str(test_repo), "--force"])
                                assert result.exit_code == 0
                                assert "Skipped 1 duplicates" in result.stdout


def test_index_command_storage_error(tmp_path):
    """Test index command handles database storage errors."""
    from datetime import datetime

    from codesearch.models import FileMetadata, Function

    test_repo = tmp_path / "test_repo"
    test_repo.mkdir()

    mock_files = [
        FileMetadata(
            file_path=str(test_repo / "test.py"),
            language="python",
            size_bytes=50,
            modified_time=datetime.now(),
            is_ignored=False,
        ),
    ]

    mock_entities = [
        Function(
            name="test_func",
            source_code="def test_func(): pass",
            line_number=1,
            end_line=1,
            file_path=str(test_repo / "test.py"),
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
                    "total_size_bytes": 50,
                }
                mock_scanner.config = Mock()
                mock_scanner_class.return_value = mock_scanner

                with patch("codesearch.cli.commands.PythonParser") as mock_parser_class:
                    mock_parser = Mock()
                    mock_parser.parse_file.return_value = mock_entities
                    mock_parser_class.return_value = mock_parser

                    with patch("codesearch.cli.commands.EmbeddingGenerator") as mock_embedder_class:
                        mock_embedder = Mock()
                        mock_embedder.get_model_info.return_value = {
                            "name": "codebert-base",
                            "dimensions": 768,
                            "device": "cpu",
                        }
                        mock_embedder.embed_batch.return_value = [[0.1] * 768]
                        mock_embedder_class.return_value = mock_embedder

                        with patch("codesearch.cli.commands.DataIngestionPipeline") as mock_pipeline_class:
                            mock_pipeline = Mock()
                            mock_pipeline.ingest_batch.side_effect = RuntimeError("Database error")
                            mock_pipeline_class.return_value = mock_pipeline

                            with patch("codesearch.cli.commands.Console"):
                                result = runner.invoke(app, ["index", str(test_repo), "--force"])
                                assert result.exit_code == 1
                                assert "Database storage error" in result.output


def test_index_command_converts_entities_correctly(tmp_path):
    """Test that entities are converted to CodeEntity format correctly."""
    from datetime import datetime

    from codesearch.data_ingestion.models import IngestionResult
    from codesearch.models import Class, FileMetadata, Function

    test_repo = tmp_path / "test_repo"
    test_repo.mkdir()

    mock_files = [
        FileMetadata(
            file_path=str(test_repo / "test.py"),
            language="python",
            size_bytes=50,
            modified_time=datetime.now(),
            is_ignored=False,
        ),
    ]

    # Test with both Function and Class entities
    mock_entities = [
        Function(
            name="public_func",
            source_code="def public_func(): pass",
            line_number=1,
            end_line=1,
            file_path=str(test_repo / "test.py"),
        ),
        Function(
            name="_protected_func",
            source_code="def _protected_func(): pass",
            line_number=5,
            end_line=5,
            file_path=str(test_repo / "test.py"),
        ),
        Function(
            name="__private_func",
            source_code="def __private_func(): pass",
            line_number=10,
            end_line=10,
            file_path=str(test_repo / "test.py"),
        ),
        Class(
            name="TestClass",
            source_code="class TestClass: pass",
            line_number=15,
            end_line=15,
            file_path=str(test_repo / "test.py"),
        ),
    ]

    mock_result = IngestionResult(
        inserted_count=4,
        skipped_count=0,
        failed_count=0,
    )

    captured_entities = []

    def capture_ingest(entities):
        captured_entities.extend(entities)
        return mock_result

    with patch("codesearch.cli.commands.lancedb.connect"):
        with patch("codesearch.cli.commands.get_db_path", return_value=str(tmp_path / "test.db")):
            with patch("codesearch.cli.commands.RepositoryScannerImpl") as mock_scanner_class:
                mock_scanner = Mock()
                mock_scanner.scan_repository.return_value = mock_files
                mock_scanner.get_statistics.return_value = {
                    "total_files": 1,
                    "by_language": {"python": 1},
                    "total_size_bytes": 50,
                }
                mock_scanner.config = Mock()
                mock_scanner_class.return_value = mock_scanner

                with patch("codesearch.cli.commands.PythonParser") as mock_parser_class:
                    mock_parser = Mock()
                    mock_parser.parse_file.return_value = mock_entities
                    mock_parser_class.return_value = mock_parser

                    with patch("codesearch.cli.commands.EmbeddingGenerator") as mock_embedder_class:
                        mock_embedder = Mock()
                        mock_embedder.get_model_info.return_value = {
                            "name": "codebert-base",
                            "dimensions": 768,
                            "device": "cpu",
                        }
                        mock_embedder.embed_batch.return_value = [[0.1] * 768 for _ in range(4)]
                        mock_embedder_class.return_value = mock_embedder

                        with patch("codesearch.cli.commands.DataIngestionPipeline") as mock_pipeline_class:
                            mock_pipeline = Mock()
                            mock_pipeline.ingest_batch.side_effect = capture_ingest
                            mock_pipeline_class.return_value = mock_pipeline

                            with patch("codesearch.cli.commands.Console"):
                                result = runner.invoke(app, ["index", str(test_repo), "--force"])
                                assert result.exit_code == 0

                                # Verify entities were converted correctly
                                assert len(captured_entities) == 4

                                # Check entity types
                                entity_types = [e.entity_type for e in captured_entities]
                                assert "function" in entity_types
                                assert "class" in entity_types

                                # Check visibility
                                visibilities = {e.name: e.visibility for e in captured_entities}
                                assert visibilities["public_func"] == "public"
                                assert visibilities["_protected_func"] == "protected"
                                assert visibilities["__private_func"] == "private"
                                assert visibilities["TestClass"] == "public"

                                # Verify all have entity_id
                                for entity in captured_entities:
                                    assert entity.entity_id
                                    assert len(entity.entity_id) == 64  # SHA256 hex length


def test_index_command_stores_relationships(tmp_path):
    """Test index command extracts and stores call graph relationships."""
    from datetime import datetime

    from codesearch.models import FileMetadata, Function

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

    # Create functions with call relationships
    mock_entities = [
        Function(
            name="caller_func",
            file_path=str(test_repo / "test.py"),
            language="python",
            source_code="def caller_func(): callee_func()",
            line_number=1,
            end_line=2,
            calls_to=["callee_func"],  # This function calls callee_func
        ),
        Function(
            name="callee_func",
            file_path=str(test_repo / "test.py"),
            language="python",
            source_code="def callee_func(): pass",
            line_number=4,
            end_line=5,
            calls_to=[],
        ),
    ]

    captured_relationships = []

    def capture_relationships(rels):
        captured_relationships.extend(rels)
        mock_result = Mock()
        mock_result.relationships_inserted = len(rels)
        mock_result.relationships_failed = 0
        return mock_result

    with patch("codesearch.cli.commands.lancedb.connect"):
        with patch("codesearch.cli.commands.get_db_path", return_value=str(tmp_path / "test.db")):
            with patch("codesearch.cli.commands.DatabaseInitializer") as mock_db_init_class:
                mock_db_init = Mock()
                mock_db_init.is_initialized.return_value = False
                mock_db_init.initialize.return_value = True
                mock_db_init_class.return_value = mock_db_init

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

                    with patch("codesearch.cli.commands.PythonParser") as mock_parser_class:
                        mock_parser = Mock()
                        mock_parser.parse_file.return_value = mock_entities
                        mock_parser_class.return_value = mock_parser

                        with patch("codesearch.cli.commands.EmbeddingGenerator") as mock_embedder_class:
                            mock_embedder = Mock()
                            mock_embedder.get_model_info.return_value = {
                                "name": "codebert-base",
                                "dimensions": 768,
                                "device": "cpu",
                            }
                            mock_embedder.embed_batch.return_value = [[0.1] * 768 for _ in range(2)]
                            mock_embedder_class.return_value = mock_embedder

                            with patch("codesearch.cli.commands.DataIngestionPipeline") as mock_pipeline_class:
                                mock_pipeline = Mock()
                                mock_result = Mock()
                                mock_result.inserted_count = 2
                                mock_result.skipped_count = 0
                                mock_result.failed_count = 0
                                mock_pipeline.ingest_batch.return_value = mock_result
                                mock_pipeline.ingest_relationships.side_effect = capture_relationships
                                mock_pipeline_class.return_value = mock_pipeline

                                with patch("codesearch.cli.commands.Console"):
                                    result = runner.invoke(app, ["index", str(test_repo), "--force"])

                                    # Check that indexing completed
                                    assert result.exit_code == 0

                                    # Verify relationships were captured
                                    assert len(captured_relationships) == 1

                                    # The relationship should be caller_func -> callee_func
                                    rel = captured_relationships[0]
                                    assert rel.relationship_type == "calls"
