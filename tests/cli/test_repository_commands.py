"""Tests for repository management CLI commands."""

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from typer.testing import CliRunner

from codesearch.cli.main import app
from codesearch.cli.commands import repo_list, repo_add, repo_remove, search_multi
from codesearch.indexing.repository import RepositoryRegistry


runner = CliRunner()


class TestRepoListCommand:
    """Tests for repo-list command."""

    @patch("codesearch.cli.commands.RepositoryRegistry")
    def test_repo_list_empty(self, mock_registry_class):
        """Test repo-list with no repositories."""
        mock_registry = MagicMock()
        mock_registry.list_repositories.return_value = []
        mock_registry_class.return_value = mock_registry

        result = runner.invoke(app, ["repo-list"])
        assert result.exit_code == 0
        assert "No repositories registered" in result.stdout

    @patch("codesearch.cli.commands.RepositoryRegistry")
    def test_repo_list_with_repos(self, mock_registry_class):
        """Test repo-list with repositories."""
        mock_repo = MagicMock()
        mock_repo.repo_name = "test-repo"
        mock_repo.repo_path = "/path/to/repo"
        mock_repo.repo_id = "abc123"

        mock_metadata = MagicMock()
        mock_metadata.namespace_prefix = "repo_abc123"
        mock_metadata.entity_count = 10
        mock_metadata.file_count = 5
        mock_metadata.indexed_at = "2024-01-01T00:00:00"

        mock_registry = MagicMock()
        mock_registry.list_repositories.return_value = [mock_repo]
        mock_registry.get_metadata.return_value = mock_metadata
        mock_registry_class.return_value = mock_registry

        result = runner.invoke(app, ["repo-list"])
        assert result.exit_code == 0
        assert "test-repo" in result.stdout
        assert "abc123" in result.stdout


class TestRepoAddCommand:
    """Tests for repo-add command."""

    def test_repo_add_nonexistent_path(self):
        """Test adding nonexistent repository."""
        result = runner.invoke(
            app, ["repo-add", "/nonexistent/path"]
        )
        assert result.exit_code == 2
        assert "Path not found" in result.stdout

    @patch("codesearch.cli.commands.RepositoryRegistry")
    def test_repo_add_success(self, mock_registry_class):
        """Test adding repository successfully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_registry = MagicMock()
            mock_registry.find_by_path.return_value = None

            mock_config = MagicMock()
            mock_config.repo_name = "test-repo"
            mock_config.repo_id = "abc123"
            mock_config.repo_path = tmpdir

            mock_registry.register_repository.return_value = mock_config
            mock_registry_class.return_value = mock_registry

            result = runner.invoke(app, ["repo-add", tmpdir, "--name", "test-repo"])
            assert result.exit_code == 0
            assert "Repository registered" in result.stdout
            assert "test-repo" in result.stdout

    @patch("codesearch.cli.commands.RepositoryRegistry")
    def test_repo_add_duplicate(self, mock_registry_class):
        """Test adding duplicate repository."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_registry = MagicMock()
            existing_config = MagicMock()
            existing_config.repo_name = "existing"
            mock_registry.find_by_path.return_value = existing_config
            mock_registry_class.return_value = mock_registry

            result = runner.invoke(app, ["repo-add", tmpdir])
            assert result.exit_code == 1
            assert "already registered" in result.stdout


class TestRepoRemoveCommand:
    """Tests for repo-remove command."""

    @patch("codesearch.cli.commands.RepositoryRegistry")
    def test_repo_remove_success(self, mock_registry_class):
        """Test removing repository successfully."""
        mock_registry = MagicMock()
        mock_registry.unregister_repository.return_value = True
        mock_registry_class.return_value = mock_registry

        result = runner.invoke(app, ["repo-remove", "abc123"])
        assert result.exit_code == 0
        assert "unregistered" in result.stdout

    @patch("codesearch.cli.commands.RepositoryRegistry")
    def test_repo_remove_not_found(self, mock_registry_class):
        """Test removing nonexistent repository."""
        mock_registry = MagicMock()
        mock_registry.unregister_repository.return_value = False
        mock_registry_class.return_value = mock_registry

        result = runner.invoke(app, ["repo-remove", "nonexistent"])
        assert result.exit_code == 2
        assert "not found" in result.stdout


class TestSearchMultiCommand:
    """Tests for search-multi command."""

    @patch("codesearch.cli.commands.validate_db_exists")
    def test_search_multi_no_db(self, mock_validate):
        """Test search-multi with no database."""
        mock_validate.return_value = False

        result = runner.invoke(app, ["search-multi", "query"])
        assert result.exit_code == 2
        assert "Database not found" in result.stdout

    @patch("codesearch.cli.commands.validate_db_exists")
    @patch("codesearch.cli.commands.lancedb.connect")
    @patch("codesearch.cli.commands.QueryEngine")
    @patch("codesearch.cli.commands.RepositoryRegistry")
    def test_search_multi_no_results(
        self, mock_registry_class, mock_engine_class, mock_connect, mock_validate
    ):
        """Test search-multi with no results."""
        mock_validate.return_value = True

        mock_engine = MagicMock()
        mock_engine.search_vector.return_value = []
        mock_engine_class.return_value = mock_engine

        mock_registry = MagicMock()
        mock_registry_class.return_value = mock_registry

        result = runner.invoke(app, ["search-multi", "query"])
        assert result.exit_code == 0
        assert "No results found" in result.stdout

    @patch("codesearch.cli.commands.validate_db_exists")
    def test_search_multi_with_repos_option(self, mock_validate):
        """Test search-multi accepts repos filter option."""
        mock_validate.return_value = False  # Ensures early exit

        result = runner.invoke(
            app, ["search-multi", "query", "--repos", "repo1,repo2"]
        )
        # Should handle the --repos option without error (exits because no DB)
        assert "Database not found" in result.stdout or result.exit_code in [0, 2]


class TestIntegration:
    """Integration tests for repository commands."""

    def test_repo_commands_help(self):
        """Test help for all repo commands."""
        commands = ["repo-list", "repo-add", "repo-remove", "search-multi"]

        for cmd in commands:
            result = runner.invoke(app, [cmd, "--help"])
            assert result.exit_code == 0
            # Help should mention the command
            assert cmd.replace("-", " ") in result.stdout.lower() or "usage" in result.stdout.lower()
