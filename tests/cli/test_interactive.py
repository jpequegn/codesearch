"""Tests for interactive CLI features."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from typer.testing import CliRunner

from codesearch.cli.main import app
from codesearch.cli.interactive import Paginator, detail, context, compare, config_show, config_init
from codesearch.query.models import SearchResult


runner = CliRunner()


class TestPaginator:
    """Tests for pagination utility."""

    def test_paginator_initialization(self):
        """Test paginator initialization."""
        items = list(range(100))
        paginator = Paginator(items, page_size=10)

        assert paginator.total_pages == 10
        assert paginator.current_page == 1
        assert paginator.page_size == 10

    def test_paginator_get_page(self):
        """Test getting a specific page."""
        items = list(range(100))
        paginator = Paginator(items, page_size=10)

        page1 = paginator.get_page(1)
        assert len(page1) == 10
        assert page1 == list(range(0, 10))

        page5 = paginator.get_page(5)
        assert len(page5) == 10
        assert page5 == list(range(40, 50))

    def test_paginator_get_page_invalid(self):
        """Test getting invalid page."""
        items = list(range(100))
        paginator = Paginator(items, page_size=10)

        assert paginator.get_page(0) == []
        assert paginator.get_page(11) == []
        assert paginator.get_page(-1) == []

    def test_paginator_next_page(self):
        """Test navigating to next page."""
        items = list(range(100))
        paginator = Paginator(items, page_size=10)

        assert paginator.current_page == 1
        page1 = paginator.next_page()
        assert paginator.current_page == 2
        assert page1 == list(range(10, 20))

    def test_paginator_prev_page(self):
        """Test navigating to previous page."""
        items = list(range(100))
        paginator = Paginator(items, page_size=10)

        paginator.current_page = 5
        page4 = paginator.prev_page()
        assert paginator.current_page == 4
        assert page4 == list(range(30, 40))

    def test_paginator_boundary(self):
        """Test paginator at boundaries."""
        items = list(range(25))
        paginator = Paginator(items, page_size=10)

        assert paginator.total_pages == 3

        # Last page
        last_page = paginator.get_page(3)
        assert len(last_page) == 5
        assert last_page == list(range(20, 25))


class TestDetailCommand:
    """Tests for detail command."""

    @patch("codesearch.cli.interactive.get_db_path")
    @patch("codesearch.cli.interactive.validate_db_exists")
    @patch("codesearch.cli.interactive.lancedb.connect")
    def test_detail_entity_not_found(self, mock_connect, mock_validate, mock_get_path):
        """Test detail command when entity not found."""
        mock_get_path.return_value = "/test/db"
        mock_validate.return_value = True

        # Mock database
        mock_client = MagicMock()
        mock_table = MagicMock()
        mock_table.search.return_value.to_list.return_value = []
        mock_client.open_table.return_value = mock_table
        mock_connect.return_value = mock_client

        result = runner.invoke(app, ["detail", "nonexistent"])
        assert result.exit_code != 0
        assert "not found" in result.stdout

    @patch("codesearch.cli.interactive.get_db_path")
    @patch("codesearch.cli.interactive.validate_db_exists")
    @patch("codesearch.cli.interactive.lancedb.connect")
    def test_detail_entity_found(self, mock_connect, mock_validate, mock_get_path):
        """Test detail command with valid entity."""
        mock_get_path.return_value = "/test/db"
        mock_validate.return_value = True

        # Mock database with entity
        mock_client = MagicMock()
        mock_table = MagicMock()
        mock_table.search.return_value.to_list.return_value = [
            {
                "name": "test_func",
                "entity_type": "function",
                "language": "python",
                "file_path": "test.py",
                "start_line": 1,
                "end_line": 5,
                "code_text": "def test_func():\n    return 42",
            }
        ]
        mock_client.open_table.return_value = mock_table
        mock_connect.return_value = mock_client

        result = runner.invoke(app, ["detail", "test_func"])
        assert result.exit_code == 0
        assert "test_func" in result.stdout
        assert "function" in result.stdout


class TestContextCommand:
    """Tests for context command."""

    @patch("codesearch.cli.interactive.get_db_path")
    @patch("codesearch.cli.interactive.validate_db_exists")
    @patch("codesearch.cli.interactive.lancedb.connect")
    def test_context_no_relationships(self, mock_connect, mock_validate, mock_get_path):
        """Test context command with no relationships."""
        mock_get_path.return_value = "/test/db"
        mock_validate.return_value = True

        # Mock database with no relationships
        mock_client = MagicMock()
        mock_table = MagicMock()
        mock_table.search.return_value.to_list.return_value = []
        mock_client.open_table.return_value = mock_table
        mock_connect.return_value = mock_client

        result = runner.invoke(app, ["context", "test_func"])
        assert "No relationships" in result.stdout

    @patch("codesearch.cli.interactive.get_db_path")
    @patch("codesearch.cli.interactive.validate_db_exists")
    @patch("codesearch.cli.interactive.lancedb.connect")
    def test_context_with_relationships(self, mock_connect, mock_validate, mock_get_path):
        """Test context command with relationships."""
        mock_get_path.return_value = "/test/db"
        mock_validate.return_value = True

        # Mock database with relationships
        mock_client = MagicMock()
        mock_table = MagicMock()
        mock_table.search.return_value.to_list.return_value = [
            {"source_id": "test_func", "target_id": "helper_func", "relationship_type": "calls"},
            {"source_id": "caller_func", "target_id": "test_func", "relationship_type": "calls"},
        ]
        mock_client.open_table.return_value = mock_table
        mock_connect.return_value = mock_client

        result = runner.invoke(app, ["context", "test_func"])
        assert result.exit_code == 0
        assert "test_func" in result.stdout


class TestCompareCommand:
    """Tests for compare command."""

    @patch("codesearch.cli.interactive.get_db_path")
    @patch("codesearch.cli.interactive.validate_db_exists")
    @patch("codesearch.cli.interactive.lancedb.connect")
    def test_compare_entity_not_found(self, mock_connect, mock_validate, mock_get_path):
        """Test compare command when entity not found."""
        mock_get_path.return_value = "/test/db"
        mock_validate.return_value = True

        # Mock database
        mock_client = MagicMock()
        mock_table = MagicMock()
        mock_table.search.return_value.to_list.return_value = []
        mock_client.open_table.return_value = mock_table
        mock_connect.return_value = mock_client

        result = runner.invoke(app, ["compare", "func1", "func2"])
        assert result.exit_code != 0
        assert "not found" in result.stdout

    @patch("codesearch.cli.interactive.get_db_path")
    @patch("codesearch.cli.interactive.validate_db_exists")
    @patch("codesearch.cli.interactive.lancedb.connect")
    def test_compare_entities(self, mock_connect, mock_validate, mock_get_path):
        """Test compare command with valid entities."""
        mock_get_path.return_value = "/test/db"
        mock_validate.return_value = True

        # Mock database with entities
        mock_client = MagicMock()
        mock_table = MagicMock()
        mock_table.search.return_value.to_list.return_value = [
            {
                "name": "func1",
                "entity_type": "function",
                "language": "python",
                "file_path": "file1.py",
                "start_line": 1,
                "end_line": 5,
            },
            {
                "name": "func2",
                "entity_type": "function",
                "language": "python",
                "file_path": "file2.py",
                "start_line": 10,
                "end_line": 15,
            },
        ]
        mock_client.open_table.return_value = mock_table
        mock_connect.return_value = mock_client

        result = runner.invoke(app, ["compare", "func1", "func2"])
        assert result.exit_code == 0
        assert "func1" in result.stdout
        assert "func2" in result.stdout


class TestConfigCommands:
    """Tests for configuration management commands."""

    @patch("codesearch.cli.interactive.get_config")
    def test_config_show(self, mock_get_config):
        """Test config-show command."""
        mock_get_config.return_value = {
            "db_path": "/test/db",
            "output": "table",
            "limit": 10,
        }

        result = runner.invoke(app, ["config-show"])
        assert result.exit_code == 0
        assert "db_path" in result.stdout
        assert "/test/db" in result.stdout

    @patch("codesearch.cli.config.init_config")
    @patch("pathlib.Path.exists")
    def test_config_init(self, mock_exists, mock_init_config):
        """Test config-init command."""
        mock_exists.return_value = False
        mock_init_config.return_value = True

        result = runner.invoke(app, ["config-init"])
        assert result.exit_code == 0
        assert "initialized" in result.stdout.lower()


class TestDetailCommandIntegration:
    """Integration tests for detail command."""

    def test_detail_help(self):
        """Test detail command help."""
        result = runner.invoke(app, ["detail", "--help"])
        assert result.exit_code == 0
        assert "Drill down" in result.stdout


class TestContextCommandIntegration:
    """Integration tests for context command."""

    def test_context_help(self):
        """Test context command help."""
        result = runner.invoke(app, ["context", "--help"])
        assert result.exit_code == 0
        assert "context" in result.stdout.lower()


class TestCompareCommandIntegration:
    """Integration tests for compare command."""

    def test_compare_help(self):
        """Test compare command help."""
        result = runner.invoke(app, ["compare", "--help"])
        assert result.exit_code == 0
        assert "compare" in result.stdout.lower()
