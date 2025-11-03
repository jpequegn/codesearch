"""Tests for CLI main app."""

import pytest
from typer.testing import CliRunner
from codesearch.cli.main import app

runner = CliRunner()


def test_cli_help():
    """Test that CLI help works."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Semantic code search tool" in result.stdout
