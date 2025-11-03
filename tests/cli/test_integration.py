"""Integration tests for CLI."""

import os
import pytest
from typer.testing import CliRunner
from codesearch.cli.main import app

runner = CliRunner()


def test_cli_shows_help():
    """Test that CLI help displays all commands."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "pattern" in result.stdout.lower()
    assert "find-similar" in result.stdout or "find_similar" in result.stdout
    assert "dependencies" in result.stdout.lower()
    assert "index" in result.stdout.lower()


def test_pattern_requires_query():
    """Test that pattern command requires query argument."""
    result = runner.invoke(app, ["pattern"])
    assert result.exit_code != 0
    assert "required" in result.stdout.lower() or "missing" in result.stdout.lower()


def test_find_similar_requires_entity():
    """Test that find-similar command requires entity argument."""
    result = runner.invoke(app, ["find-similar"])
    assert result.exit_code != 0


def test_dependencies_requires_entity():
    """Test that dependencies command requires entity argument."""
    result = runner.invoke(app, ["dependencies"])
    assert result.exit_code != 0


def test_index_requires_path():
    """Test that index command requires path argument."""
    result = runner.invoke(app, ["index"])
    assert result.exit_code != 0


def test_all_commands_have_help():
    """Test that all commands show help."""
    commands = ["pattern", "find-similar", "dependencies", "index", "refactor-dupes"]
    for cmd in commands:
        result = runner.invoke(app, [cmd, "--help"])
        assert result.exit_code == 0, f"{cmd} --help failed"
        assert cmd in result.stdout.lower() or cmd.replace("-", "_") in result.stdout.lower()


def test_pattern_help_shows_options():
    """Test that pattern help shows limit and output options."""
    result = runner.invoke(app, ["pattern", "--help"])
    assert result.exit_code == 0
    assert "--limit" in result.stdout or "-l" in result.stdout
    assert "--output" in result.stdout or "-o" in result.stdout


def test_find_similar_help_shows_options():
    """Test that find-similar help shows limit and language options."""
    result = runner.invoke(app, ["find-similar", "--help"])
    assert result.exit_code == 0
    assert "--limit" in result.stdout or "-l" in result.stdout
    assert "--language" in result.stdout or "-L" in result.stdout


def test_dependencies_help_shows_direction():
    """Test that dependencies help shows direction option."""
    result = runner.invoke(app, ["dependencies", "--help"])
    assert result.exit_code == 0
    assert "--direction" in result.stdout or "-d" in result.stdout


def test_index_help_shows_options():
    """Test that index help shows force and language options."""
    result = runner.invoke(app, ["index", "--help"])
    assert result.exit_code == 0
    assert "--force" in result.stdout or "-f" in result.stdout
    assert "--language" in result.stdout or "-L" in result.stdout


def test_refactor_dupes_help_shows_threshold():
    """Test that refactor-dupes help shows threshold option."""
    result = runner.invoke(app, ["refactor-dupes", "--help"])
    assert result.exit_code == 0
    assert "--threshold" in result.stdout or "-t" in result.stdout
