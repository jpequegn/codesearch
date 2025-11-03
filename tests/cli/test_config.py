"""Tests for configuration management."""

import os
import tempfile
import pytest
from codesearch.cli.config import get_db_path, validate_db_exists, get_config


def test_get_db_path_from_env_var():
    """Test getting database path from environment variable."""
    os.environ["CODESEARCH_DB"] = "/tmp/test.db"
    assert get_db_path() == "/tmp/test.db"
    del os.environ["CODESEARCH_DB"]


def test_get_db_path_default():
    """Test default database path."""
    if "CODESEARCH_DB" in os.environ:
        del os.environ["CODESEARCH_DB"]
    path = get_db_path()
    assert path.endswith("/.codesearch/db") or path.endswith("\\.codesearch\\db")


def test_validate_db_exists_true():
    """Test validation for existing database."""
    with tempfile.NamedTemporaryFile() as tmp:
        assert validate_db_exists(tmp.name) is True


def test_validate_db_exists_false():
    """Test validation for non-existent database."""
    assert validate_db_exists("/nonexistent/path/to/db") is False


def test_get_config_defaults():
    """Test getting configuration with defaults."""
    # Clear environment
    for key in ["CODESEARCH_DB", "CODESEARCH_LANGUAGE", "CODESEARCH_OUTPUT"]:
        if key in os.environ:
            del os.environ[key]

    config = get_config()
    assert "db_path" in config
    assert config["language"] is None
    assert config["output"] == "table"


def test_get_config_from_env():
    """Test getting configuration from environment variables."""
    os.environ["CODESEARCH_DB"] = "/custom/db"
    os.environ["CODESEARCH_LANGUAGE"] = "python"
    os.environ["CODESEARCH_OUTPUT"] = "json"

    config = get_config()
    assert config["db_path"] == "/custom/db"
    assert config["language"] == "python"
    assert config["output"] == "json"

    # Cleanup
    for key in ["CODESEARCH_DB", "CODESEARCH_LANGUAGE", "CODESEARCH_OUTPUT"]:
        del os.environ[key]
