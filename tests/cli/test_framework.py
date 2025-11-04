"""Tests for CLI framework components.

Tests cover:
- Version support
- Configuration loading (files, env vars, defaults)
- Configuration file I/O
- Command routing
- Help documentation
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch

from codesearch.cli.config import (
    get_config,
    get_db_path,
    load_config_file,
    save_config_file,
    init_config,
    get_config_file,
    DEFAULT_CONFIG,
)
from codesearch import __version__


class TestVersion:
    """Tests for version information."""

    def test_version_exists(self):
        """Test that version is defined."""
        assert __version__ is not None
        assert isinstance(__version__, str)
        assert len(__version__) > 0

    def test_version_format(self):
        """Test that version follows semantic versioning."""
        parts = __version__.split(".")
        assert len(parts) >= 2
        for part in parts[:2]:
            assert part.isdigit()


class TestConfigFile:
    """Tests for configuration file I/O."""

    def test_save_and_load_json_config(self):
        """Test saving and loading JSON configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            test_config = {
                "db_path": "/test/db",
                "output": "json",
                "limit": 20,
            }

            # Save config
            result = save_config_file(test_config, config_path, format="json")
            assert result is True
            assert config_path.exists()

            # Load config
            loaded_config = load_config_file(config_path)
            assert loaded_config == test_config

    def test_save_and_load_yaml_config(self):
        """Test saving and loading YAML configuration."""
        pytest.importorskip("yaml")  # Skip if PyYAML not installed

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            test_config = {
                "db_path": "/test/db",
                "language": "python",
            }

            # Save config
            result = save_config_file(test_config, config_path, format="yaml")
            assert result is True
            assert config_path.exists()

            # Load config
            loaded_config = load_config_file(config_path)
            assert loaded_config == test_config

    def test_load_nonexistent_config(self):
        """Test loading a nonexistent configuration file."""
        config_path = Path("/nonexistent/config.json")

        with pytest.raises(IOError):
            load_config_file(config_path)

    def test_load_invalid_json_config(self):
        """Test loading invalid JSON configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text("invalid json {")

            with pytest.raises(ValueError):
                load_config_file(config_path)

    def test_init_config(self):
        """Test initializing a new configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"

            result = init_config(config_path)
            assert result is True
            assert config_path.exists()

            loaded_config = load_config_file(config_path)
            assert "db_path" in loaded_config
            assert "output" in loaded_config


class TestConfigLoading:
    """Tests for configuration loading from various sources."""

    def test_default_config(self):
        """Test loading default configuration."""
        config = get_config()

        assert config["db_path"] is not None
        assert config["output"] == "table"
        assert config["limit"] == 10
        assert config["verbose"] is False

    def test_config_from_file(self):
        """Test loading configuration from file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / ".codesearch" / "config.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)

            test_config = {
                "output": "json",
                "limit": 25,
            }
            save_config_file(test_config, config_path)

            # Mock get_config_file to return our test config
            with patch("codesearch.cli.config.get_config_file", return_value=config_path):
                config = get_config()

                assert config["output"] == "json"
                assert config["limit"] == 25

    def test_config_override(self):
        """Test configuration override."""
        override = {
            "output": "json",
            "limit": 50,
        }

        config = get_config(override=override)

        assert config["output"] == "json"
        assert config["limit"] == 50
        assert config["db_path"] is not None  # Other defaults still present

    @patch.dict("os.environ", {"CODESEARCH_OUTPUT": "json", "CODESEARCH_LIMIT": "15"})
    def test_config_from_environment(self):
        """Test loading configuration from environment variables."""
        with patch("codesearch.cli.config.get_config_file", return_value=None):
            config = get_config()

            assert config["output"] == "json"
            assert config["limit"] == 15

    def test_config_priority(self):
        """Test configuration priority (override > env > file > default)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            file_config = {"output": "table", "limit": 5}
            save_config_file(file_config, config_path)

            # Environment variable
            with patch.dict("os.environ", {"CODESEARCH_OUTPUT": "json"}):
                with patch("codesearch.cli.config.get_config_file", return_value=config_path):
                    # Override has highest priority
                    config = get_config(override={"limit": 30})

                    # Override takes precedence
                    assert config["limit"] == 30
                    # Environment takes precedence over file
                    assert config["output"] == "json"


class TestDbPath:
    """Tests for database path resolution."""

    @patch.dict("os.environ", {}, clear=True)
    def test_db_path_default(self):
        """Test default database path."""
        with patch("codesearch.cli.config.get_config_file", return_value=None):
            db_path = get_db_path()

            assert db_path is not None
            assert ".codesearch" in db_path

    @patch.dict("os.environ", {"CODESEARCH_DB": "/custom/db/path"})
    def test_db_path_from_env(self):
        """Test database path from environment variable."""
        db_path = get_db_path()

        assert db_path == "/custom/db/path"

    def test_db_path_from_config_file(self):
        """Test database path from configuration file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            file_config = {"db_path": "/config/db/path"}
            save_config_file(file_config, config_path)

            with patch.dict("os.environ", {}, clear=True):
                with patch("codesearch.cli.config.get_config_file", return_value=config_path):
                    db_path = get_db_path()

                    assert db_path == "/config/db/path"


class TestConfigFileDiscovery:
    """Tests for configuration file discovery."""

    def test_find_config_file(self):
        """Test finding configuration file in standard locations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create config in home directory
            config_dir = Path(tmpdir) / ".codesearch"
            config_dir.mkdir()
            config_path = config_dir / "config.json"
            config_path.write_text("{}")

            # Patch CONFIG_LOCATIONS directly to point to our test config
            from codesearch.cli import config as config_module

            original_locations = config_module.CONFIG_LOCATIONS
            try:
                # Set CONFIG_LOCATIONS to only look at our test path
                config_module.CONFIG_LOCATIONS = [config_path]
                found_config = get_config_file()

                assert found_config is not None
                assert found_config == config_path
            finally:
                config_module.CONFIG_LOCATIONS = original_locations

    def test_config_file_not_found(self):
        """Test when no configuration file exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("pathlib.Path.home", return_value=Path(tmpdir)):
                found_config = get_config_file()

                assert found_config is None


class TestEnvironmentVariables:
    """Tests for environment variable configuration."""

    @patch.dict("os.environ", {"CODESEARCH_VERBOSE": "true"})
    def test_verbose_true(self):
        """Test verbose environment variable set to true."""
        with patch("codesearch.cli.config.get_config_file", return_value=None):
            config = get_config()

            assert config["verbose"] is True

    @patch.dict("os.environ", {"CODESEARCH_VERBOSE": "false"})
    def test_verbose_false(self):
        """Test verbose environment variable set to false."""
        with patch("codesearch.cli.config.get_config_file", return_value=None):
            config = get_config()

            assert config["verbose"] is False

    @patch.dict("os.environ", {"CODESEARCH_VERBOSE": "1"})
    def test_verbose_one(self):
        """Test verbose environment variable set to 1."""
        with patch("codesearch.cli.config.get_config_file", return_value=None):
            config = get_config()

            assert config["verbose"] is True

    @patch.dict("os.environ", {"CODESEARCH_LIMIT": "invalid"})
    def test_invalid_limit_env(self):
        """Test invalid limit environment variable."""
        with patch("codesearch.cli.config.get_config_file", return_value=None):
            config = get_config()

            # Should use default when invalid
            assert config["limit"] == 10


class TestConfigIntegration:
    """Integration tests for configuration system."""

    def test_full_config_stack(self):
        """Test complete configuration stack."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create config file
            config_path = Path(tmpdir) / "config.json"
            file_config = {
                "output": "table",
                "limit": 5,
                "language": "python",
            }
            save_config_file(file_config, config_path)

            # Set environment variables
            with patch.dict(
                "os.environ",
                {"CODESEARCH_OUTPUT": "json", "CODESEARCH_LIMIT": "20"},
            ):
                # Apply override
                with patch("codesearch.cli.config.get_config_file", return_value=config_path):
                    config = get_config(override={"limit": 30})

                    # Override wins
                    assert config["limit"] == 30
                    # Environment overrides file
                    assert config["output"] == "json"
                    # File value used when no override/env
                    assert config["language"] == "python"
