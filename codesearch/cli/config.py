"""Configuration management for codesearch CLI.

Supports configuration from multiple sources with the following priority:
1. Command-line arguments
2. Environment variables
3. Configuration file (~/.codesearch/config.yaml)
4. Default values
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging


logger = logging.getLogger(__name__)

# Configuration file locations to check
CONFIG_LOCATIONS = [
    Path.home() / ".codesearch" / "config.yaml",
    Path.home() / ".codesearch" / "config.json",
    Path.cwd() / ".codesearch" / "config.yaml",
    Path.cwd() / ".codesearch" / "config.json",
    Path.cwd() / "codesearch.yaml",
    Path.cwd() / "codesearch.json",
]

# Default configuration
DEFAULT_CONFIG = {
    "db_path": str(Path.home() / ".codesearch" / "db"),
    "language": None,
    "output": "table",
    "limit": 10,
    "verbose": False,
}


def get_config_file() -> Optional[Path]:
    """Find configuration file in standard locations.

    Returns:
        Path to config file if found, None otherwise
    """
    for config_path in CONFIG_LOCATIONS:
        if config_path.exists():
            logger.debug(f"Found config file at {config_path}")
            return config_path
    return None


def load_config_file(config_path: Path) -> Dict[str, Any]:
    """Load configuration from file (YAML or JSON).

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary

    Raises:
        IOError: If file cannot be read
        ValueError: If file format is invalid
    """
    if not config_path.exists():
        raise IOError(f"Configuration file not found: {config_path}")

    try:
        content = config_path.read_text()

        if config_path.suffix in [".yaml", ".yml"]:
            # Try to import yaml, fall back to JSON if not available
            try:
                import yaml
                return yaml.safe_load(content) or {}
            except ImportError:
                logger.warning("PyYAML not installed, treating as JSON")
                return json.loads(content)
        elif config_path.suffix == ".json":
            return json.loads(content)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in config file: {e}")
    except Exception as e:
        raise IOError(f"Error reading config file {config_path}: {e}")


def save_config_file(config: Dict[str, Any], config_path: Path, format: str = "json") -> bool:
    """Save configuration to file.

    Args:
        config: Configuration dictionary
        config_path: Path where to save configuration
        format: File format (json or yaml)

    Returns:
        True if successful, False otherwise
    """
    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "yaml":
            try:
                import yaml
                content = yaml.dump(config, default_flow_style=False)
            except ImportError:
                logger.warning("PyYAML not installed, saving as JSON instead")
                content = json.dumps(config, indent=2)
        else:
            content = json.dumps(config, indent=2)

        config_path.write_text(content)
        logger.info(f"Configuration saved to {config_path}")
        return True

    except Exception as e:
        logger.error(f"Error saving config file: {e}")
        return False


def get_db_path() -> str:
    """Get database path from env var, config file, or default location.

    Priority:
    1. CODESEARCH_DB environment variable
    2. db_path from config file
    3. ~/.codesearch/db (default)

    Returns:
        Path to LanceDB database
    """
    # Check environment variable first
    if "CODESEARCH_DB" in os.environ:
        return os.environ["CODESEARCH_DB"]

    # Check config file
    config_file = get_config_file()
    if config_file:
        try:
            config = load_config_file(config_file)
            if "db_path" in config:
                return config["db_path"]
        except Exception as e:
            logger.debug(f"Could not load db_path from config: {e}")

    # Return default
    default_path = Path.home() / ".codesearch" / "db"
    return str(default_path)


def validate_db_exists(db_path: str) -> bool:
    """Check if database exists at path.

    Args:
        db_path: Path to database

    Returns:
        True if database exists, False otherwise
    """
    return os.path.exists(db_path)


def get_config(override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Get CLI configuration from all sources.

    Configuration priority (highest to lowest):
    1. Override parameters
    2. Environment variables
    3. Configuration file
    4. Default values

    Args:
        override: Configuration overrides (typically from CLI args)

    Returns:
        Complete configuration dictionary
    """
    # Start with defaults
    config = DEFAULT_CONFIG.copy()

    # Layer in config file if it exists
    config_file = get_config_file()
    if config_file:
        try:
            file_config = load_config_file(config_file)
            config.update(file_config)
            logger.debug(f"Loaded config from {config_file}")
        except Exception as e:
            logger.warning(f"Could not load config file: {e}")

    # Layer in environment variables
    if "CODESEARCH_DB" in os.environ:
        config["db_path"] = os.environ["CODESEARCH_DB"]
    if "CODESEARCH_LANGUAGE" in os.environ:
        config["language"] = os.environ["CODESEARCH_LANGUAGE"]
    if "CODESEARCH_OUTPUT" in os.environ:
        config["output"] = os.environ["CODESEARCH_OUTPUT"]
    if "CODESEARCH_LIMIT" in os.environ:
        try:
            config["limit"] = int(os.environ["CODESEARCH_LIMIT"])
        except ValueError:
            logger.warning("Invalid CODESEARCH_LIMIT environment variable")
    if "CODESEARCH_VERBOSE" in os.environ:
        config["verbose"] = os.environ["CODESEARCH_VERBOSE"].lower() in ["true", "1", "yes"]

    # Layer in overrides (highest priority)
    if override:
        config.update(override)

    return config


def init_config(config_path: Optional[Path] = None) -> bool:
    """Initialize a new configuration file.

    Args:
        config_path: Path where to create config file (default: ~/.codesearch/config.json)

    Returns:
        True if successful, False otherwise
    """
    if config_path is None:
        config_path = Path.home() / ".codesearch" / "config.json"

    return save_config_file(DEFAULT_CONFIG, config_path)
