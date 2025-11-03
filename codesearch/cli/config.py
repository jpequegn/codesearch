"""Configuration management for codesearch CLI."""

import os
from pathlib import Path


def get_db_path() -> str:
    """Get database path from env var or default location.

    Priority:
    1. CODESEARCH_DB environment variable
    2. ~/.codesearch/db (default)

    Returns:
        Path to LanceDB database
    """
    if "CODESEARCH_DB" in os.environ:
        return os.environ["CODESEARCH_DB"]

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


def get_config() -> dict:
    """Get CLI configuration from environment.

    Returns:
        Dictionary with configuration values
    """
    return {
        "db_path": get_db_path(),
        "language": os.environ.get("CODESEARCH_LANGUAGE"),
        "output": os.environ.get("CODESEARCH_OUTPUT", "table"),
    }
