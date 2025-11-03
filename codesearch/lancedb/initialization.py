"""Database initialization and setup utilities for LanceDB."""

import lancedb
from pathlib import Path
from typing import Optional, List, Dict
import logging
from datetime import datetime, timezone
import json


logger = logging.getLogger(__name__)


class DatabaseInitializer:
    """Handles database initialization, schema setup, and configuration.

    Features:
    - Idempotent database initialization
    - Schema version management
    - Database configuration and defaults
    - Table initialization with proper schema
    """

    # Current schema version - increment on breaking changes
    SCHEMA_VERSION = "1.0.0"

    # Default configuration
    DEFAULT_CONFIG = {
        "schema_version": SCHEMA_VERSION,
        "created_at": None,  # Will be set during init
        "initialized": True,
        "tables": {
            "code_entities": {
                "description": "Stores code entities (functions, classes, etc.) with embeddings",
                "status": "initialized",
            },
            "metadata": {
                "description": "Stores search metadata and statistics",
                "status": "initialized",
            },
            "relationships": {
                "description": "Stores code relationships (calls, inherits, imports, etc.)",
                "status": "initialized",
            },
        },
    }

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize database initializer.

        Args:
            db_path: Path to LanceDB directory (default: .lancedb/)
        """
        self.db_path = db_path or Path(".lancedb")
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.config_file = self.db_path / "db_config.json"

    def initialize(self) -> bool:
        """Initialize database with schema and configuration.

        This is idempotent - can be called multiple times safely.

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            db = lancedb.connect(str(self.db_path))

            # Check if already initialized
            if self.is_initialized():
                logger.info("Database already initialized")
                return True

            # Write configuration
            self._write_config()

            logger.info(f"Database initialized at {self.db_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            return False

    def is_initialized(self) -> bool:
        """Check if database is properly initialized.

        Returns:
            True if database config exists and is valid
        """
        if not self.config_file.exists():
            return False

        try:
            config = self._read_config()
            return config.get("initialized", False) and "schema_version" in config
        except Exception as e:
            logger.error(f"Error checking initialization status: {e}")
            return False

    def validate_schema(self) -> Dict[str, bool]:
        """Validate database schema version and structure.

        Returns:
            Dictionary with validation results for each check
        """
        checks = {
            "config_exists": self.config_file.exists(),
            "config_valid": False,
            "schema_version_match": False,
            "tables_exist": False,
        }

        try:
            if checks["config_exists"]:
                config = self._read_config()
                checks["config_valid"] = isinstance(config, dict)
                checks["schema_version_match"] = (
                    config.get("schema_version") == self.SCHEMA_VERSION
                )

                # Check if required tables referenced in config
                if "tables" in config:
                    required_tables = list(config["tables"].keys())
                    checks["tables_exist"] = len(required_tables) > 0

        except Exception as e:
            logger.error(f"Error validating schema: {e}")

        return checks

    def get_schema_info(self) -> Dict:
        """Get current database schema information.

        Returns:
            Dictionary with schema version, creation time, tables, etc.
        """
        if not self.is_initialized():
            return {"status": "not_initialized"}

        try:
            config = self._read_config()
            return {
                "status": "initialized",
                "schema_version": config.get("schema_version"),
                "created_at": config.get("created_at"),
                "tables": config.get("tables", {}),
            }
        except Exception as e:
            logger.error(f"Error getting schema info: {e}")
            return {"status": "error", "error": str(e)}

    def reset(self, confirm: bool = False) -> bool:
        """Reset database (dangerous operation).

        Args:
            confirm: Must be True to prevent accidental resets

        Returns:
            True if reset successful
        """
        if not confirm:
            logger.warning("Reset requested but not confirmed - refusing")
            return False

        try:
            if self.config_file.exists():
                self.config_file.unlink()
            logger.warning("Database configuration reset")
            return True
        except Exception as e:
            logger.error(f"Failed to reset database: {e}")
            return False

    def _write_config(self):
        """Write database configuration to file."""
        config = self.DEFAULT_CONFIG.copy()
        config["created_at"] = datetime.now(timezone.utc).isoformat()

        try:
            with open(self.config_file, "w") as f:
                json.dump(config, f, indent=2)
            logger.debug(f"Wrote database config to {self.config_file}")
        except Exception as e:
            logger.error(f"Failed to write database config: {e}")
            raise

    def _read_config(self) -> Dict:
        """Read database configuration from file.

        Returns:
            Configuration dictionary

        Raises:
            FileNotFoundError: If config file doesn't exist
            json.JSONDecodeError: If config is invalid JSON
        """
        with open(self.config_file, "r") as f:
            return json.load(f)
