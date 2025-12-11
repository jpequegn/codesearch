"""Database initialization and setup utilities for LanceDB."""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import lancedb

from codesearch.lancedb.models import (
    DEFAULT_EMBEDDING_DIMENSION,
    get_code_entities_schema,
    get_code_relationships_schema,
    get_search_metadata_schema,
)

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingModelInfo:
    """Information about the embedding model used for indexing."""

    name: str
    model_path: str
    dimensions: int
    indexed_at: str
    entity_count: int = 0

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "model_path": self.model_path,
            "dimensions": self.dimensions,
            "indexed_at": self.indexed_at,
            "entity_count": self.entity_count,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "EmbeddingModelInfo":
        """Create from dictionary."""
        return cls(
            name=data.get("name", "unknown"),
            model_path=data.get("model_path", "unknown"),
            dimensions=data.get("dimensions", DEFAULT_EMBEDDING_DIMENSION),
            indexed_at=data.get("indexed_at", ""),
            entity_count=data.get("entity_count", 0),
        )


class ModelMismatchError(Exception):
    """Raised when embedding model doesn't match indexed model."""

    def __init__(
        self,
        indexed_model: str,
        indexed_dims: int,
        requested_model: str,
        requested_dims: int,
    ):
        self.indexed_model = indexed_model
        self.indexed_dims = indexed_dims
        self.requested_model = requested_model
        self.requested_dims = requested_dims
        super().__init__(
            f"Model mismatch: database indexed with '{indexed_model}' ({indexed_dims}d), "
            f"but requested '{requested_model}' ({requested_dims}d)"
        )


class DimensionMismatchError(Exception):
    """Raised when query vector dimensions don't match indexed dimensions."""

    def __init__(self, query_dims: int, expected_dims: int):
        self.query_dims = query_dims
        self.expected_dims = expected_dims
        super().__init__(
            f"Dimension mismatch: query vector has {query_dims} dimensions, "
            f"but database expects {expected_dims}. Re-index with matching model."
        )


# Table names as constants
TABLE_CODE_ENTITIES = "code_entities"
TABLE_CODE_RELATIONSHIPS = "code_relationships"
TABLE_SEARCH_METADATA = "search_metadata"


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

    def __init__(
        self,
        db_path: Optional[Path] = None,
        embedding_dimensions: int = DEFAULT_EMBEDDING_DIMENSION,
    ) -> None:
        """Initialize database initializer.

        Args:
            db_path: Path to LanceDB directory (default: .lancedb/)
            embedding_dimensions: Vector dimensions for embeddings (default: 768).
                                 Use 256 for CodeT5+-110M, 768 for most models,
                                 1024 for CodeT5+-770M.
        """
        self.db_path = db_path or Path(".lancedb")
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.config_file = self.db_path / "db_config.json"
        self.embedding_dimensions = embedding_dimensions

    def initialize(self) -> bool:
        """Initialize database with schema and configuration.

        This is idempotent - can be called multiple times safely.
        Creates all required tables if they don't exist.

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            db = lancedb.connect(str(self.db_path))

            # Check if already initialized
            if self.is_initialized():
                logger.info("Database already initialized")
                # Still ensure tables exist
                self._ensure_tables_exist(db)
                return True

            # Create tables
            self._create_tables(db)

            # Write configuration
            self._write_config()

            logger.info(f"Database initialized at {self.db_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            return False

    def _create_tables(self, db: lancedb.DBConnection) -> None:
        """Create all required tables with proper schemas.

        Args:
            db: LanceDB connection
        """
        existing_tables = set(db.table_names())

        # Create code_entities table with configured embedding dimensions
        if TABLE_CODE_ENTITIES not in existing_tables:
            schema = get_code_entities_schema(dimensions=self.embedding_dimensions)
            db.create_table(TABLE_CODE_ENTITIES, schema=schema)
            logger.info(
                f"Created table: {TABLE_CODE_ENTITIES} "
                f"(embedding_dim={self.embedding_dimensions})"
            )

        # Create code_relationships table
        if TABLE_CODE_RELATIONSHIPS not in existing_tables:
            schema = get_code_relationships_schema()
            db.create_table(TABLE_CODE_RELATIONSHIPS, schema=schema)
            logger.info(f"Created table: {TABLE_CODE_RELATIONSHIPS}")

        # Create search_metadata table
        if TABLE_SEARCH_METADATA not in existing_tables:
            schema = get_search_metadata_schema()
            db.create_table(TABLE_SEARCH_METADATA, schema=schema)
            logger.info(f"Created table: {TABLE_SEARCH_METADATA}")

    def _ensure_tables_exist(self, db: lancedb.DBConnection) -> None:
        """Ensure all required tables exist, creating missing ones.

        Args:
            db: LanceDB connection
        """
        self._create_tables(db)

    def get_table(self, table_name: str) -> Optional[lancedb.table.Table]:
        """Get a table by name, initializing if needed.

        Args:
            table_name: Name of the table to get

        Returns:
            LanceDB table or None if not found
        """
        try:
            db = lancedb.connect(str(self.db_path))
            self._ensure_tables_exist(db)
            if table_name in db.table_names():
                return db.open_table(table_name)
            return None
        except Exception as e:
            logger.error(f"Error getting table {table_name}: {e}")
            return None

    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists.

        Args:
            table_name: Name of the table to check

        Returns:
            True if table exists
        """
        try:
            db = lancedb.connect(str(self.db_path))
            return table_name in db.table_names()
        except Exception as e:
            logger.error(f"Error checking table existence: {e}")
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
            "code_entities_exists": False,
            "code_relationships_exists": False,
            "search_metadata_exists": False,
        }

        try:
            if checks["config_exists"]:
                config = self._read_config()
                checks["config_valid"] = isinstance(config, dict)
                checks["schema_version_match"] = (
                    config.get("schema_version") == self.SCHEMA_VERSION
                )

            # Check if actual tables exist in database
            db = lancedb.connect(str(self.db_path))
            existing_tables = set(db.table_names())

            checks["code_entities_exists"] = TABLE_CODE_ENTITIES in existing_tables
            checks["code_relationships_exists"] = TABLE_CODE_RELATIONSHIPS in existing_tables
            checks["search_metadata_exists"] = TABLE_SEARCH_METADATA in existing_tables

            # All required tables exist
            checks["tables_exist"] = all([
                checks["code_entities_exists"],
                checks["code_relationships_exists"],
                checks["search_metadata_exists"],
            ])

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

    def _write_config(self) -> None:
        """Write database configuration to file."""
        config = self.DEFAULT_CONFIG.copy()
        config["created_at"] = datetime.now(timezone.utc).isoformat()
        config["embedding_dimensions"] = self.embedding_dimensions

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

    def get_embedding_dimensions(self) -> int:
        """Get the embedding dimensions configured for this database.

        Returns:
            Configured embedding dimensions, or default if not set.
        """
        if not self.is_initialized():
            return self.embedding_dimensions

        try:
            config = self._read_config()
            return config.get("embedding_dimensions", DEFAULT_EMBEDDING_DIMENSION)
        except Exception:
            return DEFAULT_EMBEDDING_DIMENSION

    def set_embedding_model(
        self,
        model_name: str,
        model_path: str,
        dimensions: int,
        entity_count: int = 0,
    ) -> None:
        """Store embedding model metadata in database config.

        Args:
            model_name: Short name of the model (e.g., "unixcoder")
            model_path: HuggingFace model path
            dimensions: Vector dimensions
            entity_count: Number of entities indexed
        """
        try:
            config = self._read_config() if self.config_file.exists() else {}
        except Exception:
            config = {}

        config["embedding_model"] = {
            "name": model_name,
            "model_path": model_path,
            "dimensions": dimensions,
            "indexed_at": datetime.now(timezone.utc).isoformat(),
            "entity_count": entity_count,
        }

        # Also update top-level embedding_dimensions for compatibility
        config["embedding_dimensions"] = dimensions

        with open(self.config_file, "w") as f:
            json.dump(config, f, indent=2)

        logger.info(
            f"Stored embedding model metadata: {model_name} ({dimensions}d, {entity_count} entities)"
        )

    def get_embedding_model(self) -> Optional[EmbeddingModelInfo]:
        """Get embedding model metadata from database config.

        Returns:
            EmbeddingModelInfo if available, None otherwise.
        """
        if not self.config_file.exists():
            return None

        try:
            config = self._read_config()
            model_data = config.get("embedding_model")
            if model_data:
                return EmbeddingModelInfo.from_dict(model_data)
            return None
        except Exception as e:
            logger.warning(f"Failed to read embedding model metadata: {e}")
            return None

    def check_model_compatibility(
        self,
        requested_model: str,
        requested_dims: int,
    ) -> tuple[bool, Optional[str]]:
        """Check if requested model is compatible with indexed model.

        Args:
            requested_model: Model name being requested
            requested_dims: Dimensions of requested model

        Returns:
            Tuple of (is_compatible, warning_message).
            - (True, None) if compatible or no indexed model
            - (False, message) if incompatible
        """
        indexed_model = self.get_embedding_model()

        # No indexed model yet - any model is compatible
        if indexed_model is None:
            return True, None

        # Same model - compatible
        if indexed_model.name == requested_model:
            return True, None

        # Different model - dimension mismatch is critical
        if indexed_model.dimensions != requested_dims:
            msg = (
                f"⚠️  Model mismatch detected!\n\n"
                f"   Database indexed with: {indexed_model.name} ({indexed_model.dimensions} dims)\n"
                f"   Requested model:       {requested_model} ({requested_dims} dims)\n\n"
                f"   Options:\n"
                f"   1. Re-index with new model: codesearch index . --force --model {requested_model}\n"
                f"   2. Search with original model: codesearch search \"query\" --model {indexed_model.name}\n"
                f"   3. Keep existing index (not recommended for model changes)"
            )
            return False, msg

        # Same dimensions but different model - warn but allow
        msg = (
            f"⚠️  Model mismatch (same dimensions)!\n\n"
            f"   Database indexed with: {indexed_model.name} ({indexed_model.dimensions} dims)\n"
            f"   Requested model:       {requested_model} ({requested_dims} dims)\n\n"
            f"   Results may be suboptimal. Consider re-indexing for best results."
        )
        return True, msg

    def clear_for_reindex(self) -> bool:
        """Clear database for re-indexing with a new model.

        Drops all tables and resets config. Use with caution.

        Returns:
            True if successful, False otherwise.
        """
        try:
            db = lancedb.connect(str(self.db_path))

            # Drop existing tables
            for table_name in db.table_names():
                db.drop_table(table_name)
                logger.info(f"Dropped table: {table_name}")

            # Reset config but keep schema version
            if self.config_file.exists():
                self.config_file.unlink()

            logger.info("Database cleared for re-indexing")
            return True

        except Exception as e:
            logger.error(f"Failed to clear database: {e}")
            return False

    def update_entity_count(self, count: int) -> None:
        """Update the entity count in embedding model metadata.

        Args:
            count: New entity count
        """
        model_info = self.get_embedding_model()
        if model_info:
            self.set_embedding_model(
                model_name=model_info.name,
                model_path=model_info.model_path,
                dimensions=model_info.dimensions,
                entity_count=count,
            )
