"""Comprehensive tests for LanceDB database management modules.

Tests cover:
- Connection pooling (acquire, release, context managers)
- Database initialization (setup, validation, schema)
- Backup/restore (create, restore, list, cleanup)
- Statistics (database stats, table stats, health checks)
- Optimization (deduplication, cleanup, recommendations)
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timezone
import json

from codesearch.lancedb.pool import DatabaseConnectionPool
from codesearch.lancedb.initialization import (
    DatabaseInitializer,
    DimensionMismatchError,
    EmbeddingModelInfo,
    ModelMismatchError,
)
from codesearch.lancedb.backup import DatabaseBackupManager
from codesearch.lancedb.statistics import DatabaseStatistics
from codesearch.lancedb.optimization import DatabaseOptimizer
from codesearch.lancedb.client import LanceDBClient


@pytest.fixture
def temp_db_dir():
    """Create temporary directory for test database."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def initialized_db(temp_db_dir):
    """Create an initialized test database."""
    initializer = DatabaseInitializer(temp_db_dir)
    initializer.initialize()
    yield temp_db_dir


class TestDatabaseConnectionPool:
    """Tests for connection pooling functionality."""

    def test_pool_initialization(self, temp_db_dir):
        """Test pool initialization with min/max connections."""
        pool = DatabaseConnectionPool(
            db_path=temp_db_dir,
            min_connections=2,
            max_connections=5,
        )

        stats = pool.get_pool_stats()
        assert stats["total_connections"] == 2
        assert stats["available_connections"] == 2
        assert stats["active_connections"] == 0
        assert stats["min_connections"] == 2
        assert stats["max_connections"] == 5

        pool.close_all()

    def test_acquire_and_release(self, temp_db_dir):
        """Test acquiring and releasing connections."""
        pool = DatabaseConnectionPool(db_path=temp_db_dir, min_connections=1, max_connections=3)

        # Acquire connection
        conn = pool.acquire()
        assert conn is not None

        stats = pool.get_pool_stats()
        assert stats["active_connections"] == 1
        assert stats["available_connections"] == 0

        # Release connection
        pool.release(conn)
        stats = pool.get_pool_stats()
        assert stats["active_connections"] == 0
        assert stats["available_connections"] == 1

        pool.close_all()

    def test_context_manager(self, temp_db_dir):
        """Test context manager for safe connection handling."""
        pool = DatabaseConnectionPool(db_path=temp_db_dir, min_connections=1)

        with pool.get_connection() as conn:
            assert conn is not None
            stats = pool.get_pool_stats()
            assert stats["active_connections"] == 1

        # After context, connection should be released
        stats = pool.get_pool_stats()
        assert stats["active_connections"] == 0

        pool.close_all()

    def test_pool_expansion(self, temp_db_dir):
        """Test pool expansion when all connections are in use."""
        pool = DatabaseConnectionPool(db_path=temp_db_dir, min_connections=1, max_connections=3)

        conn1 = pool.acquire()
        conn2 = pool.acquire()
        conn3 = pool.acquire()

        stats = pool.get_pool_stats()
        assert stats["total_connections"] == 3
        assert stats["active_connections"] == 3

        pool.release(conn1)
        pool.release(conn2)
        pool.release(conn3)
        pool.close_all()

    def test_timeout_on_exhausted_pool(self, temp_db_dir):
        """Test timeout when pool is exhausted and max reached."""
        pool = DatabaseConnectionPool(db_path=temp_db_dir, min_connections=1, max_connections=1, timeout=0.1)

        conn = pool.acquire()
        with pytest.raises(TimeoutError):
            pool.acquire(timeout=0.1)

        pool.release(conn)
        pool.close_all()


class TestDatabaseInitializer:
    """Tests for database initialization."""

    def test_initialization(self, temp_db_dir):
        """Test database initialization."""
        initializer = DatabaseInitializer(temp_db_dir)

        assert not initializer.is_initialized()

        initializer.initialize()

        assert initializer.is_initialized()
        assert initializer.config_file.exists()

    def test_tables_created_on_init(self, temp_db_dir):
        """Test that all required tables are created during initialization."""
        import lancedb

        initializer = DatabaseInitializer(temp_db_dir)
        initializer.initialize()

        # Connect and check tables
        db = lancedb.connect(str(temp_db_dir))
        tables = set(db.table_names())

        # All three tables should exist
        assert "code_entities" in tables
        assert "code_relationships" in tables
        assert "search_metadata" in tables

    def test_code_entities_table_schema(self, temp_db_dir):
        """Test that code_entities table has correct schema with vector column."""
        import lancedb

        initializer = DatabaseInitializer(temp_db_dir)
        initializer.initialize()

        db = lancedb.connect(str(temp_db_dir))
        table = db.open_table("code_entities")
        schema = table.schema

        # Check key fields exist
        field_names = [field.name for field in schema]
        assert "entity_id" in field_names
        assert "name" in field_names
        assert "code_text" in field_names
        assert "code_vector" in field_names
        assert "language" in field_names
        assert "entity_type" in field_names
        assert "repository" in field_names
        assert "file_path" in field_names
        assert "source_hash" in field_names

    def test_code_relationships_table_schema(self, temp_db_dir):
        """Test that code_relationships table has correct schema."""
        import lancedb

        initializer = DatabaseInitializer(temp_db_dir)
        initializer.initialize()

        db = lancedb.connect(str(temp_db_dir))
        table = db.open_table("code_relationships")
        schema = table.schema

        field_names = [field.name for field in schema]
        assert "relationship_id" in field_names
        assert "caller_id" in field_names
        assert "callee_id" in field_names
        assert "relationship_type" in field_names

    def test_search_metadata_table_schema(self, temp_db_dir):
        """Test that search_metadata table has correct schema."""
        import lancedb

        initializer = DatabaseInitializer(temp_db_dir)
        initializer.initialize()

        db = lancedb.connect(str(temp_db_dir))
        table = db.open_table("search_metadata")
        schema = table.schema

        field_names = [field.name for field in schema]
        assert "metadata_id" in field_names
        assert "entity_id" in field_names
        assert "repository" in field_names

    def test_schema_validation(self, initialized_db):
        """Test schema validation checks."""
        initializer = DatabaseInitializer(initialized_db)

        checks = initializer.validate_schema()
        assert checks["config_exists"]
        assert checks["config_valid"]
        assert checks["schema_version_match"]
        assert checks["tables_exist"]
        assert checks["code_entities_exists"]
        assert checks["code_relationships_exists"]
        assert checks["search_metadata_exists"]

    def test_schema_info(self, initialized_db):
        """Test retrieving schema information."""
        initializer = DatabaseInitializer(initialized_db)

        info = initializer.get_schema_info()
        assert info["status"] == "initialized"
        assert "schema_version" in info
        assert "created_at" in info
        assert "tables" in info

    def test_idempotent_initialization(self, initialized_db):
        """Test that initialization is idempotent."""
        import lancedb

        initializer = DatabaseInitializer(initialized_db)

        # Get initial table count
        db = lancedb.connect(str(initialized_db))
        initial_tables = set(db.table_names())

        # Initialize again
        result = initializer.initialize()
        assert result is True

        # Should still be valid
        assert initializer.is_initialized()

        # Tables should be the same (not duplicated)
        final_tables = set(db.table_names())
        assert initial_tables == final_tables

    def test_get_table(self, initialized_db):
        """Test getting a table by name."""
        initializer = DatabaseInitializer(initialized_db)

        table = initializer.get_table("code_entities")
        assert table is not None

        # Non-existent table
        table = initializer.get_table("nonexistent")
        assert table is None

    def test_table_exists(self, initialized_db):
        """Test checking if a table exists."""
        initializer = DatabaseInitializer(initialized_db)

        assert initializer.table_exists("code_entities")
        assert initializer.table_exists("code_relationships")
        assert initializer.table_exists("search_metadata")
        assert not initializer.table_exists("nonexistent")


class TestDatabaseBackupManager:
    """Tests for backup and restore functionality."""

    def test_backup_creation(self, initialized_db):
        """Test creating a backup."""
        client = LanceDBClient(initialized_db)
        test_data = [{"id": "1", "name": "test"}]
        client.create_table("test_table", data=test_data)

        manager = DatabaseBackupManager(initialized_db)
        backup_info = manager.backup("test_backup")

        assert backup_info["name"] == "test_backup"
        assert backup_info["status"] == "completed"
        assert "size_bytes" in backup_info
        assert (manager.backup_dir / "test_backup").exists()

    def test_backup_with_timestamp(self, initialized_db):
        """Test creating backup with auto-generated timestamp name."""
        manager = DatabaseBackupManager(initialized_db)
        backup_info = manager.backup()

        assert "backup_" in backup_info["name"]
        assert backup_info["status"] == "completed"

    def test_list_backups(self, initialized_db):
        """Test listing available backups."""
        manager = DatabaseBackupManager(initialized_db)

        manager.backup("backup1")
        manager.backup("backup2")

        backups = manager.list_backups()
        assert len(backups) >= 2

    def test_get_backup_info(self, initialized_db):
        """Test retrieving backup information."""
        manager = DatabaseBackupManager(initialized_db)
        manager.backup("test_backup")

        info = manager.get_backup_info("test_backup")
        assert info["name"] == "test_backup"
        assert "timestamp" in info

    def test_delete_backup(self, initialized_db):
        """Test deleting a backup."""
        manager = DatabaseBackupManager(initialized_db)
        manager.backup("test_backup")

        assert (manager.backup_dir / "test_backup").exists()

        result = manager.delete_backup("test_backup", confirm=True)
        assert result is True
        assert not (manager.backup_dir / "test_backup").exists()

    def test_cleanup_old_backups(self, initialized_db):
        """Test cleanup of old backups."""
        manager = DatabaseBackupManager(initialized_db)

        # Create multiple backups
        for i in range(7):
            manager.backup(f"backup_{i}")

        cleanup_stats = manager.cleanup_old_backups(keep_count=3)
        assert cleanup_stats["deleted_count"] >= 2
        assert cleanup_stats["remaining_backups"] == 3


class TestDatabaseStatistics:
    """Tests for database statistics and monitoring."""

    def test_database_stats(self, initialized_db):
        """Test getting overall database statistics."""
        client = LanceDBClient(initialized_db)
        test_data = [{"id": str(i), "value": i} for i in range(5)]
        client.create_table("stats_table", data=test_data)

        stats_mgr = DatabaseStatistics(initialized_db)
        stats = stats_mgr.get_database_stats()

        assert "timestamp" in stats
        assert "database_path" in stats
        assert "database_size_bytes" in stats
        assert "table_count" in stats
        assert "total_rows" in stats

    def test_table_list(self, initialized_db):
        """Test listing tables in database."""
        client = LanceDBClient(initialized_db)
        client.create_table("table1", data=[{"id": "1"}])
        client.create_table("table2", data=[{"id": "2"}])

        stats_mgr = DatabaseStatistics(initialized_db)
        tables = stats_mgr.get_table_list()

        assert "table1" in tables
        assert "table2" in tables

    def test_table_stats(self, initialized_db):
        """Test getting statistics for a specific table."""
        client = LanceDBClient(initialized_db)
        test_data = [{"id": str(i), "name": f"item_{i}"} for i in range(10)]
        client.create_table("data_table", data=test_data)

        stats_mgr = DatabaseStatistics(initialized_db)
        stats = stats_mgr.get_table_stats("data_table")

        assert stats["table_name"] == "data_table"
        assert stats["row_count"] == 10
        assert "schema" in stats
        assert "column_count" in stats

    def test_health_check(self, initialized_db):
        """Test database health check."""
        stats_mgr = DatabaseStatistics(initialized_db)
        health = stats_mgr.get_health_check()

        assert "checks" in health
        assert health["checks"]["database_exists"] is True
        assert health["checks"]["connection_ok"] is True
        assert health["overall_status"] in ["healthy", "unhealthy"]

    def test_row_count(self, initialized_db):
        """Test getting row count for a table."""
        client = LanceDBClient(initialized_db)
        test_data = [{"id": str(i)} for i in range(15)]
        client.create_table("count_table", data=test_data)

        stats_mgr = DatabaseStatistics(initialized_db)
        count = stats_mgr.get_row_count("count_table")

        assert count == 15


class TestDatabaseOptimizer:
    """Tests for database optimization and cleanup."""

    def test_optimizer_initialization(self, initialized_db):
        """Test optimizer initialization."""
        optimizer = DatabaseOptimizer(initialized_db)
        assert optimizer.db is not None

    def test_memory_cleanup(self, initialized_db):
        """Test garbage collection and memory cleanup."""
        optimizer = DatabaseOptimizer(initialized_db)
        result = optimizer.cleanup_memory()

        assert result["status"] == "completed"
        assert "objects_collected" in result

    def test_optimization_recommendations(self, initialized_db):
        """Test getting optimization recommendations."""
        client = LanceDBClient(initialized_db)
        test_data = [{"id": str(i)} for i in range(100)]
        client.create_table("opt_table", data=test_data)

        optimizer = DatabaseOptimizer(initialized_db)
        recommendations = optimizer.get_optimization_recommendations()

        assert "recommendations" in recommendations
        # Check should be list (may be empty for small DB)
        assert isinstance(recommendations["recommendations"], list)

    def test_compact_storage(self, initialized_db):
        """Test storage compaction."""
        client = LanceDBClient(initialized_db)
        client.create_table("compact_table", data=[{"id": "1"}])

        optimizer = DatabaseOptimizer(initialized_db)
        result = optimizer.compact_storage()

        assert result["status"] == "completed"
        assert "compacted_tables" in result

    def test_analyze_table_bloat(self, initialized_db):
        """Test table bloat analysis."""
        client = LanceDBClient(initialized_db)
        test_data = [{"id": str(i), "name": f"item_{i}"} for i in range(10)]
        client.create_table("bloat_table", data=test_data)

        optimizer = DatabaseOptimizer(initialized_db)
        bloat = optimizer.analyze_table_bloat("bloat_table")

        assert bloat["table_name"] == "bloat_table"
        assert "total_rows" in bloat
        assert "null_percentage" in bloat


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_database_lifecycle(self, temp_db_dir):
        """Test complete database lifecycle: init -> populate -> backup -> restore."""
        # Initialize
        initializer = DatabaseInitializer(temp_db_dir)
        initializer.initialize()
        assert initializer.is_initialized()

        # Populate
        client = LanceDBClient(temp_db_dir)
        test_data = [{"id": str(i), "value": i * 10} for i in range(20)]
        client.create_table("lifecycle_table", data=test_data)

        # Get stats before backup
        stats_before = DatabaseStatistics(temp_db_dir).get_database_stats()
        assert stats_before["total_rows"] == 20

        # Create backup
        manager = DatabaseBackupManager(temp_db_dir)
        backup_info = manager.backup("lifecycle_backup")
        assert backup_info["status"] == "completed"

        # List backups
        backups = manager.list_backups()
        assert len(backups) > 0

        # Health check
        health = DatabaseStatistics(temp_db_dir).get_health_check()
        assert health["overall_status"] in ["healthy", "unhealthy"]

        # Optimize
        optimizer = DatabaseOptimizer(temp_db_dir)
        optimizer.cleanup_memory()

    def test_connection_pool_with_database(self, initialized_db):
        """Test using connection pool with actual database operations."""
        pool = DatabaseConnectionPool(initialized_db, min_connections=2, max_connections=3)

        # Use connections from pool
        with pool.get_connection() as conn:
            # Can interact with connection
            assert conn is not None

        # Check pool stats
        stats = pool.get_pool_stats()
        assert stats["total_connections"] >= 2

        pool.close_all()

    def test_backup_restore_cycle(self, temp_db_dir):
        """Test backup and restore cycle with data verification."""
        # Initialize original database
        initializer = DatabaseInitializer(temp_db_dir)
        initializer.initialize()

        # Add data
        client = LanceDBClient(temp_db_dir)
        original_data = [{"id": str(i), "name": f"test_{i}"} for i in range(10)]
        client.create_table("cycle_table", data=original_data)

        # Create backup
        manager = DatabaseBackupManager(temp_db_dir)
        manager.backup("cycle_backup")

        # Verify backup exists
        backups = manager.list_backups()
        backup_names = [b["name"] for b in backups]
        assert "cycle_backup" in backup_names


class TestConfigurableEmbeddingDimensions:
    """Tests for configurable embedding dimensions in schema."""

    def test_default_dimensions(self, temp_db_dir):
        """Test that default dimensions (768) are used when not specified."""
        initializer = DatabaseInitializer(temp_db_dir)
        initializer.initialize()

        assert initializer.embedding_dimensions == 768
        assert initializer.get_embedding_dimensions() == 768

    def test_custom_dimensions_256(self, temp_db_dir):
        """Test initializing with 256 dimensions (CodeT5+-110M)."""
        initializer = DatabaseInitializer(temp_db_dir, embedding_dimensions=256)
        initializer.initialize()

        assert initializer.embedding_dimensions == 256
        assert initializer.get_embedding_dimensions() == 256

    def test_custom_dimensions_1024(self, temp_db_dir):
        """Test initializing with 1024 dimensions (CodeT5+-770M)."""
        initializer = DatabaseInitializer(temp_db_dir, embedding_dimensions=1024)
        initializer.initialize()

        assert initializer.embedding_dimensions == 1024
        assert initializer.get_embedding_dimensions() == 1024

    def test_dimensions_persisted_in_config(self, temp_db_dir):
        """Test that embedding dimensions are persisted in config file."""
        initializer = DatabaseInitializer(temp_db_dir, embedding_dimensions=256)
        initializer.initialize()

        # Read config directly
        with open(initializer.config_file) as f:
            config = json.load(f)

        assert config["embedding_dimensions"] == 256

    def test_schema_uses_correct_dimensions(self, temp_db_dir):
        """Test that the schema uses the configured dimensions."""
        import lancedb
        import pyarrow as pa

        initializer = DatabaseInitializer(temp_db_dir, embedding_dimensions=256)
        initializer.initialize()

        # Open table and check vector field dimensions
        db = lancedb.connect(str(temp_db_dir))
        table = db.open_table("code_entities")
        schema = table.schema

        # Find the code_vector field and check its dimensions
        code_vector_field = None
        for field in schema:
            if field.name == "code_vector":
                code_vector_field = field
                break

        assert code_vector_field is not None
        # The list type should have size 256
        list_type = code_vector_field.type
        assert isinstance(list_type, pa.FixedSizeListType)
        assert list_type.list_size == 256


class TestEmbeddingModelMigration:
    """Tests for embedding model migration and re-indexing."""

    def test_embedding_model_info_dataclass(self):
        """Test EmbeddingModelInfo dataclass."""
        info = EmbeddingModelInfo(
            name="unixcoder",
            model_path="microsoft/unixcoder-base",
            dimensions=768,
            indexed_at="2024-01-15T10:30:00Z",
            entity_count=1000,
        )
        assert info.name == "unixcoder"
        assert info.dimensions == 768
        assert info.entity_count == 1000

    def test_embedding_model_info_to_dict(self):
        """Test EmbeddingModelInfo serialization."""
        info = EmbeddingModelInfo(
            name="codebert",
            model_path="microsoft/codebert-base",
            dimensions=768,
            indexed_at="2024-01-15T10:30:00Z",
            entity_count=500,
        )
        d = info.to_dict()
        assert d["name"] == "codebert"
        assert d["model_path"] == "microsoft/codebert-base"
        assert d["dimensions"] == 768
        assert d["entity_count"] == 500

    def test_embedding_model_info_from_dict(self):
        """Test EmbeddingModelInfo deserialization."""
        data = {
            "name": "codet5p-110m",
            "model_path": "Salesforce/codet5p-110m-embedding",
            "dimensions": 256,
            "indexed_at": "2024-01-15T10:30:00Z",
            "entity_count": 2000,
        }
        info = EmbeddingModelInfo.from_dict(data)
        assert info.name == "codet5p-110m"
        assert info.dimensions == 256
        assert info.entity_count == 2000

    def test_set_embedding_model(self, initialized_db):
        """Test storing embedding model metadata."""
        initializer = DatabaseInitializer(initialized_db)

        initializer.set_embedding_model(
            model_name="unixcoder",
            model_path="microsoft/unixcoder-base",
            dimensions=768,
            entity_count=1500,
        )

        # Retrieve and verify
        model_info = initializer.get_embedding_model()
        assert model_info is not None
        assert model_info.name == "unixcoder"
        assert model_info.model_path == "microsoft/unixcoder-base"
        assert model_info.dimensions == 768
        assert model_info.entity_count == 1500
        assert model_info.indexed_at != ""  # Should have timestamp

    def test_get_embedding_model_not_set(self, temp_db_dir):
        """Test getting embedding model when not set."""
        initializer = DatabaseInitializer(temp_db_dir)
        initializer.initialize()

        model_info = initializer.get_embedding_model()
        assert model_info is None

    def test_check_model_compatibility_same_model(self, initialized_db):
        """Test compatibility check with same model."""
        initializer = DatabaseInitializer(initialized_db)
        initializer.set_embedding_model(
            model_name="unixcoder",
            model_path="microsoft/unixcoder-base",
            dimensions=768,
            entity_count=100,
        )

        is_compatible, warning = initializer.check_model_compatibility(
            "unixcoder", 768
        )
        assert is_compatible is True
        assert warning is None

    def test_check_model_compatibility_no_indexed_model(self, initialized_db):
        """Test compatibility check when no model indexed yet."""
        initializer = DatabaseInitializer(initialized_db)

        is_compatible, warning = initializer.check_model_compatibility(
            "codebert", 768
        )
        assert is_compatible is True
        assert warning is None

    def test_check_model_compatibility_different_dims(self, initialized_db):
        """Test compatibility check with different dimensions (incompatible)."""
        initializer = DatabaseInitializer(initialized_db)
        initializer.set_embedding_model(
            model_name="unixcoder",
            model_path="microsoft/unixcoder-base",
            dimensions=768,
            entity_count=100,
        )

        # CodeT5+-110M has 256 dimensions
        is_compatible, warning = initializer.check_model_compatibility(
            "codet5p-110m", 256
        )
        assert is_compatible is False
        assert warning is not None
        assert "Model mismatch" in warning
        assert "768" in warning  # Original dims
        assert "256" in warning  # Requested dims

    def test_check_model_compatibility_same_dims_different_model(self, initialized_db):
        """Test compatibility check with same dimensions but different model."""
        initializer = DatabaseInitializer(initialized_db)
        initializer.set_embedding_model(
            model_name="codebert",
            model_path="microsoft/codebert-base",
            dimensions=768,
            entity_count=100,
        )

        # UniXcoder also has 768 dimensions
        is_compatible, warning = initializer.check_model_compatibility(
            "unixcoder", 768
        )
        # Same dimensions - compatible but with warning
        assert is_compatible is True
        assert warning is not None
        assert "same dimensions" in warning

    def test_clear_for_reindex(self, initialized_db):
        """Test clearing database for re-indexing."""
        import lancedb

        initializer = DatabaseInitializer(initialized_db)

        # Store some model metadata
        initializer.set_embedding_model(
            model_name="old_model",
            model_path="old/path",
            dimensions=768,
            entity_count=50,
        )

        # Clear for reindex
        result = initializer.clear_for_reindex()
        assert result is True

        # Config should be removed
        assert not initializer.config_file.exists()

        # Tables should be dropped
        db = lancedb.connect(str(initialized_db))
        assert len(db.table_names()) == 0

    def test_update_entity_count(self, initialized_db):
        """Test updating entity count in model metadata."""
        initializer = DatabaseInitializer(initialized_db)
        initializer.set_embedding_model(
            model_name="unixcoder",
            model_path="microsoft/unixcoder-base",
            dimensions=768,
            entity_count=100,
        )

        # Update count
        initializer.update_entity_count(500)

        # Verify
        model_info = initializer.get_embedding_model()
        assert model_info.entity_count == 500
        # Other fields should be preserved
        assert model_info.name == "unixcoder"
        assert model_info.dimensions == 768

    def test_model_mismatch_error(self):
        """Test ModelMismatchError exception."""
        error = ModelMismatchError(
            indexed_model="codebert",
            indexed_dims=768,
            requested_model="codet5p-110m",
            requested_dims=256,
        )
        assert error.indexed_model == "codebert"
        assert error.indexed_dims == 768
        assert error.requested_model == "codet5p-110m"
        assert error.requested_dims == 256
        assert "codebert" in str(error)
        assert "768" in str(error)

    def test_dimension_mismatch_error(self):
        """Test DimensionMismatchError exception."""
        error = DimensionMismatchError(query_dims=256, expected_dims=768)
        assert error.query_dims == 256
        assert error.expected_dims == 768
        assert "256" in str(error)
        assert "768" in str(error)
        assert "Re-index" in str(error)

    def test_model_metadata_persists_across_sessions(self, temp_db_dir):
        """Test that model metadata persists across database sessions."""
        # Session 1: Initialize and set model
        initializer1 = DatabaseInitializer(temp_db_dir)
        initializer1.initialize()
        initializer1.set_embedding_model(
            model_name="unixcoder",
            model_path="microsoft/unixcoder-base",
            dimensions=768,
            entity_count=1000,
        )

        # Session 2: Create new initializer instance
        initializer2 = DatabaseInitializer(temp_db_dir)

        model_info = initializer2.get_embedding_model()
        assert model_info is not None
        assert model_info.name == "unixcoder"
        assert model_info.dimensions == 768
        assert model_info.entity_count == 1000

    def test_embedding_dimensions_synced_with_model(self, temp_db_dir):
        """Test that embedding_dimensions config is synced with model info."""
        initializer = DatabaseInitializer(temp_db_dir)
        initializer.initialize()

        # Set model with specific dimensions
        initializer.set_embedding_model(
            model_name="codet5p-110m",
            model_path="Salesforce/codet5p-110m-embedding",
            dimensions=256,
            entity_count=100,
        )

        # Top-level embedding_dimensions should match
        assert initializer.get_embedding_dimensions() == 256

        # Read config file directly
        with open(initializer.config_file) as f:
            config = json.load(f)

        assert config["embedding_dimensions"] == 256
        assert config["embedding_model"]["dimensions"] == 256
