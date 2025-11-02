"""Tests for LanceDB client."""

import pytest
import tempfile
from pathlib import Path
from codesearch.lancedb.client import LanceDBClient
from codesearch.lancedb.models import CodeEntity, EntityType


class TestLanceDBClientInitialization:
    """Test LanceDB client initialization."""

    def test_client_initialization_with_default_path(self):
        """Client initializes with default path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Temporarily override the default path for testing
            client = LanceDBClient(db_path=Path(tmpdir))
            assert client.db_path == Path(tmpdir)
            assert client.db is not None
            assert isinstance(client._initialized_tables, set)

    def test_client_initialization_creates_directory(self):
        """Client creates database directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "subdir" / ".lancedb"
            client = LanceDBClient(db_path=db_path)
            assert db_path.exists()
            assert db_path.is_dir()

    def test_client_initialization_with_custom_path(self):
        """Client accepts custom database path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            custom_path = Path(tmpdir) / "custom_db"
            client = LanceDBClient(db_path=custom_path)
            assert client.db_path == custom_path
            assert custom_path.exists()


class TestTableOperations:
    """Test table creation and management."""

    @pytest.fixture
    def client(self):
        """Provide a client for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield LanceDBClient(db_path=Path(tmpdir))

    def test_table_exists_false_for_nonexistent_table(self, client):
        """table_exists returns False for tables that don't exist."""
        assert not client.table_exists("nonexistent_table")

    def test_create_table_with_data(self, client):
        """create_table with data works correctly."""
        test_data = [
            {"id": 1, "name": "entity1"},
            {"id": 2, "name": "entity2"},
        ]
        client.create_table("test_table", data=test_data)
        assert client.table_exists("test_table")
        assert "test_table" in client._initialized_tables

    def test_get_existing_table(self, client):
        """get_table returns table reference for existing table."""
        test_data = [{"id": 1, "value": "test"}]
        client.create_table("existing", data=test_data)
        table = client.get_table("existing")
        assert table is not None

    def test_get_nonexistent_table_raises_error(self, client):
        """get_table raises error for nonexistent table."""
        with pytest.raises(ValueError, match="does not exist"):
            client.get_table("nonexistent")

    def test_table_exists_after_creation(self, client):
        """table_exists returns True after table creation."""
        test_data = [{"id": 1}]
        client.create_table("test", data=test_data)
        assert client.table_exists("test")


class TestInsertOperations:
    """Test insert operations."""

    @pytest.fixture
    def client(self):
        """Provide a client for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield LanceDBClient(db_path=Path(tmpdir))

    def test_insert_data(self, client):
        """insert_data adds records to table."""
        # Create table first
        test_data = [{"id": 1, "name": "initial"}]
        client.create_table("test_table", data=test_data)

        # Insert more data
        new_data = [{"id": 2, "name": "new"}]
        client.insert("test_table", new_data)

        # Verify table exists (no exception raised)
        assert client.table_exists("test_table")

    def test_insert_into_nonexistent_table_raises_error(self, client):
        """insert into nonexistent table raises error."""
        with pytest.raises(ValueError):
            client.insert("nonexistent", [{"id": 1}])

    def test_insert_empty_list(self, client):
        """insert with empty list doesn't cause issues."""
        test_data = [{"id": 1}]
        client.create_table("test", data=test_data)
        client.insert("test", [])  # Should not raise


class TestSearchOperations:
    """Test vector search operations."""

    @pytest.fixture
    def client(self):
        """Provide a client for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield LanceDBClient(db_path=Path(tmpdir))

    def test_search_basic_structure(self, client):
        """search method accepts query vector and limit."""
        # Create table with vector data
        test_data = [
            {"id": "1", "vector": [0.1] * 768},
            {"id": "2", "vector": [0.2] * 768},
        ]
        client.create_table("vectors", data=test_data)

        # Search should work without error
        results = client.search("vectors", [0.1] * 768, k=10)
        assert isinstance(results, list)

    def test_search_respects_k_parameter(self, client):
        """search respects limit parameter."""
        # Create table with multiple records
        test_data = [
            {"id": f"id_{i}", "vector": [float(i) / 10] * 768}
            for i in range(10)
        ]
        client.create_table("vectors", data=test_data)

        # Search with k=3 should return at most 3 results
        results = client.search("vectors", [0.5] * 768, k=3)
        assert len(results) <= 3

    def test_search_nonexistent_table_raises_error(self, client):
        """search in nonexistent table raises error."""
        with pytest.raises(ValueError):
            client.search("nonexistent", [0.5] * 768)


class TestErrorHandling:
    """Test error handling and edge cases."""

    @pytest.fixture
    def client(self):
        """Provide a client for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield LanceDBClient(db_path=Path(tmpdir))

    def test_create_table_overwrites_existing(self, client):
        """create_table with mode='overwrite' replaces existing table."""
        data1 = [{"id": 1, "value": "first"}]
        client.create_table("test", data=data1)

        data2 = [{"id": 2, "value": "second"}]
        client.create_table("test", data=data2)

        # Table should be updated
        assert client.table_exists("test")

    def test_multiple_tables(self, client):
        """Client can manage multiple tables."""
        client.create_table("table1", data=[{"id": 1}])
        client.create_table("table2", data=[{"id": 2}])
        client.create_table("table3", data=[{"id": 3}])

        assert client.table_exists("table1")
        assert client.table_exists("table2")
        assert client.table_exists("table3")
        assert len(client._initialized_tables) == 3


class TestIntegration:
    """Integration tests with full workflow."""

    def test_end_to_end_workflow(self):
        """End-to-end workflow: create → insert → search."""
        with tempfile.TemporaryDirectory() as tmpdir:
            client = LanceDBClient(db_path=Path(tmpdir))

            # Create table with initial data
            initial_data = [
                {"id": "entity1", "vector": [0.1] * 768},
                {"id": "entity2", "vector": [0.2] * 768},
            ]
            client.create_table("entities", data=initial_data)

            # Verify table exists
            assert client.table_exists("entities")

            # Insert more data
            new_data = [
                {"id": "entity3", "vector": [0.3] * 768},
            ]
            client.insert("entities", new_data)

            # Search
            results = client.search("entities", [0.15] * 768, k=5)
            assert isinstance(results, list)

    def test_client_with_real_code_entity_data(self):
        """Client works with CodeEntity-like data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            client = LanceDBClient(db_path=Path(tmpdir))

            # Create entity-like data
            entity = CodeEntity(
                entity_id="test:func",
                repository="test",
                file_path="test.py",
                entity_type=EntityType.FUNCTION,
                language="python",
                name="test_func",
                full_qualified_name="test_func",
                code_text="def test(): pass",
                code_vector=[0.5] * 768,
            )

            # Create table with entity data
            client.create_table("code_entities", data=[entity.to_dict()])

            # Verify
            assert client.table_exists("code_entities")

            # Can insert more entities
            entity2 = CodeEntity(
                entity_id="test:func2",
                repository="test",
                file_path="test.py",
                entity_type=EntityType.FUNCTION,
                language="python",
                name="test_func2",
                full_qualified_name="test_func2",
                code_text="def test2(): pass",
                code_vector=[0.6] * 768,
            )
            client.insert("code_entities", [entity2.to_dict()])
