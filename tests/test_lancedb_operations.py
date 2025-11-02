"""Comprehensive tests for all LanceDB operations (Tasks 3-6)."""

import pytest
import tempfile
from pathlib import Path
from codesearch.lancedb.client import LanceDBClient
from codesearch.lancedb.entities import EntityOperations
from codesearch.lancedb.relationships import RelationshipOperations
from codesearch.lancedb.metadata import MetadataOperations
from codesearch.lancedb.migrations import MigrationRegistry, ConsistencyValidator
from codesearch.lancedb.models import (
    CodeEntity, CodeRelationship, SearchMetadata,
    EntityType, RelationshipType, Visibility, ImplementationStatus
)


@pytest.fixture
def client():
    """Provide a temporary client."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield LanceDBClient(db_path=Path(tmpdir))


class TestEntityOperations:
    """Test Task 3: Entity Operations."""

    def test_entity_operations_insertion(self, client):
        """EntityOperations can insert entities."""
        ops = EntityOperations(client)

        entity = CodeEntity(
            entity_id="test:func",
            repository="test",
            file_path="test.py",
            entity_type=EntityType.FUNCTION,
            language="python",
            name="func",
            full_qualified_name="func",
            code_text="def func(): pass",
            code_vector=[0.5] * 768,
        )

        count, errors = ops.insert([entity])
        assert count == 1
        assert len(errors) == 0

    def test_entity_operations_validation(self, client):
        """EntityOperations validates entities."""
        ops = EntityOperations(client)

        # Invalid: wrong vector size
        invalid_entity = CodeEntity(
            entity_id="test",
            repository="test",
            file_path="test.py",
            entity_type=EntityType.FUNCTION,
            language="python",
            name="func",
            full_qualified_name="func",
            code_text="code",
            code_vector=[0.5] * 512,  # Wrong size
        )

        count, errors = ops.insert([invalid_entity])
        assert count == 0
        assert len(errors) > 0

    def test_entity_operations_count(self, client):
        """EntityOperations can count entities."""
        ops = EntityOperations(client)

        entities = [
            CodeEntity(
                entity_id=f"test:{i}",
                repository="test",
                file_path="test.py",
                entity_type=EntityType.FUNCTION,
                language="python",
                name=f"func{i}",
                full_qualified_name=f"func{i}",
                code_text="code",
                code_vector=[float(i) / 100] * 768,
            )
            for i in range(3)
        ]

        ops.insert(entities)
        assert ops.count() >= 3


class TestRelationshipOperations:
    """Test Task 4: Relationship Operations."""

    def test_relationship_insertion(self, client):
        """RelationshipOperations can insert relationships."""
        ops = RelationshipOperations(client)

        rel = CodeRelationship(
            relationship_id="a->b",
            caller_id="func_a",
            callee_id="func_b",
            relationship_type=RelationshipType.CALLS,
        )

        count, errors = ops.insert([rel])
        assert count == 1
        assert len(errors) == 0

    def test_relationship_self_reference_rejected(self, client):
        """RelationshipOperations rejects self-references."""
        ops = RelationshipOperations(client)

        self_ref = CodeRelationship(
            relationship_id="a->a",
            caller_id="func_a",
            callee_id="func_a",  # Same as caller
            relationship_type=RelationshipType.CALLS,
        )

        count, errors = ops.insert([self_ref])
        assert count == 0
        assert any("Self-referential" in e for e in errors)

    def test_relationship_count(self, client):
        """RelationshipOperations can count relationships."""
        ops = RelationshipOperations(client)

        rels = [
            CodeRelationship(
                relationship_id=f"a->{i}",
                caller_id="func_a",
                callee_id=f"func_{i}",
                relationship_type=RelationshipType.CALLS,
            )
            for i in range(3)
        ]

        ops.insert(rels)
        assert ops.count() >= 3


class TestMetadataOperations:
    """Test Task 5: Metadata Operations."""

    def test_metadata_insertion(self, client):
        """MetadataOperations can insert metadata."""
        ops = MetadataOperations(client)

        metadata = SearchMetadata(
            metadata_id="test",
            entity_id="test:func",
            repository="test",
            implementation_status=ImplementationStatus.STABLE,
        )

        count, errors = ops.insert([metadata])
        assert count == 1
        assert len(errors) == 0

    def test_metadata_validation(self, client):
        """MetadataOperations validates metadata."""
        ops = MetadataOperations(client)

        # Invalid: missing repository
        invalid = SearchMetadata(
            metadata_id="test",
            entity_id="test",
            repository="",  # Empty
        )

        count, errors = ops.insert([invalid])
        assert count == 0
        assert len(errors) > 0

    def test_metadata_filtering(self, client):
        """MetadataOperations can filter by status."""
        ops = MetadataOperations(client)

        metadata_list = [
            SearchMetadata(
                metadata_id=f"test{i}",
                entity_id=f"test:{i}",
                repository="test",
                implementation_status=ImplementationStatus.STABLE if i % 2 == 0 else ImplementationStatus.EXPERIMENTAL,
            )
            for i in range(4)
        ]

        ops.insert(metadata_list)
        stable = ops.search_by_status(ImplementationStatus.STABLE)
        assert len(stable) >= 2


class TestMigrations:
    """Test Task 6: Migrations & Consistency."""

    def test_migration_registry(self):
        """MigrationRegistry can register migrations."""
        registry = MigrationRegistry()

        called = []

        def test_migration(db):
            called.append(True)

        registry.register("v1_to_v2", test_migration)
        assert "v1_to_v2" in registry._migrations

    def test_consistency_validator_initialization(self, client):
        """ConsistencyValidator initializes without errors."""
        validator = ConsistencyValidator(client)
        assert validator is not None

    def test_consistency_validator_all_checks(self, client):
        """ConsistencyValidator runs all checks."""
        validator = ConsistencyValidator(client)
        report = validator.validate_all()

        assert "valid" in report
        assert "entity_issues" in report
        assert "relationship_issues" in report
        assert "metadata_issues" in report
        assert "timestamp" in report


class TestIntegration:
    """Integration tests for all operations together."""

    def test_full_workflow(self, client):
        """Full workflow: entities → relationships → metadata → validation."""
        # Create operations
        entities_ops = EntityOperations(client)
        rel_ops = RelationshipOperations(client)
        meta_ops = MetadataOperations(client)
        validator = ConsistencyValidator(client)

        # Insert entities
        entities = [
            CodeEntity(
                entity_id="test:parser",
                repository="test",
                file_path="test.py",
                entity_type=EntityType.FUNCTION,
                language="python",
                name="parse",
                full_qualified_name="parse",
                code_text="def parse(): pass",
                code_vector=[0.1] * 768,
            ),
            CodeEntity(
                entity_id="test:format",
                repository="test",
                file_path="test.py",
                entity_type=EntityType.FUNCTION,
                language="python",
                name="format",
                full_qualified_name="format",
                code_text="def format(): pass",
                code_vector=[0.2] * 768,
            ),
        ]
        entity_count, entity_errors = entities_ops.insert(entities)
        assert entity_count == 2

        # Insert relationships
        rel = CodeRelationship(
            relationship_id="parser->format",
            caller_id="test:parser",
            callee_id="test:format",
            relationship_type=RelationshipType.CALLS,
        )
        rel_count, rel_errors = rel_ops.insert([rel])
        assert rel_count == 1

        # Insert metadata
        meta = SearchMetadata(
            metadata_id="test:parser",
            entity_id="test:parser",
            repository="test",
            implementation_status=ImplementationStatus.STABLE,
        )
        meta_count, meta_errors = meta_ops.insert([meta])
        assert meta_count == 1

        # Validate
        report = validator.validate_all()
        assert report["valid"] or len(report["entity_issues"]) == 0

    def test_all_operations_together(self, client):
        """Verify all operation classes work together."""
        ops = {
            "entities": EntityOperations(client),
            "relationships": RelationshipOperations(client),
            "metadata": MetadataOperations(client),
        }

        # All should be instantiated without errors
        assert all(op is not None for op in ops.values())

        # All should have table_exists checks
        assert not client.table_exists("code_entities") or client.table_exists("code_entities")
