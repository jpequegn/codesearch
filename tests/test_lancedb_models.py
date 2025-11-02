"""Tests for LanceDB data models."""

import pytest
from datetime import datetime
from codesearch.lancedb.models import (
    EntityType,
    Visibility,
    RelationshipType,
    ImplementationStatus,
    CodeEntity,
    CodeRelationship,
    SearchMetadata,
)


class TestEntityType:
    """Test EntityType enum."""

    def test_entity_type_values(self):
        """EntityType has correct values."""
        assert EntityType.FUNCTION.value == "function"
        assert EntityType.CLASS.value == "class"
        assert EntityType.FILE.value == "file"
        assert EntityType.SNIPPET.value == "snippet"

    def test_entity_type_members(self):
        """EntityType has all expected members."""
        members = {e.name for e in EntityType}
        assert members == {"FUNCTION", "CLASS", "FILE", "SNIPPET"}


class TestVisibility:
    """Test Visibility enum."""

    def test_visibility_values(self):
        """Visibility has correct values."""
        assert Visibility.PUBLIC.value == "public"
        assert Visibility.PRIVATE.value == "private"
        assert Visibility.INTERNAL.value == "internal"
        assert Visibility.TEST.value == "test"

    def test_visibility_members(self):
        """Visibility has all expected members."""
        members = {v.name for v in Visibility}
        assert members == {"PUBLIC", "PRIVATE", "INTERNAL", "TEST"}


class TestRelationshipType:
    """Test RelationshipType enum."""

    def test_relationship_type_values(self):
        """RelationshipType has correct values."""
        assert RelationshipType.CALLS.value == "calls"
        assert RelationshipType.INHERITS_FROM.value == "inherits_from"
        assert RelationshipType.IMPORTS.value == "imports"
        assert RelationshipType.USES.value == "uses"

    def test_relationship_type_members(self):
        """RelationshipType has all expected members."""
        members = {r.name for r in RelationshipType}
        assert members == {"CALLS", "INHERITS_FROM", "IMPORTS", "USES"}


class TestImplementationStatus:
    """Test ImplementationStatus enum."""

    def test_implementation_status_values(self):
        """ImplementationStatus has correct values."""
        assert ImplementationStatus.STABLE.value == "stable"
        assert ImplementationStatus.EXPERIMENTAL.value == "experimental"
        assert ImplementationStatus.DEPRECATED.value == "deprecated"

    def test_implementation_status_members(self):
        """ImplementationStatus has all expected members."""
        members = {s.name for s in ImplementationStatus}
        assert members == {"STABLE", "EXPERIMENTAL", "DEPRECATED"}


class TestCodeEntity:
    """Test CodeEntity model."""

    def test_code_entity_creation(self):
        """CodeEntity can be created with required fields."""
        entity = CodeEntity(
            entity_id="repo:file.py:Parser:parse",
            repository="pytorch",
            file_path="/path/to/file.py",
            entity_type=EntityType.FUNCTION,
            language="python",
            name="parse",
            full_qualified_name="Parser.parse",
            code_text="def parse(self): pass",
            code_vector=[0.5] * 768,
        )

        assert entity.entity_id == "repo:file.py:Parser:parse"
        assert entity.repository == "pytorch"
        assert entity.language == "python"
        assert len(entity.code_vector) == 768

    def test_code_entity_default_values(self):
        """CodeEntity has sensible defaults."""
        entity = CodeEntity(
            entity_id="test:func",
            repository="test",
            file_path="test.py",
            entity_type=EntityType.FUNCTION,
            language="python",
            name="func",
            full_qualified_name="func",
            code_text="def func(): pass",
            code_vector=[0.1] * 768,
        )

        assert entity.visibility == Visibility.PUBLIC
        assert entity.complexity == 0
        assert entity.line_count == 0
        assert entity.keyword_tags == []
        assert entity.user_tags == []

    def test_code_entity_to_dict(self):
        """CodeEntity.to_dict() serialization works correctly."""
        entity = CodeEntity(
            entity_id="test:func",
            repository="test",
            file_path="test.py",
            entity_type=EntityType.FUNCTION,
            language="python",
            name="func",
            full_qualified_name="func",
            code_text="def func(): pass",
            code_vector=[0.1] * 768,
            visibility=Visibility.PRIVATE,
            complexity=3,
            line_count=5,
            keyword_tags=["async", "database"],
        )

        data = entity.to_dict()

        assert data["entity_id"] == "test:func"
        assert data["entity_type"] == "function"
        assert data["visibility"] == "private"
        assert data["complexity"] == 3
        assert data["line_count"] == 5
        assert data["keyword_tags"] == ["async", "database"]
        assert len(data["code_vector"]) == 768
        assert data["created_at"]  # ISO format timestamp
        assert data["updated_at"]  # ISO format timestamp

    def test_code_entity_vector_validation(self):
        """CodeEntity accepts 768-dimensional vectors."""
        entity = CodeEntity(
            entity_id="test",
            repository="test",
            file_path="test.py",
            entity_type=EntityType.FUNCTION,
            language="python",
            name="test",
            full_qualified_name="test",
            code_text="test",
            code_vector=[0.5] * 768,
        )
        assert len(entity.code_vector) == 768

    def test_code_entity_optional_fields(self):
        """CodeEntity handles optional fields correctly."""
        entity = CodeEntity(
            entity_id="test",
            repository="test",
            file_path="test.py",
            entity_type=EntityType.FILE,
            language="python",
            name="file",
            full_qualified_name="file",
            code_text="content",
            code_vector=[0.5] * 768,
            docstring="A file",
            class_name="ParentClass",
            argument_count=3,
            return_type="str",
        )

        assert entity.docstring == "A file"
        assert entity.class_name == "ParentClass"
        assert entity.argument_count == 3
        assert entity.return_type == "str"


class TestCodeRelationship:
    """Test CodeRelationship model."""

    def test_code_relationship_creation(self):
        """CodeRelationship can be created."""
        rel = CodeRelationship(
            relationship_id="caller->callee",
            caller_id="repo:file.py:Parser:parse",
            callee_id="repo:file.py:Utils:format",
            relationship_type=RelationshipType.CALLS,
        )

        assert rel.relationship_id == "caller->callee"
        assert rel.caller_id == "repo:file.py:Parser:parse"
        assert rel.callee_id == "repo:file.py:Utils:format"
        assert rel.relationship_type == RelationshipType.CALLS

    def test_code_relationship_default_values(self):
        """CodeRelationship has sensible defaults."""
        rel = CodeRelationship(
            relationship_id="test",
            caller_id="caller",
            callee_id="callee",
            relationship_type=RelationshipType.CALLS,
        )

        assert rel.call_count == 0
        assert rel.call_context is None
        assert rel.is_indirect is False
        assert rel.is_test_only is False

    def test_code_relationship_to_dict(self):
        """CodeRelationship.to_dict() serialization works correctly."""
        rel = CodeRelationship(
            relationship_id="test->func",
            caller_id="repo:a.py:test_func",
            callee_id="repo:b.py:target_func",
            relationship_type=RelationshipType.CALLS,
            call_count=5,
            call_context="main_loop",
            is_indirect=True,
            is_test_only=False,
        )

        data = rel.to_dict()

        assert data["relationship_id"] == "test->func"
        assert data["caller_id"] == "repo:a.py:test_func"
        assert data["callee_id"] == "repo:b.py:target_func"
        assert data["relationship_type"] == "calls"
        assert data["call_count"] == 5
        assert data["call_context"] == "main_loop"
        assert data["is_indirect"] is True
        assert data["is_test_only"] is False

    def test_code_relationship_all_types(self):
        """CodeRelationship supports all relationship types."""
        for rel_type in RelationshipType:
            rel = CodeRelationship(
                relationship_id="test",
                caller_id="a",
                callee_id="b",
                relationship_type=rel_type,
            )
            assert rel.relationship_type == rel_type


class TestSearchMetadata:
    """Test SearchMetadata model."""

    def test_search_metadata_creation(self):
        """SearchMetadata can be created."""
        metadata = SearchMetadata(
            metadata_id="repo:file.py:Parser:parse",
            entity_id="repo:file.py:Parser:parse",
            repository="pytorch",
        )

        assert metadata.metadata_id == "repo:file.py:Parser:parse"
        assert metadata.entity_id == "repo:file.py:Parser:parse"
        assert metadata.repository == "pytorch"

    def test_search_metadata_default_values(self):
        """SearchMetadata has sensible defaults."""
        metadata = SearchMetadata(
            metadata_id="test",
            entity_id="test",
            repository="test",
        )

        assert metadata.cyclomatic_complexity == 0
        assert metadata.test_coverage_percent is None
        assert metadata.last_modified_days_ago == 0
        assert metadata.feature_area is None
        assert metadata.implementation_status == ImplementationStatus.STABLE
        assert metadata.search_keywords == ""
        assert metadata.referenced_by_count == 0
        assert metadata.reference_count == 0
        assert metadata.is_hot_path is False
        assert metadata.is_public_api is False

    def test_search_metadata_to_dict(self):
        """SearchMetadata.to_dict() serialization works correctly."""
        metadata = SearchMetadata(
            metadata_id="test",
            entity_id="test",
            repository="pytorch",
            repository_description="PyTorch ML framework",
            cyclomatic_complexity=5,
            test_coverage_percent=92.5,
            last_modified_days_ago=3,
            feature_area="parsing",
            implementation_status=ImplementationStatus.EXPERIMENTAL,
            search_keywords="parser tokenizer lexer",
            referenced_by_count=15,
            reference_count=8,
            is_hot_path=True,
            is_public_api=True,
        )

        data = metadata.to_dict()

        assert data["metadata_id"] == "test"
        assert data["entity_id"] == "test"
        assert data["repository"] == "pytorch"
        assert data["repository_description"] == "PyTorch ML framework"
        assert data["cyclomatic_complexity"] == 5
        assert data["test_coverage_percent"] == 92.5
        assert data["last_modified_days_ago"] == 3
        assert data["feature_area"] == "parsing"
        assert data["implementation_status"] == "experimental"
        assert data["search_keywords"] == "parser tokenizer lexer"
        assert data["referenced_by_count"] == 15
        assert data["reference_count"] == 8
        assert data["is_hot_path"] is True
        assert data["is_public_api"] is True

    def test_search_metadata_all_statuses(self):
        """SearchMetadata supports all implementation statuses."""
        for status in ImplementationStatus:
            metadata = SearchMetadata(
                metadata_id="test",
                entity_id="test",
                repository="test",
                implementation_status=status,
            )
            assert metadata.implementation_status == status


class TestSerializationIntegration:
    """Test serialization of all models together."""

    def test_code_entity_serialization_round_trip(self):
        """CodeEntity serializes and key fields are present."""
        entity = CodeEntity(
            entity_id="repo:file.py:Parser:parse",
            repository="pytorch",
            file_path="/path/to/file.py",
            entity_type=EntityType.FUNCTION,
            language="python",
            name="parse",
            full_qualified_name="Parser.parse",
            code_text="def parse(self): pass",
            code_vector=[0.1] * 768,
            visibility=Visibility.PUBLIC,
            complexity=2,
        )

        data = entity.to_dict()

        # Verify all required fields are present
        assert "entity_id" in data
        assert "code_vector" in data
        assert "entity_type" in data
        assert "visibility" in data
        assert "complexity" in data
        assert data["code_vector"] == [0.1] * 768

    def test_code_relationship_serialization_round_trip(self):
        """CodeRelationship serializes correctly."""
        rel = CodeRelationship(
            relationship_id="test->target",
            caller_id="repo:a.py:func",
            callee_id="repo:b.py:func",
            relationship_type=RelationshipType.CALLS,
            call_count=3,
        )

        data = rel.to_dict()

        assert "relationship_id" in data
        assert "caller_id" in data
        assert "callee_id" in data
        assert "relationship_type" in data
        assert data["call_count"] == 3

    def test_search_metadata_serialization_round_trip(self):
        """SearchMetadata serializes correctly."""
        metadata = SearchMetadata(
            metadata_id="test",
            entity_id="test",
            repository="pytorch",
            implementation_status=ImplementationStatus.STABLE,
            referenced_by_count=10,
        )

        data = metadata.to_dict()

        assert "metadata_id" in data
        assert "entity_id" in data
        assert "repository" in data
        assert "implementation_status" in data
        assert data["referenced_by_count"] == 10
