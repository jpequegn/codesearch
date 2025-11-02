import pytest
from codesearch.data_ingestion.validation import IngestionValidator, ValidationError
from codesearch.models import CodeEntity, CodeRelationship, SearchMetadata


@pytest.fixture
def validator():
    """Create validator instance."""
    return IngestionValidator()


@pytest.fixture
def valid_entity():
    """Create a valid CodeEntity."""
    return CodeEntity(
        entity_id="repo:file.py:Parser:parse",
        name="parse",
        code_text="def parse(self): return self.data",
        code_vector=[0.1] * 768,
        language="python",
        entity_type="function",
        repository="repo",
        file_path="file.py",
        start_line=10,
        end_line=12,
        visibility="public",
        source_hash="abc123"
    )


def test_validator_initialization(validator):
    """Test that validator initializes without errors."""
    assert validator is not None


def test_validate_entity_with_valid_entity(validator, valid_entity):
    """Test that valid entity passes validation."""
    errors = validator.validate_entity(valid_entity)
    assert errors == []


def test_validate_entity_missing_entity_id(validator):
    """Test validation catches missing entity_id."""
    invalid_entity = CodeEntity(
        entity_id="",  # Empty entity_id
        name="parse",
        code_text="def parse(self): pass",
        code_vector=[0.1] * 768,
        language="python",
        entity_type="function",
        repository="repo",
        file_path="file.py",
        start_line=10,
        end_line=12,
        visibility="public",
        source_hash="abc123"
    )

    errors = validator.validate_entity(invalid_entity)
    assert len(errors) > 0
    assert any("entity_id" in str(e) for e in errors)


def test_validate_entity_wrong_vector_dimensions(validator):
    """Test validation catches wrong vector dimensions."""
    invalid_entity = CodeEntity(
        entity_id="repo:file.py:Parser:parse",
        name="parse",
        code_text="def parse(self): pass",
        code_vector=[0.1] * 512,  # Wrong: should be 768
        language="python",
        entity_type="function",
        repository="repo",
        file_path="file.py",
        start_line=10,
        end_line=12,
        visibility="public",
        source_hash="abc123"
    )

    errors = validator.validate_entity(invalid_entity)
    assert len(errors) > 0
    assert any("code_vector" in str(e) for e in errors)


def test_validate_entity_vector_values_out_of_range(validator):
    """Test validation catches out-of-range vector values."""
    invalid_entity = CodeEntity(
        entity_id="repo:file.py:Parser:parse",
        name="parse",
        code_text="def parse(self): pass",
        code_vector=[3.0] * 768,  # Out of range: should be [-2.0, 2.0]
        language="python",
        entity_type="function",
        repository="repo",
        file_path="file.py",
        start_line=10,
        end_line=12,
        visibility="public",
        source_hash="abc123"
    )

    errors = validator.validate_entity(invalid_entity)
    assert len(errors) > 0
    assert any("range" in str(e) or "bounds" in str(e) for e in errors)


def test_validate_relationship_valid(validator):
    """Test that valid relationship passes validation."""
    relationship = CodeRelationship(
        caller_id="repo:file.py:Parser:parse",
        callee_id="repo:file.py:Utils:format",
        relationship_type="calls"
    )

    errors = validator.validate_relationship(relationship)
    assert errors == []


def test_validate_relationship_self_referential(validator):
    """Test validation catches self-referential relationships."""
    relationship = CodeRelationship(
        caller_id="repo:file.py:Parser:parse",
        callee_id="repo:file.py:Parser:parse",  # Same as caller
        relationship_type="calls"
    )

    errors = validator.validate_relationship(relationship)
    assert len(errors) > 0
    assert any("Self-referential" in str(e) for e in errors)


def test_validate_metadata_valid(validator):
    """Test that valid metadata passes validation."""
    entity_id = "repo:file.py:Parser:parse"
    metadata = SearchMetadata(
        metadata_id=entity_id,
        entity_id=entity_id,
        repository="repo",
        file_path="file.py"
    )

    errors = validator.validate_metadata(metadata, entity_id)
    assert errors == []


def test_validate_metadata_id_mismatch(validator):
    """Test validation catches metadata_id mismatch."""
    entity_id = "repo:file.py:Parser:parse"
    metadata = SearchMetadata(
        metadata_id="wrong_id",  # Doesn't match entity_id
        entity_id=entity_id,
        repository="repo",
        file_path="file.py"
    )

    errors = validator.validate_metadata(metadata, entity_id)
    assert len(errors) > 0
    assert any("metadata_id" in str(e) for e in errors)
