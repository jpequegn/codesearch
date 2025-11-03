from dataclasses import dataclass
from typing import List
from codesearch.models import CodeEntity, CodeRelationship, SearchMetadata, EntityType, Visibility


@dataclass
class ValidationError:
    """Single validation error for an entity or relationship."""
    field: str
    reason: str
    severity: str  # "error" or "warning"


class IngestionValidator:
    """Pre-insertion validation for all entities, relationships, and metadata."""

    VECTOR_DIMENSIONS = 768
    VECTOR_MIN = -2.0
    VECTOR_MAX = 2.0

    def validate_entity(self, entity: CodeEntity) -> List[ValidationError]:
        """Validate a CodeEntity before insertion.

        Args:
            entity: CodeEntity to validate

        Returns:
            List of ValidationError objects (empty if valid)
        """
        errors = []

        # entity_id: non-null, non-empty format
        if not entity.entity_id or not isinstance(entity.entity_id, str):
            errors.append(ValidationError(
                field="entity_id",
                reason="entity_id must be non-empty string",
                severity="error"
            ))

        # code_vector: 768 dimensions
        if not isinstance(entity.code_vector, (list, tuple)):
            errors.append(ValidationError(
                field="code_vector",
                reason="code_vector must be a list or tuple",
                severity="error"
            ))
        elif len(entity.code_vector) != self.VECTOR_DIMENSIONS:
            errors.append(ValidationError(
                field="code_vector",
                reason=f"code_vector must have {self.VECTOR_DIMENSIONS} dimensions, got {len(entity.code_vector)}",
                severity="error"
            ))
        else:
            # Check vector values are in [-2.0, 2.0] range
            if any(v < self.VECTOR_MIN or v > self.VECTOR_MAX for v in entity.code_vector):
                errors.append(ValidationError(
                    field="code_vector",
                    reason=f"code_vector values must be in range [{self.VECTOR_MIN}, {self.VECTOR_MAX}]",
                    severity="error"
                ))

        # Required fields
        required_fields = ["name", "code_text", "language", "entity_type"]
        for field in required_fields:
            value = getattr(entity, field, None)
            if not value or (isinstance(value, str) and not value.strip()):
                errors.append(ValidationError(
                    field=field,
                    reason=f"{field} is required and cannot be empty",
                    severity="error"
                ))

        # Valid enum values
        try:
            EntityType(entity.entity_type)
        except (ValueError, AttributeError):
            errors.append(ValidationError(
                field="entity_type",
                reason=f"entity_type '{entity.entity_type}' is not a valid EntityType",
                severity="error"
            ))

        try:
            Visibility(entity.visibility)
        except (ValueError, AttributeError):
            errors.append(ValidationError(
                field="visibility",
                reason=f"visibility '{entity.visibility}' is not a valid Visibility",
                severity="error"
            ))

        return errors

    def validate_relationship(self, relationship: CodeRelationship) -> List[ValidationError]:
        """Validate a CodeRelationship before insertion.

        Args:
            relationship: CodeRelationship to validate

        Returns:
            List of ValidationError objects (empty if valid)
        """
        errors = []

        # Both IDs must exist and be non-empty
        if not relationship.caller_id or not isinstance(relationship.caller_id, str):
            errors.append(ValidationError(
                field="caller_id",
                reason="caller_id must be non-empty string",
                severity="error"
            ))

        if not relationship.callee_id or not isinstance(relationship.callee_id, str):
            errors.append(ValidationError(
                field="callee_id",
                reason="callee_id must be non-empty string",
                severity="error"
            ))

        # No self-referential relationships
        if relationship.caller_id == relationship.callee_id:
            errors.append(ValidationError(
                field="relationship",
                reason="Self-referential relationships (caller_id == callee_id) are not allowed",
                severity="error"
            ))

        return errors

    def validate_metadata(self, metadata: SearchMetadata, entity_id: str) -> List[ValidationError]:
        """Validate SearchMetadata before insertion.

        Args:
            metadata: SearchMetadata to validate
            entity_id: Expected entity_id for this metadata

        Returns:
            List of ValidationError objects (empty if valid)
        """
        errors = []

        # metadata_id should match entity_id
        if metadata.metadata_id != entity_id:
            errors.append(ValidationError(
                field="metadata_id",
                reason=f"metadata_id must match entity_id (expected '{entity_id}', got '{metadata.metadata_id}')",
                severity="error"
            ))

        # entity_id in metadata should be set
        if not metadata.entity_id:
            errors.append(ValidationError(
                field="entity_id",
                reason="metadata.entity_id must be set",
                severity="error"
            ))

        return errors
