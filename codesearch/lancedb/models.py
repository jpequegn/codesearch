"""LanceDB schema data models and PyArrow schemas for table creation."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional

import pyarrow as pa

# Vector dimension for CodeBERT embeddings
EMBEDDING_DIMENSION = 768


# Enums for categorical fields
class EntityType(str, Enum):
    """Code entity types."""
    FUNCTION = "function"
    CLASS = "class"
    FILE = "file"
    SNIPPET = "snippet"


class Visibility(str, Enum):
    """Code visibility levels."""
    PUBLIC = "public"
    PRIVATE = "private"
    INTERNAL = "internal"
    TEST = "test"


class RelationshipType(str, Enum):
    """Relationship types in call graph."""
    CALLS = "calls"
    INHERITS_FROM = "inherits_from"
    IMPORTS = "imports"
    USES = "uses"


class ImplementationStatus(str, Enum):
    """Code implementation status."""
    STABLE = "stable"
    EXPERIMENTAL = "experimental"
    DEPRECATED = "deprecated"


# Core data models
@dataclass
class CodeEntity:
    """Represents a code entity (function, class, file, snippet)."""

    # Identifiers
    entity_id: str
    repository: str
    file_path: str
    entity_type: EntityType
    language: str

    # Content
    name: str
    full_qualified_name: str
    code_text: str
    docstring: Optional[str] = None

    # Embedding
    code_vector: List[float] = field(default_factory=list)  # 768-dimensional

    # Metadata
    visibility: Visibility = Visibility.PUBLIC
    class_name: Optional[str] = None

    # Code properties
    complexity: int = 0
    line_count: int = 0
    argument_count: Optional[int] = None
    return_type: Optional[str] = None

    # Tags
    keyword_tags: List[str] = field(default_factory=list)
    user_tags: List[str] = field(default_factory=list)

    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source_hash: str = ""  # SHA256

    def to_dict(self) -> dict:
        """Convert to dictionary for LanceDB insertion."""
        return {
            "entity_id": self.entity_id,
            "repository": self.repository,
            "file_path": self.file_path,
            "entity_type": self.entity_type.value,
            "language": self.language,
            "name": self.name,
            "full_qualified_name": self.full_qualified_name,
            "code_text": self.code_text,
            "docstring": self.docstring,
            "code_vector": self.code_vector,
            "visibility": self.visibility.value,
            "class_name": self.class_name,
            "complexity": self.complexity,
            "line_count": self.line_count,
            "argument_count": self.argument_count,
            "return_type": self.return_type,
            "keyword_tags": self.keyword_tags,
            "user_tags": self.user_tags,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "source_hash": self.source_hash,
        }


@dataclass
class CodeRelationship:
    """Represents a relationship in the call graph."""

    # Identifiers
    relationship_id: str
    caller_id: str
    callee_id: str
    relationship_type: RelationshipType

    # Context
    call_count: int = 0
    call_context: Optional[str] = None

    # Analysis
    is_indirect: bool = False
    is_test_only: bool = False

    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        """Convert to dictionary for LanceDB insertion."""
        return {
            "relationship_id": self.relationship_id,
            "caller_id": self.caller_id,
            "callee_id": self.callee_id,
            "relationship_type": self.relationship_type.value,
            "call_count": self.call_count,
            "call_context": self.call_context,
            "is_indirect": self.is_indirect,
            "is_test_only": self.is_test_only,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class SearchMetadata:
    """Metadata for search and filtering."""

    # Identifiers
    metadata_id: str
    entity_id: str
    repository: str

    # Documentation
    repository_description: Optional[str] = None

    # Analysis
    cyclomatic_complexity: int = 0
    test_coverage_percent: Optional[float] = None
    last_modified_days_ago: int = 0

    # Categorization
    feature_area: Optional[str] = None
    implementation_status: ImplementationStatus = ImplementationStatus.STABLE

    # Searchability
    search_keywords: str = ""
    referenced_by_count: int = 0
    reference_count: int = 0

    # Performance hints
    is_hot_path: bool = False
    is_public_api: bool = False

    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        """Convert to dictionary for LanceDB insertion."""
        return {
            "metadata_id": self.metadata_id,
            "entity_id": self.entity_id,
            "repository": self.repository,
            "repository_description": self.repository_description,
            "cyclomatic_complexity": self.cyclomatic_complexity,
            "test_coverage_percent": self.test_coverage_percent,
            "last_modified_days_ago": self.last_modified_days_ago,
            "feature_area": self.feature_area,
            "implementation_status": self.implementation_status.value,
            "search_keywords": self.search_keywords,
            "referenced_by_count": self.referenced_by_count,
            "reference_count": self.reference_count,
            "is_hot_path": self.is_hot_path,
            "is_public_api": self.is_public_api,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


# PyArrow Schemas for LanceDB table creation
# These define the exact schema for each table including vector columns

def get_code_entities_schema() -> pa.Schema:
    """Get PyArrow schema for code_entities table.

    Returns:
        PyArrow schema with vector column for similarity search.
    """
    return pa.schema([
        pa.field("entity_id", pa.string(), nullable=False),
        pa.field("repository", pa.string(), nullable=False),
        pa.field("file_path", pa.string(), nullable=False),
        pa.field("entity_type", pa.string(), nullable=False),
        pa.field("language", pa.string(), nullable=False),
        pa.field("name", pa.string(), nullable=False),
        pa.field("full_qualified_name", pa.string(), nullable=False),
        pa.field("code_text", pa.string(), nullable=False),
        pa.field("docstring", pa.string(), nullable=True),
        pa.field("code_vector", pa.list_(pa.float32(), EMBEDDING_DIMENSION), nullable=False),
        pa.field("visibility", pa.string(), nullable=False),
        pa.field("class_name", pa.string(), nullable=True),
        pa.field("complexity", pa.int32(), nullable=False),
        pa.field("line_count", pa.int32(), nullable=False),
        pa.field("argument_count", pa.int32(), nullable=True),
        pa.field("return_type", pa.string(), nullable=True),
        pa.field("keyword_tags", pa.list_(pa.string()), nullable=False),
        pa.field("user_tags", pa.list_(pa.string()), nullable=False),
        pa.field("created_at", pa.string(), nullable=False),
        pa.field("updated_at", pa.string(), nullable=False),
        pa.field("source_hash", pa.string(), nullable=False),
    ])


def get_code_relationships_schema() -> pa.Schema:
    """Get PyArrow schema for code_relationships table.

    Returns:
        PyArrow schema for relationship storage.
    """
    return pa.schema([
        pa.field("relationship_id", pa.string(), nullable=False),
        pa.field("caller_id", pa.string(), nullable=False),
        pa.field("callee_id", pa.string(), nullable=False),
        pa.field("relationship_type", pa.string(), nullable=False),
        pa.field("call_count", pa.int32(), nullable=False),
        pa.field("call_context", pa.string(), nullable=True),
        pa.field("is_indirect", pa.bool_(), nullable=False),
        pa.field("is_test_only", pa.bool_(), nullable=False),
        pa.field("created_at", pa.string(), nullable=False),
        pa.field("updated_at", pa.string(), nullable=False),
    ])


def get_search_metadata_schema() -> pa.Schema:
    """Get PyArrow schema for search_metadata table.

    Returns:
        PyArrow schema for search metadata storage.
    """
    return pa.schema([
        pa.field("metadata_id", pa.string(), nullable=False),
        pa.field("entity_id", pa.string(), nullable=False),
        pa.field("repository", pa.string(), nullable=False),
        pa.field("repository_description", pa.string(), nullable=True),
        pa.field("cyclomatic_complexity", pa.int32(), nullable=False),
        pa.field("test_coverage_percent", pa.float32(), nullable=True),
        pa.field("last_modified_days_ago", pa.int32(), nullable=False),
        pa.field("feature_area", pa.string(), nullable=True),
        pa.field("implementation_status", pa.string(), nullable=False),
        pa.field("search_keywords", pa.string(), nullable=False),
        pa.field("referenced_by_count", pa.int32(), nullable=False),
        pa.field("reference_count", pa.int32(), nullable=False),
        pa.field("is_hot_path", pa.bool_(), nullable=False),
        pa.field("is_public_api", pa.bool_(), nullable=False),
        pa.field("created_at", pa.string(), nullable=False),
        pa.field("updated_at", pa.string(), nullable=False),
    ])
