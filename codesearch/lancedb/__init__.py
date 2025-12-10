"""LanceDB module for semantic code search."""

from codesearch.lancedb.models import (
    DEFAULT_EMBEDDING_DIMENSION,
    EntityType,
    Visibility,
    RelationshipType,
    ImplementationStatus,
    CodeEntity,
    CodeRelationship,
    SearchMetadata,
)
from codesearch.lancedb.client import LanceDBClient
from codesearch.lancedb.pool import DatabaseConnectionPool
from codesearch.lancedb.initialization import DatabaseInitializer
from codesearch.lancedb.backup import DatabaseBackupManager
from codesearch.lancedb.statistics import DatabaseStatistics
from codesearch.lancedb.optimization import DatabaseOptimizer

__all__ = [
    # Constants
    "DEFAULT_EMBEDDING_DIMENSION",
    # Models
    "EntityType",
    "Visibility",
    "RelationshipType",
    "ImplementationStatus",
    "CodeEntity",
    "CodeRelationship",
    "SearchMetadata",
    # Database Management
    "LanceDBClient",
    "DatabaseConnectionPool",
    "DatabaseInitializer",
    "DatabaseBackupManager",
    "DatabaseStatistics",
    "DatabaseOptimizer",
]
