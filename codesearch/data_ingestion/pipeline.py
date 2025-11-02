import uuid
from datetime import datetime
from typing import List
import time

from codesearch.models import CodeEntity, CodeRelationship, SearchMetadata
from codesearch.data_ingestion.models import IngestionResult, IngestionError
from codesearch.data_ingestion.deduplication import DeduplicationCache
from codesearch.data_ingestion.validation import IngestionValidator


class DataIngestionPipeline:
    """Main orchestrator for batch ingestion of code entities into LanceDB."""

    def __init__(self, client, batch_size: int = 1000):
        """Initialize pipeline with LanceDB client and batch settings.

        Args:
            client: LanceDB client instance
            batch_size: Maximum entities per batch (default 1000)
        """
        self.client = client
        self.batch_size = batch_size
        self.dedup_cache = DeduplicationCache(client)
        self.validator = IngestionValidator()

    def ingest_batch(self, entities: List[CodeEntity]) -> IngestionResult:
        """Ingest a batch of entities with coordinated relationships and metadata.

        Process flow:
        1. Check each entity against dedup cache
        2. Validate each entity
        3. Insert valid entities into code_entities table
        4. Extract and insert relationships if provided
        5. Generate and insert metadata records
        6. Return comprehensive IngestionResult

        Args:
            entities: List of CodeEntity objects to ingest

        Returns:
            IngestionResult with detailed metrics and errors
        """
        start_time = time.time()
        batch_id = str(uuid.uuid4())
        result = IngestionResult(batch_id=batch_id)

        # Step 1: Separate duplicates, updates, and new entities
        duplicates = []
        updates = []
        new_entities = []

        for entity in entities:
            if self.dedup_cache.is_duplicate(entity):
                duplicates.append(entity)
                result.skipped_count += 1
            else:
                new_entities.append(entity)

        # Step 2: Validate all entities
        valid_entities = []
        for entity in new_entities:
            validation_errors = self.validator.validate_entity(entity)
            if validation_errors:
                # Collect validation errors
                error_msg = "; ".join(str(e.reason) for e in validation_errors)
                result.add_error(IngestionError(
                    entity_id=entity.entity_id,
                    error_type="validation",
                    message=error_msg,
                    recoverable=True,
                    timestamp=datetime.utcnow()
                ))
            else:
                valid_entities.append(entity)

        # Step 3: Insert valid entities
        if valid_entities:
            try:
                # Get code_entities table
                code_entities_table = self.client.get_table("code_entities")

                # Convert entities to dict format for insertion
                entity_dicts = [self._entity_to_dict(e) for e in valid_entities]

                # Insert into database
                code_entities_table.add(entity_dicts)
                result.inserted_count = len(valid_entities)

                # Add hashes to cache for all successfully inserted entities
                for entity in valid_entities:
                    self.dedup_cache.add(entity)
            except Exception as e:
                result.add_error(IngestionError(
                    entity_id="batch",
                    error_type="insertion",
                    message=f"Batch insertion failed: {str(e)}",
                    recoverable=False,
                    timestamp=datetime.utcnow()
                ))
                # Mark all entities as failed
                result.inserted_count = 0
                result.failed_count += len(valid_entities)

        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000
        result.duration_ms = duration_ms

        return result

    def _entity_to_dict(self, entity: CodeEntity) -> dict:
        """Convert CodeEntity to dictionary for database insertion.

        Args:
            entity: CodeEntity to convert

        Returns:
            Dictionary representation of entity
        """
        return {
            "entity_id": entity.entity_id,
            "name": entity.name,
            "code_text": entity.code_text,
            "code_vector": entity.code_vector,
            "language": entity.language,
            "entity_type": entity.entity_type,
            "repository": entity.repository,
            "file_path": entity.file_path,
            "start_line": entity.start_line,
            "end_line": entity.end_line,
            "visibility": entity.visibility,
            "source_hash": entity.source_hash,
        }

    def ingest_relationships(self, relationships: List[CodeRelationship]) -> IngestionResult:
        """Ingest relationships separately if needed.

        Args:
            relationships: List of CodeRelationship objects to ingest

        Returns:
            IngestionResult with relationship metrics
        """
        start_time = time.time()
        batch_id = str(uuid.uuid4())
        result = IngestionResult(batch_id=batch_id)

        # Validate all relationships
        valid_relationships = []
        for rel in relationships:
            validation_errors = self.validator.validate_relationship(rel)
            if validation_errors:
                error_msg = "; ".join(str(e.reason) for e in validation_errors)
                result.add_error(IngestionError(
                    entity_id=f"{rel.caller_id}->{rel.callee_id}",
                    error_type="relationship",
                    message=error_msg,
                    recoverable=True,
                    timestamp=datetime.utcnow()
                ))
                result.relationships_failed += 1
            else:
                valid_relationships.append(rel)

        # Insert valid relationships
        if valid_relationships:
            try:
                code_relationships_table = self.client.get_table("code_relationships")

                rel_dicts = [self._relationship_to_dict(r) for r in valid_relationships]
                code_relationships_table.add(rel_dicts)

                result.relationships_inserted = len(valid_relationships)
            except Exception as e:
                result.add_error(IngestionError(
                    entity_id="batch-relationships",
                    error_type="insertion",
                    message=f"Relationship insertion failed: {str(e)}",
                    recoverable=False,
                    timestamp=datetime.utcnow()
                ))
                result.relationships_inserted = 0
                result.relationships_failed += len(valid_relationships)

        duration_ms = (time.time() - start_time) * 1000
        result.duration_ms = duration_ms

        return result

    def _relationship_to_dict(self, rel: CodeRelationship) -> dict:
        """Convert CodeRelationship to dictionary for database insertion.

        Args:
            rel: CodeRelationship to convert

        Returns:
            Dictionary representation of relationship
        """
        return {
            "caller_id": rel.caller_id,
            "callee_id": rel.callee_id,
            "relationship_type": rel.relationship_type,
        }

    def ingest_metadata(self, metadata_list: List[SearchMetadata]) -> IngestionResult:
        """Ingest metadata separately if needed.

        Args:
            metadata_list: List of SearchMetadata objects to ingest

        Returns:
            IngestionResult with metadata metrics
        """
        result = IngestionResult()
        result.batch_id = str(uuid.uuid4())

        # TODO: Implement metadata ingestion

        return result

    def rollback_batch(self, batch_id: str) -> bool:
        """Rollback a previously ingested batch using audit logs.

        Args:
            batch_id: UUID of batch to rollback

        Returns:
            True if rollback successful, False otherwise
        """
        # TODO: Implement rollback using audit trail
        return False
