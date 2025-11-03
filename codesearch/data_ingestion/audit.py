from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class AuditRecord:
    """Single audit trail entry for tracking ingestion operations."""

    batch_id: str
    timestamp: datetime
    operation: str  # "insert", "update", "delete", "rollback"
    entity_id: str
    table: str  # "code_entities", "code_relationships", "search_metadata"
    old_value: Optional[dict] = None
    new_value: Optional[dict] = None


class AuditTrail:
    """Track all ingestion operations for audit and rollback capabilities."""

    def __init__(self, client):
        """Initialize audit trail with LanceDB client.

        Args:
            client: LanceDB client instance
        """
        self.client = client
        self.records: List[AuditRecord] = []

    def record_insert(self, batch_id: str, entity_id: str, table: str,
                     new_value: Optional[dict] = None) -> None:
        """Record an insert operation.

        Args:
            batch_id: Batch identifier for grouping operations
            entity_id: ID of entity being inserted
            table: Table name (code_entities, code_relationships, search_metadata)
            new_value: Optional dictionary of inserted values
        """
        record = AuditRecord(
            batch_id=batch_id,
            timestamp=datetime.utcnow(),
            operation="insert",
            entity_id=entity_id,
            table=table,
            new_value=new_value
        )
        self.records.append(record)
        self._persist_to_database(record)

    def record_update(self, batch_id: str, entity_id: str, table: str,
                     old_value: Optional[dict] = None,
                     new_value: Optional[dict] = None) -> None:
        """Record an update operation.

        Args:
            batch_id: Batch identifier for grouping operations
            entity_id: ID of entity being updated
            table: Table name
            old_value: Optional dictionary of old values
            new_value: Optional dictionary of new values
        """
        record = AuditRecord(
            batch_id=batch_id,
            timestamp=datetime.utcnow(),
            operation="update",
            entity_id=entity_id,
            table=table,
            old_value=old_value,
            new_value=new_value
        )
        self.records.append(record)
        self._persist_to_database(record)

    def record_delete(self, batch_id: str, entity_id: str, table: str,
                     old_value: Optional[dict] = None) -> None:
        """Record a delete operation.

        Args:
            batch_id: Batch identifier for grouping operations
            entity_id: ID of entity being deleted
            table: Table name
            old_value: Optional dictionary of deleted values
        """
        record = AuditRecord(
            batch_id=batch_id,
            timestamp=datetime.utcnow(),
            operation="delete",
            entity_id=entity_id,
            table=table,
            old_value=old_value
        )
        self.records.append(record)
        self._persist_to_database(record)

    def get_batch_records(self, batch_id: str) -> List[AuditRecord]:
        """Retrieve all records for a specific batch.

        Args:
            batch_id: Batch identifier to retrieve

        Returns:
            List of AuditRecord objects for this batch
        """
        return [r for r in self.records if r.batch_id == batch_id]

    def _persist_to_database(self, record: AuditRecord) -> None:
        """Persist audit record to database.

        Args:
            record: AuditRecord to persist
        """
        # TODO: Implement database persistence
        pass
